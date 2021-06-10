import multiprocessing
import os
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Tuple, Optional, List, Dict

import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from .engines.base import XZTrainerEngine, XZTrainerEngineConfig


class SchedulerType(Enum):
    STEP = 'step'
    EPOCH = 'epoch'


class SavePolicy(Enum):
    NEVER = 'never'
    LAST_EPOCH = 'last_epoch'
    EVERY_EPOCH = 'every_epoch'


@dataclass
class XZTrainerConfig:
    batch_size: int
    batch_size_eval: int
    epochs: int
    optimizer: Callable[[nn.Module], Optimizer]

    experiment_name: str = 'master'
    gradient_clipping: bool = True
    metrics: Dict[str, Callable[[List, List], object]] = field(default_factory=dict)
    scheduler: Optional[Callable[[Optimizer, int], Tuple[object, SchedulerType]]] = None
    shuffle_train_dataset: bool = False
    dataloader_num_workers: int = multiprocessing.cpu_count()
    accumulation_steps: int = 1
    print_steps: int = 100
    save_policy: SavePolicy = SavePolicy.EVERY_EPOCH
    save_dir: str = 'checkpoint'
    use_tpu: bool = False


def _convert_model_outputs(outs):
    if isinstance(outs, Tensor):
        outs = outs.detach().tolist()
    elif isinstance(outs, dict):
        mp = []
        for k, v in outs.items():
            if isinstance(v, Tensor):
                v = v.detach().tolist()
            outs[k] = v
        for i in range(len(outs[next(iter(outs))])):
            mm = {}
            for k in outs:
                mm[k] = outs[k][i]
            mp.append(mm)
        outs = mp
    return outs


class XZTrainer(metaclass=ABCMeta):
    config: XZTrainerConfig
    engine: XZTrainerEngine

    def __init__(self, config: XZTrainerConfig, engine_cfg: XZTrainerEngineConfig, model, device=None):
        self.config = config

        if device is None:
            device = torch.device('cuda')
        self.device = device

        self.model = model.to(device)

        self.engine = engine_cfg.create_engine(self)

    def _create_dataloader(self, data, **kwargs):
        return DataLoader(data, num_workers=self.config.dataloader_num_workers, **kwargs)

    def _prepare_training(self, train_data):
        if self.config.use_tpu:
            raise NotImplementedError("TPU training currently isn't available, sorry guys :(")
        model = self.model
        total_steps = int(math.ceil(len(train_data) / self.config.batch_size)) // self.config.accumulation_steps * self.config.epochs
        dataloader = self._create_dataloader(train_data, batch_size=self.config.batch_size, shuffle=self.config.shuffle_train_dataset)
        optim = self.config.optimizer(self.model)
        scheduler, scheduler_type = self.config.scheduler(optim, total_steps) if self.config.scheduler is not None else (None, None)
        model, optim, scheduler = self.engine.wrap_model(model, optim, scheduler, scheduler_type)
        return model, optim, dataloader, scheduler, scheduler_type

    def calculate_metrics(self, label, predictions):
        return {k: v(label, predictions) for k, v in self.config.metrics.items()}

    def _print_metrics(self, prefix, loss, label, predictions):
        results = self.calculate_metrics(label, predictions)
        results['Loss'] = loss
        print(
            f'{prefix}',
            *[f'{k}: {v:10.4f}' for k, v in results.items()]
        )

    def _train_eval(self, do_train, model, optimizer, scheduler, data_loader):
        model.train() if do_train else model.eval()
        losses, labels, preds = np.empty(len(data_loader)), [], []
        prefix_len = len(str(len(data_loader)))
        prev_print_len = 0

        for i, data in enumerate(tqdm(data_loader)):
            # prepare data
            data = {k: v.to(self.device) if isinstance(v, Tensor) else v for k, v in data.items()}

            # do forward pass
            loss, label, pred = self.step(model, data)
            labels.extend(_convert_model_outputs(label))
            preds.extend(_convert_model_outputs(pred))

            # do backward pass
            losses[i] = self.engine.backward_pass(do_train, model, optimizer, scheduler, i, loss)

            # print metrics, checkpoint the model, etc...
            if do_train:
                if self.config.print_steps > 0 and (i + 1) % self.config.print_steps == 0:
                    self._print_metrics(f'[{i + 1:>{prefix_len}}]', np.mean(losses[i - self.config.print_steps + 1:i + 1]), labels[prev_print_len:], preds[prev_print_len:])
                    prev_print_len = len(labels)

        # print metrics, checkpoint the model, etc...
        self._print_metrics(f'[{"=" * prefix_len}]', np.mean(losses), labels, preds)
        return losses, labels, preds

    def _get_experiment_name(self):
        edir = edir_ = f'{self.config.save_dir}/{self.config.experiment_name}'
        i = 1
        while os.path.isdir(edir_):
            edir_ = f'{edir}_{i}'
            i += 1
        return edir_[len(self.config.save_dir) + 1:]

    def _save(self, model, exp_name, subdir):
        # TODO: saving optimizer state for further training
        edir = f'{self.config.save_dir}/{exp_name}/{subdir}'
        os.makedirs(edir, exist_ok=True)
        torch.save(model.state_dict(), f'{edir}/model.pt')

    @abstractmethod
    def step(self, model, data):
        pass

    @abstractmethod
    def step_predict(self, model, data):
        pass

    def train(self, train_data, eval_data):
        exp_name = self._get_experiment_name()
        print(f"Starting training experiment '{exp_name}'...")
        model, optim, dl_train, scheduler, scheduler_type = self._prepare_training(train_data)
        if eval_data is not None:
            dl_val = self._create_dataloader(eval_data, batch_size=self.config.batch_size_eval)
        for epoch in range(self.config.epochs):
            s = f'* Epoch {epoch + 1} / {self.config.epochs}'
            print(s)
            print('=' * len(s))
            # TODO: SAVING THE MODEL
            sched = scheduler if scheduler_type == SchedulerType.STEP else None
            self._train_eval(True, model, optim, sched, dl_train)
            if self.config.save_policy == SavePolicy.EVERY_EPOCH or (epoch + 1 == self.config.epochs and self.config.save_policy == SavePolicy.LAST_EPOCH):
                print('Saving model...')
                self._save(model, exp_name, f'epoch_{epoch + 1}')
            if eval_data is not None:
                with torch.no_grad():
                    self._train_eval(False, model, optim, sched, dl_val)

            if scheduler_type == SchedulerType.EPOCH:
                scheduler.step()
        return exp_name

    def load(self, exp_name=None, subdir=None):
        if exp_name is None:
            exp_name = self.config.experiment_name

        direct = f'{self.config.save_dir}/{exp_name}'
        if subdir is None:
            if not os.path.isdir(direct):
                print(f"'{direct}' directory doesn't exist")
                return
            subdir = -1
            for x in os.listdir(direct):
                x_dir = f'{direct}/{x}'
                if os.path.isdir(x_dir):
                    if x.startswith('epoch_'):
                        try:
                            num = int(x[len('epoch_'):])
                            if num > subdir:
                                subdir = num
                        except ValueError:
                            pass
            if subdir == -1:
                print(f"'{direct}' directory doesn't contain any suitable checkpoints")
                return
            direct = f'{direct}/epoch_{subdir}'

        checkpoint_file = f'{direct}/model.pt'
        if not os.path.isfile(checkpoint_file):
            print(f"'{checkpoint_file}' file doesn't exist")
            return
        print(f"Loading checkpoint '{checkpoint_file}'")
        self.model.load_state_dict(torch.load(checkpoint_file, map_location=self.device))
        print("Loaded checkpoint successfully")

    def predict(self, data, stop_at=-1, print_predictions=False):
        dl = self._create_dataloader(data, batch_size=self.config.batch_size_eval)
        model = self.model.eval()
        preds = []

        for i, d in enumerate(tqdm(dl)):
            d = {k: v.to(self.device) if isinstance(v, Tensor) else v for k, v in d.items()}
            with torch.no_grad():
                pred = _convert_model_outputs(self.step_predict(model, d))
            preds.extend(pred)
            if print_predictions:
                print(i, *pred)
            if i == stop_at:
                break
        return preds

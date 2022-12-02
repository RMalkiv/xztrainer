import math
import os
from abc import abstractmethod, ABC
from collections import defaultdict
from collections.abc import Mapping, Set
from dataclasses import dataclass
from typing import TypeVar, Generic, Optional, Dict, Any, Tuple, List, Union, Iterable

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .model import XZTrainerConfig, SchedulerType, LRSchedulerProtocol, SavePolicy
from .engine import TrainingEngine
from .logger import LoggingEngine, ClassifierType


ModelOutputType = Union[Tensor, List]
ModelOutputsType = Dict[str, ModelOutputType]
DataType = Union[Dict[str, Any], Iterable]


def _convert_model_outputs(out: ModelOutputType) -> List:
    if isinstance(out, Tensor):
        if out.ndim == 0:
            return [out.item()]
        else:
            return [x for x in out.detach().cpu().numpy()]
    elif isinstance(out, List):
        return out
    else:
        raise ValueError(f'Invalid model output type: {type(out)}')


@dataclass
class BaseContext:
    trainer: 'XZTrainer'
    data_loader: DataLoader
    model: Module


@dataclass
class BaseTrainContext(BaseContext):
    engine: TrainingEngine
    logger: LoggingEngine
    optimizer: Optimizer
    scheduler: LRSchedulerProtocol
    model_unwrapped: Module

    epoch: int

    @property
    def total_batches_in_epoch(self) -> int:
        return len(self.data_loader)


@dataclass
class TrainContext(BaseTrainContext):
    total_steps: int
    evaluate_data_loader: Optional[DataLoader]

    @property
    def total_steps_in_epoch(self) -> int:
        return int(math.ceil(len(self.data_loader) / self.trainer.config.accumulation_batches))

    def should_do_update_step(self, batch_i: int) -> bool:
        is_accumulated = (batch_i + 1) % self.trainer.config.accumulation_batches == 0
        is_final = (batch_i + 1) == self.total_batches_in_epoch
        return is_accumulated or is_final

    def should_perform_step_action(self, every_nth_step: int, batch_i: int):
        if every_nth_step < 0:
            return False
        local_step = self.get_local_step_from_batch(batch_i)
        last_step = local_step == self.total_steps_in_epoch
        if every_nth_step == 0:
            return last_step
        else:
            return (local_step % every_nth_step == 0) or last_step

    def get_local_step_from_batch(self, batch_i: int) -> int:
        return int(math.ceil((batch_i + 1) / self.trainer.config.accumulation_batches))

    def get_step_from_batch(self, batch_i: int) -> int:
        steps_in_epoch = int(math.ceil(self.total_batches_in_epoch / self.trainer.config.accumulation_batches))
        return steps_in_epoch * (self.epoch - 1) + self.get_local_step_from_batch(batch_i)

    def get_number_of_accumulations(self, batch_i: int) -> int:
        final_accumulations = self.total_batches_in_epoch % self.trainer.config.accumulation_batches
        if batch_i < self.total_batches_in_epoch - final_accumulations:
            return self.trainer.config.accumulation_batches
        else:
            return final_accumulations


@dataclass
class EvalContext(BaseTrainContext):
    @classmethod
    def from_train_context(cls: 'EvalContext', context: TrainContext):
        return cls(
            trainer=context.trainer,
            engine=context.engine,
            logger=context.logger,
            optimizer=context.optimizer,
            scheduler=context.scheduler,
            data_loader=context.evaluate_data_loader,
            model=context.model,
            model_unwrapped=context.model_unwrapped,
            epoch=context.epoch
        )


class InferContext(BaseContext):
    pass


class XZTrainable(ABC):
    @abstractmethod
    def step(
            self,
            context: BaseContext,
            data: DataType
    ) -> Tuple[Tensor, ModelOutputsType]:
        ...

    def calculate_metrics(
            self,
            context: BaseContext,
            model_outputs: Dict[str, List]
    ) -> Dict[ClassifierType, float]:
        return {}

    def log(self, context: BaseTrainContext):
        pass

    def on_update(self, context: TrainContext, step: int):
        pass


class XZTrainer:
    config: XZTrainerConfig

    def __init__(self, config: XZTrainerConfig, model: Module, trainable: XZTrainable,
                 device: Optional[torch.device] = None):
        self.config = config

        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device

        self.model = model.to(device)
        self.trainable = trainable

    def _create_dataloader(self, data: Dataset, **kwargs) -> DataLoader:
        return DataLoader(
            data,
            collate_fn=self.config.collate_fn,
            num_workers=self.config.dataloader_num_workers,
            persistent_workers=self.config.dataloader_persistent_workers,
            pin_memory=self.config.dataloader_pin_memory,
            **kwargs
        )

    def _log_trainable(self, context: BaseTrainContext, model_outputs: ModelOutputsType,
                       prev_output_lens: Optional[Dict[str, int]] = None) -> Dict[str, int]:
        new_output_lens = {k: len(v) for k, v in model_outputs.items()}
        if prev_output_lens is not None:
            model_outputs = {k: v[prev_output_lens[k]:] for k, v in model_outputs.items()}
        scalars = self.trainable.calculate_metrics(context, model_outputs)
        scalars['loss'] = float(np.mean(model_outputs['loss']))
        for k, v in scalars.items():
            context.logger.log_scalar(k, v)
        self.trainable.log(context)
        context.logger.flush()
        return new_output_lens

    def _move_data_to_device(self, data: Any) -> DataType:
        if isinstance(data, Tensor):
            return data.to(self.device)
        elif isinstance(data, Mapping):
            return {k: self._move_data_to_device(v) for k, v in data.items()}
        elif isinstance(data, Tuple):
            return tuple(self._move_data_to_device(v) for v in data)
        elif isinstance(data, List):
            return [self._move_data_to_device(v) for v in data]
        elif isinstance(data, Set):
            return set(self._move_data_to_device(v) for v in data)
        else:
            return data

    def _forward_pass(self, context: BaseContext, model_outputs: Dict[str, ModelOutputType], data: DataType) -> Tuple[Tensor, ModelOutputsType]:
        data = self._move_data_to_device(data)

        loss, model_output = self.trainable.step(context, data)
        if loss is not None:
            model_outputs['loss'].append(loss.item())
        for k, v in model_output.items():
            model_outputs[k].extend(_convert_model_outputs(v))
        return loss, model_output

    def _set_training_state(self, context: BaseContext):
        context.model.train()
        if isinstance(context, BaseTrainContext):
            context.logger.update_top_classifier(('step', 'train'))

    def _set_evaluating_state(self, context: BaseContext):
        context.model.eval()
        if isinstance(context, BaseTrainContext):
            context.logger.update_top_classifier(('step', 'eval'))

    def _train_epoch(self, context: TrainContext):
        self._set_training_state(context)

        model_outputs = defaultdict(lambda: list())
        prev_output_lens = defaultdict(lambda: 0)

        with tqdm(total=context.total_steps_in_epoch, desc=f'Train > Epoch {context.epoch}') as progress_bar:
            for batch_i, data in enumerate(context.data_loader):
                step = context.get_step_from_batch(batch_i)
                do_update = context.should_do_update_step(batch_i)

                if do_update:
                    context.logger.update_time_step(step)

                loss, _ = self._forward_pass(context, model_outputs, data)

                if do_update:
                    for group_i, group in enumerate(context.optimizer.param_groups):
                        context.logger.log_scalar(['lr', str(group_i)], group['lr'])

                context.engine.backward_pass(context, batch_i, loss)

                if do_update:
                    self.trainable.on_update(context, step)

                    if context.should_perform_step_action(self.config.print_steps, batch_i):
                        prev_output_lens = self._log_trainable(context, model_outputs, prev_output_lens)

                    progress_bar.update()

                    if context.evaluate_data_loader and context.should_perform_step_action(self.config.eval_steps,
                                                                                           batch_i):
                        self._set_evaluating_state(context)
                        context_eval = EvalContext.from_train_context(context)
                        with torch.no_grad():
                            eval_model_outputs = defaultdict(lambda: list())
                            for eval_data in context_eval.data_loader:
                                self._forward_pass(context_eval, eval_model_outputs, eval_data)
                        self._log_trainable(context, eval_model_outputs)
                        self._set_training_state(context)

        context.logger.update_top_classifier(('epoch', 'train'))
        context.logger.update_time_step(context.epoch)
        self._log_trainable(context, model_outputs)

    def _get_experiment_name(self):
        edir = edir_ = f'{self.config.save_dir}/{self.config.experiment_name}'
        i = 1
        while os.path.isdir(edir_):
            edir_ = f'{edir}_{i}'
            i += 1
        return edir_[len(self.config.save_dir) + 1:]

    def _save(self, model: Module, exp_name: str, subdir: str):
        # TODO: saving optimizer state for further training
        edir = f'{self.config.save_dir}/{exp_name}/{subdir}'
        os.makedirs(edir, exist_ok=True)
        torch.save(model.state_dict(), f'{edir}/model.pt')

    def _calculate_total_steps(self, dataset: Dataset):
        epoch_batches = int(math.ceil(len(dataset) / self.config.batch_size))
        epoch_steps = int(math.ceil(epoch_batches / self.config.accumulation_batches))
        return epoch_steps * self.config.epochs

    def train(self, train_data: Dataset, eval_data: Dataset):
        engine = self.config.engine.create_engine(self)
        exp_name = self._get_experiment_name()
        total_train_steps = self._calculate_total_steps(train_data)

        print(f"Starting training experiment '{exp_name}' with total {total_train_steps} steps...")

        # Initialize and wrap model, optimizer and scheduler
        optim = self.config.optimizer(self.model)
        if self.config.scheduler and self.config.scheduler_type:
            scheduler = self.config.scheduler(optim, total_train_steps)
            scheduler_type = self.config.scheduler_type
        else:
            scheduler = None
            scheduler_type = None
        model, optim, scheduler = engine.wrap_model(self.model, optim, scheduler, scheduler_type)

        # Create DataLoaders
        train_dl = self._create_dataloader(
            train_data,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle_train_dataset
        )
        if eval_data:
            eval_dl = self._create_dataloader(eval_data, batch_size=self.config.batch_size_eval)
        else:
            eval_dl = None

        # Run epoch loop
        with self.config.logger.create_engine(exp_name) as logger:
            for epoch_i in range(self.config.epochs):
                epoch = epoch_i + 1
                s = f'* Epoch {epoch} / {self.config.epochs}'
                print(s)
                print('=' * len(s))

                if scheduler_type == SchedulerType.STEP:
                    _scheduler = scheduler
                else:
                    _scheduler = None

                self._train_epoch(
                    TrainContext(
                        trainer=self,
                        engine=engine,
                        logger=logger,
                        optimizer=optim,
                        scheduler=_scheduler,
                        data_loader=train_dl,
                        model=model,
                        model_unwrapped=self.model,
                        epoch=epoch,
                        total_steps=total_train_steps,
                        evaluate_data_loader=eval_dl
                    )
                )

                if self._should_save(epoch):
                    print('Saving model...')
                    self._save(model, exp_name, f'epoch_{epoch}')

                if scheduler_type == SchedulerType.EPOCH:
                    scheduler.step()
        return exp_name

    def load(self, exp_name=None, epoch=None):
        if exp_name is None:
            exp_name = self.config.experiment_name

        direct = f'{self.config.save_dir}/{exp_name}'
        if epoch is None:
            if not os.path.isdir(direct):
                print(f"'{direct}' directory doesn't exist")
                return
            epoch = -1
            for x in os.listdir(direct):
                x_dir = f'{direct}/{x}'
                if os.path.isdir(x_dir):
                    if x.startswith('epoch_'):
                        try:
                            num = int(x[len('epoch_'):])
                            if num > epoch:
                                epoch = num
                        except ValueError:
                            pass
            if epoch == -1:
                print(f"'{direct}' directory doesn't contain any suitable checkpoints")
                return
        direct = f'{direct}/epoch_{epoch}'

        checkpoint_file = f'{direct}/model.pt'
        self.load_checkpoint_file(checkpoint_file)

    def load_checkpoint_file(self, checkpoint_file: str):
        if not os.path.isfile(checkpoint_file):
            print(f"'{checkpoint_file}' file doesn't exist")
            return
        print(f"Loading checkpoint '{checkpoint_file}'")
        result = self.model.load_state_dict(torch.load(checkpoint_file, map_location=self.device), strict=False)
        print(f'Result of loading a checkpoint: {result}')
        print("Loaded checkpoint successfully")

    def _should_save(self, epoch: int) -> bool:
        policy = self.config.save_policy
        if policy == SavePolicy.EVERY_EPOCH:
            return True
        elif policy == SavePolicy.LAST_EPOCH:
            return epoch == self.config.epochs
        elif policy == SavePolicy.NEVER:
            return False
        else:
            raise ValueError(f'Illegal SavePolicy: {policy}')

    def infer(
            self, dataset: Dataset, calculate_metrics: bool = False
    ) -> Tuple[ModelOutputsType, Dict[ClassifierType, float]]:
        dataloader = self._create_dataloader(dataset, batch_size=self.config.batch_size_eval)
        context = InferContext(
            trainer=self,
            data_loader=dataloader,
            model=self.model
        )
        self._set_evaluating_state(context)
        with torch.no_grad():
            model_outputs = defaultdict(lambda: list())
            with tqdm(total=len(dataloader), desc=f'Inference') as progress_bar:
                for data in dataloader:
                    self._forward_pass(context, model_outputs, data)
                    progress_bar.update()
        self._set_training_state(context)
        if calculate_metrics:
            metrics = self.trainable.calculate_metrics(context, model_outputs)
            return model_outputs, metrics
        else:
            return model_outputs, {}


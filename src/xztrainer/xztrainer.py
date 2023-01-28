import abc
import math
import random
import re
from abc import abstractmethod, ABC
from collections import defaultdict
from collections.abc import Mapping, Set
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union, Iterable

import numpy as np
import torch
from torch import Tensor, autocast
from torch.cuda.amp import GradScaler
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .logger import LoggingEngine, ClassifierType
from .model import XZTrainerConfig, SchedulerType, LRSchedulerProtocol, CheckpointType
from .sampler import ReusableSequentialSampler

ModelOutputType = Union[Tensor, List]
ModelOutputsType = Dict[str, ModelOutputType]
DataType = Union[Dict[str, Any], Iterable]

_RE_SAVE_NAME = re.compile('save-\d+\.pt')


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
    dataset_batches: int
    data_loader: DataLoader
    model: Module


@dataclass
class BaseTrainContext(BaseContext):
    logger: LoggingEngine
    scaler: Optional[GradScaler]
    optimizer: Optimizer
    scheduler: LRSchedulerProtocol
    model_unwrapped: Module

    epoch: int

    @property
    def total_batches_in_epoch(self) -> int:
        return self.dataset_batches


@dataclass
class TrainContext(BaseTrainContext):
    total_steps: int
    shift_batch_i: int
    sampler: ReusableSequentialSampler
    progress_bar: tqdm
    evaluate_data_loader: Optional[DataLoader]

    @property
    def total_steps_in_epoch(self) -> int:
        return int(math.ceil(self.dataset_batches / self.trainer.config.accumulation_batches))

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

    def get_actual_batch_i(self, batch_i: int) -> int:
        return self.shift_batch_i + batch_i

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
            logger=context.logger,
            optimizer=context.optimizer,
            scaler=context.scaler,
            scheduler=context.scheduler,
            data_loader=context.evaluate_data_loader,
            model=context.model,
            model_unwrapped=context.model_unwrapped,
            epoch=context.epoch,
            dataset_batches=context.dataset_batches
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

    @abc.abstractmethod
    def calculate_metrics(
            self,
            context: BaseContext,
            model_outputs: Dict[str, List]
    ) -> Dict[ClassifierType, float]:
        ...

    def on_load(self, context: TrainContext, step: int):
        pass

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

    def _forward_pass(self, context: BaseContext, model_outputs: Dict[str, ModelOutputType], data: DataType) -> Optional[Tuple[Tensor, ModelOutputsType]]:
        data = self._move_data_to_device(data)
        loss, model_output = self.trainable.step(context, data)
        if loss is not None:
            if torch.isnan(loss):
                print('NAN loss found!')
                if self.config.skip_nan_loss:
                    return None
            model_outputs['loss'].append(loss.item())
        for k, v in model_output.items():
            model_outputs[k].extend(_convert_model_outputs(v))
        return loss, model_output

    @staticmethod
    def _set_training_state(context: BaseContext):
        context.model.train()
        if isinstance(context, BaseTrainContext):
            context.logger.update_top_classifier(('step', 'train'))

    @staticmethod
    def _set_evaluating_state(context: BaseContext):
        context.model.eval()
        if isinstance(context, BaseTrainContext):
            context.logger.update_top_classifier(('step', 'eval'))

    def _train_epoch(self, context: TrainContext):
        self._set_training_state(context)
        context.progress_bar.update(
            context.get_step_from_batch(context.get_actual_batch_i(0)) - 1 - context.progress_bar.n)

        model_outputs = defaultdict(lambda: list())
        prev_output_lens = defaultdict(lambda: 0)

        for batch_i, data in enumerate(context.data_loader):
            batch_i = context.get_actual_batch_i(batch_i)
            step = context.get_step_from_batch(batch_i)
            do_update = context.should_do_update_step(batch_i)

            if do_update:
                context.logger.update_time_step(step)

            model_op_ctx = nullcontext() if self.config.amp_dtype is None else autocast(device_type='cuda',
                                                                                        dtype=self.config.amp_dtype)
            with model_op_ctx:
                loss, _ = self._forward_pass(context, model_outputs, data)

            if do_update:
                for group_i, group in enumerate(context.optimizer.param_groups):
                    context.logger.log_scalar(['lr', str(group_i)], group['lr'])

            # engine start
            with model_op_ctx:
                loss = loss / context.get_number_of_accumulations(batch_i)
            if context.scaler is not None:
                loss = context.scaler.scale(loss)
            # multiple consecutive loss.backward() sum up the gradients, so we need to divide loss by num of accumulations
            loss.backward()
            if do_update:
                if context.scaler is not None:
                    context.scaler.unscale_(context.optimizer)
                l2_grad_norm = torch.norm(
                    torch.stack(
                        [torch.norm(p.grad.detach(), 2.0)
                         for p in context.model.parameters()
                         if p.grad is not None]
                    ),
                    2
                ).item()
                context.logger.log_scalar('l2 grad norm before clip', l2_grad_norm)
                max_norm = context.trainer.config.gradient_clipping
                if max_norm > 0:
                    clip_grad_norm_(context.model.parameters(), max_norm=max_norm)
                if context.scaler is not None:
                    context.scaler.step(context.optimizer)
                    context.scaler.update()
                else:
                    context.optimizer.step()
                if context.scheduler is not None:
                    context.scheduler.step()
                context.optimizer.zero_grad()
            # engine end

            if do_update:
                self.trainable.on_update(context, step)

                if context.should_perform_step_action(self.config.print_steps, batch_i):
                    prev_output_lens = self._log_trainable(context, model_outputs, prev_output_lens)

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
                if context.should_perform_step_action(self.config.save_steps, batch_i):
                    self._save(context, step, batch_i)
                context.progress_bar.update()

        if len(context.data_loader) > 0:
            context.logger.update_top_classifier(('epoch', 'train'))
            context.logger.update_time_step(context.epoch)
            self._log_trainable(context, model_outputs)

    def _cleanup_saves(self):
        if self.config.save_keep_n >= 0:
            save_dir = Path(self.config.save_dir) / self.config.experiment_name
            save_files = sorted(self._get_save_files(save_dir), reverse=True)
            save_files_to_delete = save_files[self.config.save_keep_n:]
            for step, file in save_files_to_delete:
                file.unlink()

    def _save(self, context: TrainContext, step: int, batch_i: int):
        save_dir = Path(self.config.save_dir) / self.config.experiment_name
        save_dir.mkdir(exist_ok=True, parents=True)
        save_path = save_dir / f'save-{step}.pt'
        save_obj = {
            'model': context.model.state_dict(),
            'optimizer': context.optimizer.state_dict(),
            'scaler': context.scaler.state_dict() if context.scaler is not None else None,
            'scheduler': context.scheduler.state_dict(),
            'epoch': context.epoch,
            'batch_i_saved_at': batch_i,
            'sampler': context.sampler.save_state(batch_i, self.config.batch_size),
            'rng_state_torch': torch.get_rng_state(),
            'rng_state_numpy': np.random.get_state(),
            'rng_state_python': random.getstate()
        }
        torch.save(save_obj, str(save_path))
        self._cleanup_saves()

    @staticmethod
    def _get_save_files(save_dir: Path) -> List[Tuple[int, Path]]:
        save_files = [x for x in save_dir.iterdir() if _RE_SAVE_NAME.fullmatch(x.name)]
        save_files_with_step = [(int(x.stem.split('-')[1]), x) for x in save_files]
        return save_files_with_step

    def _load(self, step: int) -> Optional[Dict[str, Any]]:
        save_dir = Path(self.config.save_dir) / self.config.experiment_name
        if step == -1:
            if save_dir.is_dir():
                save_files = self._get_save_files(save_dir)
                if len(save_files) == 0:
                    return None
                else:
                    save_file = max(save_files)[1]
            else:
                return None
        else:
            save_file = save_dir / f'save-{step}.pt'
        print(f'Loading state from {save_file}')
        return torch.load(save_file, map_location=self.device)

    def _calculate_batches_in_epoch(self, dataset: Dataset):
        return int(math.ceil(len(dataset) / self.config.batch_size))

    def _calculate_steps_in_epoch(self, dataset: Dataset):
        return int(math.ceil(self._calculate_batches_in_epoch(dataset) / self.config.accumulation_batches))

    def _calculate_total_steps(self, dataset: Dataset):
        return self._calculate_steps_in_epoch(dataset) * self.config.epochs

    def train(self, train_data: Dataset, eval_data: Dataset, resume_from: int = -1):
        exp_name = self.config.experiment_name
        batches_in_epoch = self._calculate_batches_in_epoch(train_data)
        total_train_steps = self._calculate_total_steps(train_data)

        print(f"Starting training experiment '{exp_name}' with total {total_train_steps} steps...")

        # Initialize and wrap model, optimizer and scheduler
        optim = self.config.optimizer(self.model)
        if self.config.amp_dtype is not None:
            scaler = GradScaler()
        else:
            scaler = None
        if self.config.scheduler and self.config.scheduler_type:
            scheduler = self.config.scheduler(optim, total_train_steps)
            scheduler_type = self.config.scheduler_type
        else:
            scheduler = None
            scheduler_type = None
        model = self.model

        # Load the state
        state = self._load(resume_from)
        if state is not None:
            model.load_state_dict(state['model'])
            optim.load_state_dict(state['optimizer'])
            if scaler is not None:
                scaler.load_state_dict(state['scaler'])
            scheduler.load_state_dict(state['scheduler'])
            torch.random.set_rng_state(state['rng_state_torch'].cpu())
            np.random.set_state(state['rng_state_numpy'])
            random.setstate(state['rng_state_python'])
            start_from_epoch = state['epoch']
            batch_i_saved_at = state['batch_i_saved_at']
            sampler_state = state['sampler']
        else:
            start_from_epoch = 1
            batch_i_saved_at = -1
            sampler_state = None

        del state

        if eval_data:
            eval_dl = self._create_dataloader(eval_data, batch_size=self.config.batch_size_eval)
        else:
            eval_dl = None

        # Run epoch loop
        with self.config.logger.create_engine(exp_name) as logger:
            with tqdm(total=total_train_steps, desc=f'Train') as progress_bar:
                for epoch in range(start_from_epoch, self.config.epochs + 1):
                    if scheduler_type == SchedulerType.STEP:
                        _scheduler = scheduler
                    else:
                        _scheduler = None

                    if epoch == start_from_epoch and sampler_state is not None:
                        train_sampler = ReusableSequentialSampler.from_state(sampler_state)
                        shift_batch_i = batch_i_saved_at + 1
                    else:
                        train_sampler = ReusableSequentialSampler.new(train_data,
                                                                      self.config.dataloader_shuffle_train_dataset)
                        shift_batch_i = 0

                    train_dl = self._create_dataloader(
                        train_data,
                        batch_size=self.config.batch_size,
                        sampler=train_sampler
                    )
                    context = TrainContext(
                            trainer=self,
                            logger=logger,
                            optimizer=optim,
                            scaler=scaler,
                            scheduler=_scheduler,
                            data_loader=train_dl,
                            sampler=train_sampler,
                            model=model,
                            model_unwrapped=self.model,
                            epoch=epoch,
                            total_steps=total_train_steps,
                            evaluate_data_loader=eval_dl,
                            dataset_batches=batches_in_epoch,
                            shift_batch_i=shift_batch_i,
                            progress_bar=progress_bar
                        )
                    if epoch == start_from_epoch:
                        self.trainable.on_load(context, context.get_step_from_batch(context.get_actual_batch_i(0)))
                    self._train_epoch(context)

                    if scheduler_type == SchedulerType.EPOCH:
                        scheduler.step()
        return exp_name

    def load_model_checkpoint(self, checkpoint_file: str, checkpoint_type: CheckpointType):
        if not Path(checkpoint_file).is_file():
            print(f"'{checkpoint_file}' file doesn't exist")
            return
        print(f"Loading checkpoint '{checkpoint_file}'")
        checkpoint_obj = torch.load(checkpoint_file, map_location=self.device)
        if checkpoint_type == CheckpointType.MODEL_ONLY:
            checkpoint_obj = checkpoint_obj
        elif checkpoint_type == CheckpointType.XZTRAINER:
            checkpoint_obj = checkpoint_obj['model']
        else:
            raise ValueError(f'invalid checkpoint type: {checkpoint_type}')
        result = self.model.load_state_dict(checkpoint_obj, strict=False)
        print(f'Result of loading a checkpoint: {result}')
        print("Loaded checkpoint successfully")

    def infer(
            self, dataset: Dataset, calculate_metrics: bool = False
    ) -> Tuple[ModelOutputsType, Dict[ClassifierType, float]]:
        dataloader = self._create_dataloader(dataset, batch_size=self.config.batch_size_eval)
        context = InferContext(
            trainer=self,
            data_loader=dataloader,
            model=self.model,
            dataset_batches=len(dataloader)
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

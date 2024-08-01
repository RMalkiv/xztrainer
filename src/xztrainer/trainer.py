import math
import re
import shutil
import typing as t
from logging import basicConfig
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils.tqdm import tqdm
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Metric

from xztrainer.auto_tracker_config import create_tracker_config
from xztrainer.context import ContextType, TrainContext, BaseContext, BaseTrainContext, EvalContext
from xztrainer.functional import count_parameters
from xztrainer.logger import AccelerateLogger
from xztrainer.model import XZTrainerConfig, DataType, ModelOutputsType, MetricMultiOutputNamedProtocol
from xztrainer.tensor_helper import detach_tensor, move_data_to_device
from xztrainer.trainable import XZTrainable

_RE_SAVE_NAME = re.compile(r'save-\d+')


class XZTrainState:
    def __init__(self, accelearator: Accelerator, print_metrics: dict[str, Metric], train_metrics: dict[str, Metric],
                 eval_metrics: dict[str, Metric]):
        self.accelerator = accelearator
        self.print_metrics = print_metrics
        self.train_metrics = train_metrics
        self.eval_metrics = eval_metrics
        self.current_epoch = 1
        self.passed_steps = 0

    @staticmethod
    def _synced_state_dict(metric: Metric):
        with metric.sync_context():
            return metric.state_dict()

    @staticmethod
    def _metrics_to_state_dict(metrics: dict[str, Metric]) -> dict[str, dict[str, t.Any]]:
        return {k: XZTrainState._synced_state_dict(v) for k, v in metrics.items()}

    def _load_metrics_from_state_dict(self, metrics: dict[str, Metric], state_dict: dict[str, dict[str, t.Any]]):
        for k, v in metrics.items():
            v.load_state_dict(move_data_to_device(state_dict[k], self.accelerator))

    def update_epoch(self):
        self.current_epoch += 1

    def update_passed_steps(self):
        self.passed_steps += 1

    def load_state_dict(self, dct: dict[str, t.Any]):
        self._load_metrics_from_state_dict(self.print_metrics, dct['print_metrics'])
        self._load_metrics_from_state_dict(self.train_metrics, dct['train_metrics'])
        self._load_metrics_from_state_dict(self.eval_metrics, dct['eval_metrics'])
        self.current_epoch = dct['current_epoch']
        self.passed_steps = dct['passed_steps']

    def state_dict(self) -> dict[str, t.Any]:
        return {
            'print_metrics': self._metrics_to_state_dict(self.print_metrics),
            'train_metrics': self._metrics_to_state_dict(self.train_metrics),
            'eval_metrics': self._metrics_to_state_dict(self.eval_metrics),
            'current_epoch': self.current_epoch,
            'passed_steps': self.passed_steps
        }


class XZTrainer:
    config: XZTrainerConfig
    """Configuration used by this trainer"""

    def __init__(
            self,
            config: XZTrainerConfig,
            model: nn.Module,
            trainable: XZTrainable,
            accelerator: Accelerator
    ):
        """
        Creates a trainer instance

        Args:
            config: Trainer config to use
            model: Model instance to train
            trainable: Trainable instance to use
            accelerator: Accelerator object
        """
        self.config = config

        self.accelerator = accelerator
        self.model = model
        self.trainable = trainable
        self.logger = get_logger(name='xztrainer')

    def _create_dataloader(self, data: Dataset, **kwargs) -> DataLoader:
        return DataLoader(
            data,
            collate_fn=self.config.collate_fn,
            num_workers=self.config.dataloader_num_workers,
            persistent_workers=self.config.dataloader_persistent_workers,
            pin_memory=self.config.dataloader_pin_memory,
            **kwargs
        )

    def _create_metrics(self, context_type: ContextType) -> dict[str, Metric]:
        metrics = {}
        for k, metric in self.trainable.create_metrics(context_type).items():
            metric.persistent(True)
            metrics[k] = metric.to(self.accelerator.device)
        return metrics

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

    def _forward_pass(self, context: BaseContext, data: DataType) -> tuple[Tensor, ModelOutputsType]:
        loss, model_output = self.trainable.step(context, data)
        return loss, model_output

    def _update_metrics(self, context_type: ContextType, model_outputs: ModelOutputsType, metrics: dict[str, Metric]):
        model_outputs = {k: detach_tensor(v, move_to_cpu=False) for k, v in model_outputs.items()}
        self.trainable.update_metrics(context_type, model_outputs, metrics)

    def _calculate_reset_metrics(
            self, trainable: XZTrainable, context_type: ContextType, metrics: dict[str, Metric]
    ) -> dict[str, float]:
        metric_values = {}
        for name, metric in metrics.items():
            metric_val = metric.compute()
            metric_val_els = metric_val.numel()
            if metric_val_els == 0:
                raise ValueError(f'empty metric {name}')
            elif metric_val_els == 1:
                metric_values[name] = metric_val.item()
            else:
                if isinstance(metric, MetricMultiOutputNamedProtocol):
                    metric_names = metric.multi_output_names
                else:
                    metric_names = [str(i) for i in range(metric_val_els)]
                for itm_name, itm in zip(metric_names, metric_val.flatten()):
                    metric_values[f'{name}_{itm_name}'] = itm.item()
            metric.reset()
        metric_values.update(trainable.calculate_composition_metrics(context_type, metric_values))
        return metric_values

    def _log_trainable(self, context: BaseTrainContext, metrics: dict[str, Metric]):
        self.accelerator.wait_for_everyone()  # wait for everyone to contribute to metrics
        for k, v in self._calculate_reset_metrics(self.trainable, context.context_type, metrics).items():
            context.logger.log_scalar(k, v)
        self.trainable.log(context)

    def _cleanup_saves(self):
        if self.config.save_keep_n >= 0:
            save_dir = Path(self.accelerator.project_dir) / 'checkpoint' / self.config.experiment_name
            save_files = sorted(self._get_save_files(save_dir), reverse=True)
            save_files_to_delete = save_files[self.config.save_keep_n:]
            for step, file in save_files_to_delete:
                shutil.rmtree(file)

    def _save(self, step: int):
        save_dir = Path(self.accelerator.project_dir) / 'checkpoint' / self.config.experiment_name
        save_dir.mkdir(exist_ok=True, parents=True)
        save_path = save_dir / f'save-{step}'
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(str(save_path))
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self._cleanup_saves()

    def _load(self):
        save_dir = Path(self.accelerator.project_dir) / 'checkpoint' / self.config.experiment_name
        if save_dir.is_dir():
            save_files = self._get_save_files(save_dir)
            if len(save_files) > 0:
                save_file = max(save_files)[1]
                self.logger.info(f'Loading state from {save_file}')
                return self.accelerator.load_state(str(save_file))

    @staticmethod
    def _get_save_files(save_dir: Path) -> list[tuple[int, Path]]:
        save_files = [x for x in save_dir.iterdir() if _RE_SAVE_NAME.fullmatch(x.name) and x.is_dir()]
        save_files_with_step = [(int(x.stem.split('-')[1]), x) for x in save_files]
        return save_files_with_step

    def _train_epoch(self, context: TrainContext):
        self._set_training_state(context)
        context.progress_bar.update(
            context.train_state.passed_steps - context.progress_bar.n
        )

        for data in context.data_loader:
            current_step = context.train_state.passed_steps + 1

            if self.accelerator.sync_gradients:
                context.logger.update_time_step(current_step)

            with self.accelerator.accumulate(context.model):
                loss, model_out = self._forward_pass(context, data)
                model_out = self.trainable.cut_model_outputs(
                    context, model_out,
                    self.accelerator.gradient_state.remainder
                )

                self._update_metrics(context.context_type, model_out, context.train_state.print_metrics)
                self._update_metrics(context.context_type, model_out, context.train_state.train_metrics)

                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients and self.accelerator.is_main_process:
                    for group_i, group in enumerate(context.optimizer.param_groups):
                        context.logger.log_scalar(['lr', str(group_i)], group['lr'])

                if self.accelerator.sync_gradients:
                    self.trainable.on_pre_update(context, current_step)

                if self.accelerator.sync_gradients:
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
                        self.accelerator.clip_grad_norm_(context.model.parameters(), max_norm=max_norm)

                context.optimizer.step()
                if self.accelerator.sync_gradients:
                    context.scheduler.step()
                context.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    self.trainable.on_update(context, current_step)

            if self.accelerator.sync_gradients:
                if context.should_perform_step_action(self.config.log_steps, current_step):
                    self._log_trainable(context, context.train_state.print_metrics)

                if context.evaluate_data_loader and context.should_perform_step_action(self.config.eval_steps,
                                                                                       current_step):
                    self._set_evaluating_state(context)
                    context_eval = EvalContext.from_train_context(context)
                    with torch.no_grad():
                        for eval_data in context_eval.data_loader:
                            _, model_out_eval = self._forward_pass(context_eval, eval_data)
                            model_out_eval = self.trainable.cut_model_outputs(
                                context, model_out_eval,
                                self.accelerator.gradient_state.remainder
                            )
                            self._update_metrics(context_eval.context_type, model_out_eval,
                                                 context.train_state.eval_metrics)
                    self._log_trainable(context_eval, context.train_state.eval_metrics)
                    self._set_training_state(context)
                context.train_state.update_passed_steps()
                if self.accelerator.gradient_state.end_of_dataloader:
                    context.train_state.update_epoch()
                if context.should_perform_step_action(self.config.save_steps, current_step):
                    self._save(current_step)
                context.progress_bar.update()

        if len(context.data_loader) > 0:
            context.logger.update_top_classifier(('epoch', 'train'))
            context.logger.update_time_step(context.train_state.current_epoch)
            self._log_trainable(context, context.train_state.train_metrics)

    def train(self, train_data: Dataset, eval_data: t.Optional[Dataset]):
        """
        This function:

        1. Prepares all the objects used for further training, such as optimizer, dataloaders, metrics, learning rate scheduler, logger.
        2. Wraps these objects using `accelerator.prepare(...)`
        3. If a training checkpoint exists - loads it and initializes all the previously prepared objects from it.
        4. Prints out total parameter count used for training
        5. Trains for N epochs, where N specified in trainer config
        6. While training, the trainer can save its state, run evaluation and log metrics, - it is specified in trainer config.

        Args:
            train_data: Dataset used for training
            eval_data: Dataset used for evaluation. Can be `None` to disable evaluation.
        """
        if self.accelerator.project_configuration.project_dir is None:
            self.accelerator.project_configuration.project_dir = '.'
        if self.config.logging_level is not None:
            basicConfig(format='%(asctime)s [%(name)s] [%(levelname)s] %(message)s', level=self.config.logging_level)

        if self.config.tracker_logging_dir is None:
            log_dir = Path(self.accelerator.project_configuration.project_dir) / 'runs'
            log_dir.mkdir(parents=True, exist_ok=True)
            self.accelerator.project_configuration.logging_dir = str(log_dir)
        else:
            self.accelerator.project_configuration.logging_dir = self.config.tracker_logging_dir

        # Initialize and wrap model, optimizer and scheduler
        optim = self.config.optimizer(self.model)
        train_dl = self._create_dataloader(train_data, batch_size=self.config.minibatch_size)
        sync_steps_per_epoch_unscaled = int(math.ceil(len(train_dl) / self.accelerator.gradient_accumulation_steps))
        sync_steps_unscaled = sync_steps_per_epoch_unscaled * self.config.epochs
        scheduler = self.config.scheduler(optim, sync_steps_unscaled)
        model_unwrapped = self.model
        metrics_print = self._create_metrics(ContextType.TRAIN)
        metrics_train = self._create_metrics(ContextType.TRAIN)
        metrics_eval = self._create_metrics(ContextType.EVAL)
        if eval_data:
            eval_dl = self._create_dataloader(eval_data, batch_size=self.config.minibatch_size_eval)
        else:
            eval_dl = None

        model, optim, scheduler, train_dl, eval_dl = self.accelerator.prepare(
            model_unwrapped, optim, scheduler, train_dl, eval_dl
        )
        sync_steps_per_epoch = int(math.ceil(len(train_dl) / self.accelerator.gradient_accumulation_steps))
        sync_steps = sync_steps_per_epoch * self.config.epochs

        additional_state = XZTrainState(self.accelerator, metrics_print, metrics_train, metrics_eval)
        self.accelerator.register_for_checkpointing(additional_state)

        self.trainable.on_register_objects(self)

        self._load()
        start_epoch = additional_state.current_epoch

        self.logger.info(f"Starting training experiment...", main_process_only=True)
        self.logger.info(f'Total synchronized steps: {sync_steps:,}', main_process_only=True)
        self.logger.info(f'Parameters [total]: {count_parameters(model_unwrapped):,}', main_process_only=True)
        self.logger.info(f'Parameters [train]: {count_parameters(model_unwrapped, lambda p: p.requires_grad):,}',
                         main_process_only=True)
        self.logger.info(f'Parameters [fixed]: {count_parameters(model_unwrapped, lambda p: not p.requires_grad):,}',
                         main_process_only=True)

        # Run epoch loop
        logger = AccelerateLogger(self.accelerator)
        with tqdm(total=sync_steps, desc=f'Train') as progress_bar:
            for epoch in range(start_epoch, self.config.epochs + 1):
                if epoch == start_epoch and additional_state.passed_steps != 0:  # resumed
                    steps_remainder = additional_state.passed_steps % sync_steps_per_epoch
                    passed_minibatches_last_epoch = steps_remainder * self.accelerator.gradient_accumulation_steps
                    train_dl_current = self.accelerator.skip_first_batches(
                        train_dl,
                        num_batches=passed_minibatches_last_epoch
                    )
                else:
                    train_dl_current = train_dl
                context = TrainContext(
                    trainer=self,
                    logger=logger,
                    optimizer=optim,
                    scheduler=scheduler,
                    data_loader=train_dl_current,
                    model=model,
                    model_unwrapped=model_unwrapped,
                    sync_steps=sync_steps,
                    evaluate_data_loader=eval_dl,
                    progress_bar=progress_bar,
                    train_state=additional_state
                )
                if epoch == start_epoch:
                    self.accelerator.init_trackers(
                        self.config.experiment_name,
                        config=create_tracker_config(context),
                        init_kwargs=self.config.tracker_init_kwargs
                    )
                    self.trainable.on_load(context, additional_state.passed_steps + 1)
                self._train_epoch(context)
        self.accelerator.end_training()

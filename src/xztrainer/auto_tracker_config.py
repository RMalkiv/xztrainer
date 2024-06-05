import typing as t

from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.scheduler import AcceleratedScheduler
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from xztrainer.context import TrainContext
from xztrainer.model import XZTrainerConfig, TrackerConfigType, LRSchedulerProtocol


def _prepend_tracker_config(config: TrackerConfigType, prefix: str) -> TrackerConfigType:
    return {f'{prefix}_{k}': v for k, v in config.items()}


def _filter_dict_for_tracker_confix(values: dict[str, t.Any]) -> TrackerConfigType:
    out_dict = {}

    for k, v in values.items():
        if isinstance(v, (float, int, bool, str)) or v is None:
            out_dict[k] = v
        elif isinstance(v, (tuple, str)):
            out_dict.update(_filter_dict_for_tracker_confix({f'{k}_{i}': itm for i, itm in enumerate(v)}))

    return out_dict


def _extract_optimizer_config(optimizer: Optimizer) -> TrackerConfigType:
    if isinstance(optimizer, AcceleratedOptimizer):
        name = optimizer.optimizer.__class__.__name__
    else:
        name = optimizer.__class__.__name__
    dct = {
        'name': name,
        **_filter_dict_for_tracker_confix(optimizer.defaults)
    }
    dct.pop('foreach', None)
    dct.pop('fused', None)
    dct.pop('differentiable', None)
    dct.pop('capturable', None)
    dct.pop('maximize', None)
    return dct


def _extract_accelerator_config(accelerator: Accelerator) -> TrackerConfigType:
    return {
        'gradient_accumulation_steps': accelerator.gradient_accumulation_steps,
        'num_processes': accelerator.num_processes,
        'mixed_precision': accelerator.mixed_precision
    }


def _extract_trainer_config(config: XZTrainerConfig) -> TrackerConfigType:
    return {
        'minibatch_size': config.minibatch_size,
        'epochs': config.epochs,
        'gradient_clipping': config.gradient_clipping,
        'shuffle_train_dataset': config.dataloader_shuffle_train_dataset,
        'skip_nan_loss': config.skip_nan_loss
    }


def _extract_dataloader_config(dataloader: DataLoader) -> TrackerConfigType:
    return {
        'len': len(dataloader)
    }

def _extract_lr_scheduler_config(scheduler: LRSchedulerProtocol) -> TrackerConfigType:
    if isinstance(scheduler, AcceleratedScheduler):
        name = scheduler.scheduler.__class__.__name__
    else:
        name = scheduler.__class__.__name__
    return {
        'name': name
    }

def create_tracker_config(context: TrainContext) -> TrackerConfigType:
    out_dict = {
        **_prepend_tracker_config(context.trainer.config.tracker_config, 'config'),
        **_prepend_tracker_config(context.trainer.trainable.tracker_config(context), 'trainable'),
        **_prepend_tracker_config(_extract_trainer_config(context.trainer.config), 'trainer'),
        **_prepend_tracker_config(_extract_optimizer_config(context.optimizer), 'optimizer'),
        **_prepend_tracker_config(_extract_accelerator_config(context.trainer.accelerator), 'accelerator'),
        **_prepend_tracker_config(_extract_dataloader_config(context.data_loader), 'dataloader'),
        **_prepend_tracker_config(_extract_lr_scheduler_config(context.scheduler), 'scheduler')
    }
    return out_dict

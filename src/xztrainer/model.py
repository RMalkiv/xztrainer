import multiprocessing
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Callable, List, Optional, Any

from torch import nn, dtype
from torch.optim import Optimizer
from torch.utils.data.dataloader import default_collate

from xztrainer.logger import LoggingEngineConfig
from xztrainer.logger.stream import StreamLoggingEngineConfig


class SchedulerType(Enum):
    STEP = 'step'
    EPOCH = 'epoch'


class CheckpointType(Enum):
    MODEL_ONLY = 'model_only'
    XZTRAINER = 'xztrainer'


class LRSchedulerProtocol(Protocol):
    def step(self):
        ...

    def state_dict(self):
        ...

    def load_state_dict(self, state_dict):
        ...


@dataclass
class XZTrainerConfig:
    batch_size: int
    batch_size_eval: int
    epochs: int
    optimizer: Callable[[nn.Module], Optimizer]
    amp_dtype: Optional[dtype] = None
    experiment_name: str = 'master'
    gradient_clipping: float = 1.0
    scheduler: Optional[Callable[[Optimizer, int], LRSchedulerProtocol]] = None
    scheduler_type: Optional[SchedulerType] = None
    dataloader_num_workers: int = multiprocessing.cpu_count()
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    dataloader_shuffle_train_dataset: bool = True
    accumulation_batches: int = 1
    print_steps: int = 100
    eval_steps: int = 0
    skip_nan_loss: bool = True
    save_steps: int = 100
    save_keep_n: int = -1
    save_dir: str = 'checkpoint'
    collate_fn: Callable[[List[object]], Any] = default_collate
    logger: LoggingEngineConfig = StreamLoggingEngineConfig()

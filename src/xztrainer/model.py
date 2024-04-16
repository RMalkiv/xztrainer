import logging
import multiprocessing
from dataclasses import dataclass, field
from enum import Enum
import typing as t

from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import default_collate


ModelOutputType = t.Union[Tensor, list, tuple]
ModelOutputsType = dict[str, ModelOutputType]
DataType = t.Union[dict[str, t.Any], t.Iterable]


class CheckpointType(Enum):
    MODEL_ONLY = 'model_only'
    XZTRAINER = 'xztrainer'


class LRSchedulerProtocol(t.Protocol):
    def step(self):
        ...

    def state_dict(self):
        ...

    def load_state_dict(self, state_dict):
        ...


@dataclass
class XZTrainerConfig:
    experiment_name: str
    minibatch_size: int
    minibatch_size_eval: int
    epochs: int
    optimizer: t.Callable[[nn.Module], Optimizer]
    scheduler: t.Callable[[Optimizer, int], LRSchedulerProtocol]
    gradient_clipping: float = 1.0
    dataloader_num_workers: int = multiprocessing.cpu_count()
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    dataloader_shuffle_train_dataset: bool = True
    log_steps: int = 100
    eval_steps: int = 0
    skip_nan_loss: bool = True
    save_steps: int = 100
    save_keep_n: int = 3
    collate_fn: t.Callable[[list[object]], t.Any] = default_collate
    tracker_config: dict[str, t.Any] = field(default_factory=dict)
    logging_level: t.Union[int, None] = logging.INFO


@t.runtime_checkable
class MetricMultiOutputNamedProtocol(t.Protocol):
    @property
    def multi_output_names(self) -> t.List[str]:
        ...

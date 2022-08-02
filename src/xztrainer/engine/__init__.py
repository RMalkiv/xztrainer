from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Tuple, TYPE_CHECKING, Optional

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

if TYPE_CHECKING:
    from ..xztrainer import XZTrainer, TrainContext
    from ..model import LRSchedulerProtocol, SchedulerType


class TrainingEngine(metaclass=ABCMeta):
    @abstractmethod
    def wrap_model(self, model: Module, optimizer: Optional[Optimizer],
                   scheduler: Optional[LRSchedulerProtocol],
                   scheduler_type: Optional[SchedulerType]) -> Tuple[Module, Optimizer, LRSchedulerProtocol]:
        ...

    @abstractmethod
    def backward_pass(self, context: TrainContext, batch_i: int, loss: Tensor):
        ...


class TrainingEngineConfig(metaclass=ABCMeta):
    @abstractmethod
    def create_engine(self, trainer: XZTrainer) -> TrainingEngine:
        pass

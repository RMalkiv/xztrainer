from abc import ABCMeta, abstractmethod
from typing import Tuple

from torch import Tensor, Module
from torch.optim import Optimizer

from .. import XZTrainer, LRSchedulerProtocol, SchedulerType, TrainContext


class TrainingEngine(metaclass=ABCMeta):
    @abstractmethod
    def wrap_model(self, model: Module, optimizer: Optimizer, scheduler: LRSchedulerProtocol,
                   scheduler_type: SchedulerType) -> Tuple[Module, Optimizer, LRSchedulerProtocol]:
        ...

    @abstractmethod
    def backward_pass(self, context: TrainContext, batch_i: int, loss: Tensor):
        ...


class TrainingEngineConfig(metaclass=ABCMeta):
    @abstractmethod
    def create_engine(self, trainer: XZTrainer) -> TrainingEngine:
        pass

from abc import ABCMeta, abstractmethod

from .. import XZTrainer


class XZTrainerEngine(metaclass=ABCMeta):
    def __init__(self, trainer: XZTrainer):
        self.trainer = trainer

    @abstractmethod
    def wrap_model(self, model, optim, scheduler, scheduler_type):
        pass

    @abstractmethod
    def backward_pass(self, do_train, model, optimizer, scheduler, i, loss):
        pass


class XZTrainerEngineConfig(metaclass=ABCMeta):
    @abstractmethod
    def create_engine(self, trainer: XZTrainer) -> XZTrainerEngine:
        pass

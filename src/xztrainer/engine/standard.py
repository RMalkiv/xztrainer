from dataclasses import dataclass
from typing import Tuple

from torch import Tensor
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer

from ..xztrainer import XZTrainer, TrainContext
from ..model import LRSchedulerProtocol, SchedulerType
from . import TrainingEngineConfig, TrainingEngine


@dataclass
class StandardEngineConfig(TrainingEngineConfig):
    def create_engine(self, trainer: XZTrainer) -> TrainingEngine:
        return StandardEngine()


class StandardEngine(TrainingEngine):
    def wrap_model(self, model: Module, optimizer: Optimizer, scheduler: LRSchedulerProtocol,
                   scheduler_type: SchedulerType) -> Tuple[Module, Optimizer, LRSchedulerProtocol]:
        return model, optimizer, scheduler

    def backward_pass(self, context: TrainContext, batch_i: int, loss: Tensor):
        loss = loss / context.get_number_of_accumulations(batch_i)
        # multiple consecutive loss.backward() sum up the gradients, so we need to divide loss by num of accumulations
        loss.backward()
        if context.should_do_update_step(batch_i):
            clip_grad_norm_(context.model.parameters(), max_norm=1.0)
            context.optimizer.step()
            if context.scheduler is not None:
                context.scheduler.step()
            context.optimizer.zero_grad()

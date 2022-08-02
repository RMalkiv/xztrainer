from dataclasses import dataclass
from typing import Tuple, Optional

import torch
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
    def wrap_model(self, model: Module, optimizer: Optional[Optimizer],
                   scheduler: Optional[LRSchedulerProtocol],
                   scheduler_type: Optional[SchedulerType]) -> Tuple[Module, Optimizer, LRSchedulerProtocol]:
        return model, optimizer, scheduler

    def backward_pass(self, context: TrainContext, batch_i: int, loss: Tensor):
        loss = loss / context.get_number_of_accumulations(batch_i)
        # multiple consecutive loss.backward() sum up the gradients, so we need to divide loss by num of accumulations
        loss.backward()
        if context.should_do_update_step(batch_i):
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
            context.optimizer.step()
            if context.scheduler is not None:
                context.scheduler.step()
            context.optimizer.zero_grad()

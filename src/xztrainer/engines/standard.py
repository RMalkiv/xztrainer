from dataclasses import dataclass

from torch import nn

from .base import XZTrainerEngine, XZTrainerEngineConfig


@dataclass
class StandardEngineConfig(XZTrainerEngineConfig):
    def create_engine(self, trainer) -> XZTrainerEngine:
        return StandardEngine(trainer, self)


class StandardEngine(XZTrainerEngine):
    def __init__(self, trainer, config: StandardEngineConfig):
        super().__init__(trainer)
        self.config = config

    def wrap_model(self, model, optim, scheduler, scheduler_type):
        return model, optim, scheduler

    def backward_pass(self, do_train, model, optimizer, scheduler, i, loss):
        cfg = self.trainer.config
        loss /= cfg.accumulation_steps
        itm = loss.item()
        if do_train:
            loss.backward()
            if (i + 1) % cfg.accumulation_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
        return itm

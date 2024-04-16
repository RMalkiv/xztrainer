import abc
import typing as t
from dataclasses import dataclass
from enum import Enum

from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from xztrainer.logger import AccelerateLogger
from xztrainer.model import LRSchedulerProtocol

if t.TYPE_CHECKING:
    from xztrainer.trainer import XZTrainer, XZTrainState


class ContextType(Enum):
    TRAIN = 'train'
    EVAL = 'eval'
    INFERENCE = 'inference'


@dataclass
class BaseContext(abc.ABC):
    trainer: 'XZTrainer'
    data_loader: DataLoader
    model: nn.Module

    @property
    @abc.abstractmethod
    def context_type(self) -> ContextType:
        ...


@dataclass
class BaseTrainContext(BaseContext):
    logger: AccelerateLogger
    optimizer: Optimizer
    scheduler: LRSchedulerProtocol
    model_unwrapped: nn.Module
    train_state: 'XZTrainState'


@dataclass
class TrainContext(BaseTrainContext):
    sync_steps: int
    progress_bar: tqdm
    evaluate_data_loader: t.Union[DataLoader, None]

    @property
    def context_type(self) -> ContextType:
        return ContextType.TRAIN

    def should_perform_step_action(self, every_nth_step: int, current_step: int):
        if every_nth_step < 0:
            return False
        last_step = current_step == self.sync_steps
        if every_nth_step == 0:
            return last_step
        else:
            return (current_step % every_nth_step == 0) or last_step


@dataclass
class EvalContext(BaseTrainContext):
    @classmethod
    def from_train_context(cls: 'EvalContext', context: TrainContext):
        return cls(
            trainer=context.trainer,
            logger=context.logger,
            optimizer=context.optimizer,
            scheduler=context.scheduler,
            data_loader=context.evaluate_data_loader,
            model=context.model,
            model_unwrapped=context.model_unwrapped,
            train_state=context.train_state
        )

    @property
    def context_type(self) -> ContextType:
        return ContextType.EVAL


class InferContext(BaseContext):
    @property
    def context_type(self) -> ContextType:
        return ContextType.INFERENCE

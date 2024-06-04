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
    """
    Context type object. Can be used for checking current environment.
    """

    TRAIN = 'train'
    """Training context"""
    EVAL = 'eval'
    """Evaluation context"""
    INFERENCE = 'inference'
    """Inference context"""

@dataclass
class BaseContext(abc.ABC):
    """Base context that contains fields shared across training, evaluation and inference context."""

    trainer: 'XZTrainer'
    """Trainer instance"""

    data_loader: DataLoader
    """DataLoader instance"""

    model: nn.Module
    """Model instance. The model is wrapped using Accelerate"""

    @property
    @abc.abstractmethod
    def context_type(self) -> ContextType:
        ...


@dataclass
class BaseTrainContext(BaseContext):
    """Base train context that contains fields shared across training and evaluation context."""

    logger: AccelerateLogger
    """Logger instance"""

    optimizer: Optimizer
    """Optimizer instance"""

    scheduler: LRSchedulerProtocol
    """Learning rate scheduler instance"""

    model_unwrapped: nn.Module
    """Model instance. The model is not wrapped using Accelerate."""

    train_state: 'XZTrainState'
    """Custom state used by trainer"""


@dataclass
class TrainContext(BaseTrainContext):
    """Context used for training"""

    sync_steps: int
    """Total count of training steps"""

    progress_bar: tqdm
    """Progress bar used for displaying progress"""

    evaluate_data_loader: t.Union[DataLoader, None]
    """Evaluation DataLoader instance"""

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
    """Context used for evaluation"""

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
    """Context used for inference"""

    @property
    def context_type(self) -> ContextType:
        return ContextType.INFERENCE

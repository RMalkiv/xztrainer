import abc
import typing
from abc import ABC, abstractmethod

from torch import Tensor
from torchmetrics import Metric

if typing.TYPE_CHECKING:
    from xztrainer import XZTrainer
from xztrainer.context import BaseContext, ContextType, TrainContext, BaseTrainContext
from xztrainer.model import DataType, ModelOutputsType, TrackerConfigType


class XZTrainable(ABC):
    """
    Class representing a custom training logic.
    """

    @abstractmethod
    def step(
            self,
            context: BaseContext,
            data: DataType
    ) -> tuple[Tensor, ModelOutputsType]:
        """
        A function that:

        1. Performs a forward pass
        2. Computes loss function value
        3. Outputs loss value along with arbitrary model artifacts (used for metric computation further)

        Args:
            context: Current **train**/**evaluation** context.
            data: Model inputs from a collating function, moved to a target device

        Returns:
            A tuple containing (loss function value tensor, dictionary of model output artifacts)
        """

        ...

    def cut_model_outputs(
            self,
            context: BaseContext,
            model_outputs: ModelOutputsType,
            remainder: int
    ) -> ModelOutputsType:
        """
        Function used to cut out `model_outputs` based on a `remainder` value.

        Since accelerate repeats some last samples in both validation and training datasets
        (let's call these samples "padded samples") to ensure batches are split across GPUs evenly in distributed
        setups, you may want to cut out these padded samples before metric calculation.

        Default implementation just doesn't do anything because it does not know model output artifacts structure.

        If you doesn't implement this function, you may see a slight difference between calculated metrics for a distributed
        and non-distributed setups.

        Examples:
            >>> return {'predictions': model_outputs['predictions'][:-remainder]}

        Args:
            context: Current **train**/**evaluation** context
            model_outputs: Model output artifacts
            remainder: Number of last samples that are padded samples and need to be cut out.

        """
        return model_outputs

    @abc.abstractmethod
    def create_metrics(self, context_type: ContextType) -> dict[str, Metric]:
        """
        Creates a dictionary of {metric name: metric object}. Use metrics from `torchmetrics` package.
        You can create different metrics for different contexts (train, eval).

        Args:
            context_type: Current context type
        """
        ...

    @abc.abstractmethod
    def update_metrics(self, context_type: ContextType, model_outputs: ModelOutputsType, metrics: dict[str, Metric]):
        """
        Updates metric states after `step()` function using artifacts from it

        Args:
            context_type: Current context type
            model_outputs: Model output artifacts from `step()` call
            metrics: Metric objects from `create_metrics()` call
        """
        ...

    def calculate_composition_metrics(self, context_type: ContextType, metric_values: dict[str, float]) -> dict[str, float]:
        """
        Utility function to calculate metrics that depends on other metrics only.

        Args:
            context_type: Current context type
            metric_values: Calculated other metric values
        """
        return {}

    def on_load(self, context: TrainContext, step: int):
        """
        Callback that is called when a training is either started from scratch or started from a checkpoint.

        Use it, for example, to initialize weight tensors for entropy loss.

        Args:
            context: Current **train** context
            step: Current training step. Will be `1` when training is started from scratch. Will be other positive value when training is started from a checkponit.
        """
        pass

    def log(self, context: BaseTrainContext):
        """
        Callback that is called when trainer logs metrics values (both for training and evaluation).

        Args:
            context: Current **train**/**evaluation** context
        """
        pass

    def on_pre_update(self, context: TrainContext, step: int):
        """
        Callback that is called **before** an optimizer step occurs.

        Args:
            context: Current **train** context
            step: Current training step
        """
        pass

    def on_update(self, context: TrainContext, step: int):
        """
        Callback that is called **after** an optimizer step occurs.

        Args:
            context: Current **train** context
            step: Current training step
        """
        pass

    def tracker_config(self, context: TrainContext) -> TrackerConfigType:
        """
        Function returning additional hyperparameters that will be logged to experiment tracker. This function
        is called once when training starts.

        Args:
            context: Current **train** context

        Returns: Dictionary with additional parameter names and their values.

        """
        return {}

    def on_register_objects(self, trainer: 'XZTrainer'):
        """
        Callback that is called **after** preparing training pipeline objects with Accelerate and registering
        trainer state for checkpointing. You can register your custom objects for checkpointing here.
        Args:
            trainer: Trainer instance.
        """
        pass

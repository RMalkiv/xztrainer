import abc
from abc import ABC, abstractmethod

from torch import Tensor
from torchmetrics import Metric

from xztrainer.context import BaseContext, ContextType, TrainContext, BaseTrainContext
from xztrainer.model import DataType, ModelOutputsType


class XZTrainable(ABC):
    @abstractmethod
    def step(
            self,
            context: BaseContext,
            data: DataType
    ) -> tuple[Tensor, ModelOutputsType]:
        ...

    def cut_model_outputs(
            self,
            context: BaseContext,
            model_outputs: ModelOutputsType,
            remainder: int
    ) -> ModelOutputsType:
        return model_outputs

    @abc.abstractmethod
    def create_metrics(self, context_type: ContextType) -> dict[str, Metric]:
        ...

    @abc.abstractmethod
    def update_metrics(self, context_type: ContextType, model_outputs: dict[str, list], metrics: dict[str, Metric]):
        ...

    def calculate_composition_metrics(self, context_type: ContextType, metric_values: dict[str, float]) -> dict[str, float]:
        return {}

    def on_load(self, context: TrainContext, step: int):
        pass

    def log(self, context: BaseTrainContext):
        pass

    def on_pre_update(self, context: TrainContext, step: int):
        pass

    def on_update(self, context: TrainContext, step: int):
        pass

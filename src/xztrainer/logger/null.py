from types import TracebackType
from typing import Optional, Type

from xztrainer.logger.base import LoggingEngine, LoggingEngineConfig, ClassifierType


class NullLoggingEngine(LoggingEngine):
    def log_scalar(self, classifier: ClassifierType, value: float):
        pass

    def flush(self):
        pass


class NullLoggingEngineConfig(LoggingEngineConfig):
    def create_engine(self, experiment_name: str) -> NullLoggingEngine:
        return NullLoggingEngine()

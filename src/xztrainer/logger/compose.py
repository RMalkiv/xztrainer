from types import TracebackType
from typing import Optional, Type, List

from . import LoggingEngine, LoggingEngineConfig, ClassifierType


class ComposeLoggingEngine(LoggingEngine):
    def __init__(self, experiment_name: str, config: 'ComposeLoggingEngineConfig'):
        super().__init__()

        self.engines = [engine.create_engine(experiment_name) for engine in config.engines]

    def log_scalar(self, classifier: ClassifierType, value: float):
        for engine in self.engines:
            engine.log_scalar(classifier, value)

    def flush(self):
        for engine in self.engines:
            engine.flush()

    def update_time_step(self, time_step: int):
        super().update_time_step(time_step)
        for engine in self.engines:
            engine.update_time_step(time_step)

    def update_top_classifier(self, classifier: ClassifierType):
        super().update_top_classifier(classifier)
        for engine in self.engines:
            engine.update_top_classifier(classifier)

    def __enter__(self) -> 'LoggingEngine':
        return self

    def __exit__(self, __exc_type: Optional[Type[BaseException]], __exc_value: Optional[BaseException], __traceback: Optional[TracebackType]):
        for engine in self.engines:
            engine.__exit__(__exc_type, __exc_value, __traceback)


class ComposeLoggingEngineConfig(LoggingEngineConfig):
    def __init__(self, *engines: LoggingEngineConfig):
        self.engines = engines

    def create_engine(self, experiment_name: str) -> ComposeLoggingEngine:
        return ComposeLoggingEngine(experiment_name, self)

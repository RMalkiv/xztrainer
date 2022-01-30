from abc import ABCMeta, abstractmethod
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Union, Iterable, Optional, Type

ClassifierType = Union[str, Iterable[str]]


def convert_classifier(classifier: ClassifierType) -> Iterable[str]:
    if isinstance(classifier, str):
        return classifier,
    elif isinstance(classifier, Iterable):
        return tuple(classifier)
    else:
        raise ValueError(f'Invalid classifier type: {type(classifier)}')


class LoggingEngine(AbstractContextManager, metaclass=ABCMeta):
    def __init__(self):
        self._time_step = 0
        self._top_classifier = ()

    @abstractmethod
    def log_scalar(self, classifier: ClassifierType, value: float):
        ...

    @abstractmethod
    def flush(self):
        ...

    def update_time_step(self, time_step: int):
        self._time_step = time_step

    def update_top_classifier(self, classifier: ClassifierType):
        self._top_classifier = convert_classifier(classifier)

    def __enter__(self) -> 'LoggingEngine':
        return self

    def __exit__(self, __exc_type: Optional[Type[BaseException]], __exc_value: Optional[BaseException],
                 __traceback: Optional[TracebackType]):
        pass


class LoggingEngineConfig(metaclass=ABCMeta):
    @abstractmethod
    def create_engine(self, experiment_name: str) -> LoggingEngine:
        pass

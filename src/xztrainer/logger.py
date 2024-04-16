from abc import abstractmethod
from typing import Union, Iterable

from accelerate import Accelerator

ClassifierType = Union[str, Iterable[str]]


def convert_classifier(classifier: ClassifierType) -> Iterable[str]:
    if isinstance(classifier, str):
        return classifier,
    elif isinstance(classifier, Iterable):
        return tuple(classifier)
    else:
        raise ValueError(f'Invalid classifier type: {type(classifier)}')


class AccelerateLogger:
    def __init__(self, accelerator: Accelerator):
        self._time_step = 0
        self._top_classifier = ()
        self._accelerator = accelerator

    @abstractmethod
    def log_scalar(self, classifier: ClassifierType, value: float):
        classifier = '/'.join(self._top_classifier + convert_classifier(classifier))
        step = self._time_step
        self._accelerator.log({classifier: value}, step=step)

    def update_time_step(self, time_step: int):
        self._time_step = time_step

    def update_top_classifier(self, classifier: ClassifierType):
        self._top_classifier = convert_classifier(classifier)

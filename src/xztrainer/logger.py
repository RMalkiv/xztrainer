from typing import Union, Iterable

from accelerate import Accelerator

ClassifierType = Union[str, Iterable[str]]
"""A type that represents a classifier, string ("perplexity")  or collection of strings (["perplexity", "books"])"""


def _convert_classifier(classifier: ClassifierType) -> Iterable[str]:
    if isinstance(classifier, str):
        return classifier,
    elif isinstance(classifier, Iterable):
        return tuple(classifier)
    else:
        raise ValueError(f'Invalid classifier type: {type(classifier)}')


class AccelerateLogger:
    """
    A wrapper for Accelerate logging API that is able to update its current time step and top-level classifier
    (train/eval/etc...) by xztrainer loop. It can be used within
    """

    def __init__(self, accelerator: Accelerator):
        self._time_step = 0
        self._top_classifier = ()
        self._accelerator = accelerator

    def log_scalar(self, classifier: ClassifierType, value: float):
        """
        Logs a scalar value
        :param classifier: Value classifier (e.g. "loss" or "perplexity")
        :param value: Scalar value
        """
        classifier = '/'.join(self._top_classifier + _convert_classifier(classifier))
        step = self._time_step
        self._accelerator.log({classifier: value}, step=step)

    def update_time_step(self, time_step: int):
        """
        Sets current time step of this logger (used by xztrainer internals).
        :param time_step: Time step to set
        """
        self._time_step = time_step

    def update_top_classifier(self, classifier: ClassifierType):
        """
        Sets top-level classifier of this logger (used by xztrainer internals).
        :param classifier: Classifier to set
        """
        self._top_classifier = _convert_classifier(classifier)

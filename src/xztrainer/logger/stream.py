import sys
from dataclasses import dataclass
from typing import TextIO, Dict

from . import LoggingEngine, LoggingEngineConfig, convert_classifier, ClassifierType


class StreamLoggingEngine(LoggingEngine):
    _buffer_scalar: Dict[str, float]

    def __init__(self, config: 'StreamLoggingEngineConfig'):
        super().__init__()

        self._file = config.output
        self._round_to = config.round_scalars_to
        self._buffer_scalar = {}

    def log_scalar(self, classifier: ClassifierType, value: float):
        self._buffer_scalar['/'.join(self._top_classifier + convert_classifier(classifier))] = value

    def flush(self):
        scalars = '\t'.join([f'{k}={round(v, self._round_to)}' for k, v in self._buffer_scalar.items()])
        print(f'[{self._time_step}] {scalars}', file=self._file)
        self._buffer_scalar.clear()


@dataclass
class StreamLoggingEngineConfig(LoggingEngineConfig):
    output: TextIO = sys.stdout
    round_scalars_to: int = 5

    def create_engine(self, experiment_name: str) -> StreamLoggingEngine:
        return StreamLoggingEngine(self)

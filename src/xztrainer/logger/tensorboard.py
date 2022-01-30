from dataclasses import dataclass
from types import TracebackType
from typing import Optional, Type

from torch.utils.tensorboard import SummaryWriter

from . import LoggingEngine, LoggingEngineConfig, ClassifierType, convert_classifier


class TensorboardLoggingEngine(LoggingEngine):
    def __init__(self, experiment_name: str, config: 'TensorboardLoggingEngineConfig'):
        super().__init__()

        self.writer = SummaryWriter(f'{config.output_dir}/{experiment_name}', max_queue=config.max_queue, flush_secs=config.flush_secs)

    def log_scalar(self, classifier: ClassifierType, value: float):
        self.writer.add_scalar('/'.join(self._top_classifier + convert_classifier(classifier)), value, self._time_step)

    def flush(self):
        self.writer.flush()

    def __exit__(self, __exc_type: Optional[Type[BaseException]], __exc_value: Optional[BaseException],
                 __traceback: Optional[TracebackType]):
        self.writer.close()


@dataclass
class TensorboardLoggingEngineConfig(LoggingEngineConfig):
    output_dir: str = 'runs'
    flush_secs: int = 30
    max_queue: int = 10

    def create_engine(self, experiment_name: str) -> TensorboardLoggingEngine:
        return TensorboardLoggingEngine(experiment_name, self)

import logging
import multiprocessing
import typing as t
from dataclasses import dataclass, field

from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import default_collate

ModelOutputType = t.Union[Tensor, list, tuple]
"""Type representing a model output artifact. Can be a tensor, a nested list or nested tuple of tensors."""

ModelOutputsType = dict[str, ModelOutputType]
"""Type representing multiple model output artifacts: a dictionary with artifact names as keys and model output artifacts as values."""

DataType = t.Union[dict[str, t.Any], t.Iterable]
"""Type representing model input data. Can be a nested object of dictionaries, lists, sets, tuples containing arbitrary types such as tensors or strings."""

TrackerConfigType = dict[str, t.Union[float, int, bool, str, None]]
"""Type representing dictionary with parameters logged to experiment tracker"""


class LRSchedulerProtocol(t.Protocol):
    """
    A protocol (https://peps.python.org/pep-0544/) representing a learning rate scheduler.
    """

    def step(self):
        """Function that updates a scheduler each training step."""
        ...

    def state_dict(self):
        """Function that returns scheduler's internal state dict."""
        ...

    def load_state_dict(self, state_dict):
        """Function that loads this scheduler's state from a state dict."""
        ...


@dataclass
class XZTrainerConfig:
    """
    A configuration for XZTrainer training setup
    """

    experiment_name: str
    """Name of current experiment. Used for logging and checkpoint saving"""

    minibatch_size: int
    """
    Minibatch size used for training.
    In case of a multi-gpu or gradient accumulated train setup, it is a physical batch size for a single GPU.
    """

    minibatch_size_eval: int
    """
    Minibatch size used for evaluation.
    In case of a multi-gpu or gradient accumulated train setup, it is a physical batch size for a single GPU.
    """

    epochs: int
    """
    Number of epochs to train for.
    """

    optimizer: t.Callable[[nn.Module], Optimizer]
    """
    A callable that creates an Optimizer object from a Module object.
    Module object can be used to pass its parameters to Optimizer constructor, for example.
    
    Examples:
        >>> optimizer=lambda module: AdamW(module.parameters(), lr=1e-3, weight_decay=1e-4)
    """

    scheduler: t.Callable[[Optimizer, int], LRSchedulerProtocol]
    """
    A callable that creates a Learning Rate Scheduler object from an Optimizer and total number of training steps.
    
    Examples:
        >>> scheduler=lambda optim, total_steps: transformers.get_linear_schedule_with_warmup(optim, int(0.1 * total_steps), total_steps)
    """

    gradient_clipping: float = 1.0
    """
    Max gradient norm used for gradient clipping. Can be set to `0` to disable gradient clipping.
    """

    dataloader_num_workers: int = multiprocessing.cpu_count()
    """
    Number of worker processes used for data loaders.
    """

    dataloader_pin_memory: bool = True
    """
    Boolean value indicating whether to pin memory for data loaders.
    """

    dataloader_persistent_workers: bool = True
    """
    Boolean value indicating whether to keep workers alive after each epoch.
    """

    dataloader_shuffle_train_dataset: bool = True
    """
    Boolean value indicating whether to shuffle a training dataset within a dataloader.
    """

    log_steps: int = 100
    """
    Controls frequency of logging model metrics.
    
    Can be either:
    
    - `-1` - to disable logging
    
    - `0` - to log only after an epoch finishes
    
    - Any positive value - to log at each Nth training step
    """

    eval_steps: int = 0
    """
    Controls frequency of running a model on evaluation set.

    Can be either:

    - `-1` - to disable evaluation

    - `0` - to evaluate only after an epoch finishes

    - Any positive value - to evaluate at each Nth training step
    """

    skip_nan_loss: bool = True
    """
    Boolean value indicating whether to skip update when a NaN loss value occurs.
    """

    save_steps: int = 100
    """
    Controls frequency of saving a model and trainer state.

    Can be either:

    - `-1` - to disable saving

    - `0` - to save only after an epoch finishes

    - Any positive value - to save at each Nth training step
    """

    save_keep_n: int = 3
    """
    Number of saves to keep while training. After a new save finished, all the other saves will be deleted except for `save_keep_n` last ones. Set to `-1` to disable save removal.
    """

    collate_fn: t.Callable[[list[object]], t.Any] = default_collate
    """
    A function used to collate multiple dataset items into a single batch.
    """

    tracker_config: dict[str, t.Any] = field(default_factory=dict)
    """
    Arbitrary hyperparameters logged to experiment tracker.
    """

    tracker_init_kwargs: dict[str, t.Any] = field(default_factory=dict)
    """
    Arbitrary keyword arguments used for Accelerate experiment tracker. Directly passed to `accelerator.init_trackers(...)`. See [Accelerate docs](https://huggingface.co/docs/accelerate/en/usage_guides/tracking)
    """

    tracker_logging_dir: t.Optional[str] = None
    """
    Logging directory (or remote address) for Accelerate experiment tracker. If `None` - it will be set to `{project_dir}/runs` automatically.
    """

    logging_level: t.Union[int, None] = logging.INFO
    """
    Logging level that xztrainer initializes Python loggers with. Set to `None` to make xztrainer not do anything with Python loggers.
    """


@t.runtime_checkable
class MetricMultiOutputNamedProtocol(t.Protocol):
    @property
    def multi_output_names(self) -> t.List[str]:
        ...

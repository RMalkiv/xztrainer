from collections import OrderedDict
from typing import Mapping, Any

from accelerate import Accelerator
from torch import Tensor

from xztrainer.model import ModelOutputType, DataType


def detach_tensor(output: ModelOutputType, move_to_cpu: bool) -> ModelOutputType:
    """
    Recursively detaches each tensor in a model output structure.

    Args:
        output: Model output to detach tensors in
        move_to_cpu: Whether to move a tensor to CPU after detaching

    Returns:

    """
    if isinstance(output, Tensor):
        output = output.detach()
        if move_to_cpu:
            output = output.cpu()
        return output
    elif isinstance(output, list):
        return [detach_tensor(x, move_to_cpu) for x in output]
    elif isinstance(output, tuple):
        return tuple(detach_tensor(x, move_to_cpu) for x in output)
    else:
        return output


def move_data_to_device(data: Any, accelerator: Accelerator) -> DataType:
    """
    Recursively moves each tensor in a data structure to device specified by `Accelerator` object.
    Args:
        data: Structure to move tensors in
        accelerator: `Accelerator` object

    Returns:

    """
    if isinstance(data, Tensor):
        return data.to(accelerator.device)
    elif isinstance(data, OrderedDict):
        return OrderedDict({k: move_data_to_device(v, accelerator) for k, v in data.items()})
    elif isinstance(data, Mapping):
        return {k: move_data_to_device(v, accelerator) for k, v in data.items()}
    elif isinstance(data, tuple):
        return tuple(move_data_to_device(v, accelerator) for v in data)
    elif isinstance(data, list):
        return [move_data_to_device(v, accelerator) for v in data]
    elif isinstance(data, set):
        return set(move_data_to_device(v, accelerator) for v in data)
    else:
        return data

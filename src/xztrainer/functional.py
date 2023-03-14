from typing import Callable

from torch import nn


def count_parameters(module: nn.Module, parameter_predicate: Callable[[nn.Parameter], bool] = lambda p: True):
    return sum(param.numel() for param in module.parameters() if parameter_predicate(param))

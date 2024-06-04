from typing import Callable

from torch import nn


def count_parameters(module: nn.Module, parameter_predicate: Callable[[nn.Parameter], bool] = lambda p: True) -> int:
    """
    Filters all the module parameter tensors based on a given predicate and counts a total parameter count after that.
    :param module: Module to count parameters in
    :param parameter_predicate: Predicate to filter out model parameters to be counted
    :return: Parameter count
    """
    return sum(param.numel() for param in module.parameters() if parameter_predicate(param))

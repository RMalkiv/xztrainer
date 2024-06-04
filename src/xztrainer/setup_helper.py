import torch
from accelerate.utils import set_seed


def set_seeds(seed: int):
    """
    Sets all the RNG states (python, torch, numpy). Internally calls accelerate's set_seed, left only for backward compatibility.

    Args:
        seed: RNG state
    """
    set_seed(seed)


def enable_tf32():
    """
    Enables tensorfloat32 computations for pytorch.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

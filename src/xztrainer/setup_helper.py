import random

import torch
from accelerate.utils import set_seed


def set_seeds(seed: int):
    random.seed(seed)
    set_seed(seed)


def enable_tf32():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

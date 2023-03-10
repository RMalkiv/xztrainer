import random
from typing import Any, Dict

import torch


def _get_rng_states() -> Dict[str, Any]:
    try:
        import numpy as np
        numpy_state = np.random.get_state()
    except ImportError:
        numpy_state = None
    return {
        'torch': torch.get_rng_state(),
        'numpy': numpy_state,
        'python': random.getstate()
    }


def _set_rng_states(states: Dict[str, Any]):
    if states['torch'] is not None:
        torch.set_rng_state(states['torch'].cpu())
    if states['numpy'] is not None:
        try:
            import numpy as np
            np.random.set_state(states['numpy'])
        except ImportError:
            pass
    if states['python'] is not None:
        random.setstate(states['python'])

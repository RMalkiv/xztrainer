import random
from typing import Iterator, List, Any, Dict

from torch.utils.data import Sampler, Dataset


class ReusableSequentialSampler(Sampler[int]):
    _indices: List[int]

    def __init__(self, indices: List[int], shift_steps: int) -> None:
        super().__init__(indices)
        self._indices = indices
        self._shift_steps = shift_steps

    def __iter__(self) -> Iterator[int]:
        return iter(self._indices[self._shift_steps:])

    def __len__(self) -> int:
        return max(len(self._indices) - self._shift_steps, 0)

    def save_state(self, batch_i: int, batch_size: int) -> Dict[str, Any]:
        return {
            'indices': self._indices,
            'shift_steps': (batch_i + 1) * batch_size
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]):
        return cls(state['indices'], state['shift_steps'])

    @classmethod
    def new(cls, data: Dataset, shuffle: bool):
        indices = list(range(len(data)))
        if shuffle:
            random.shuffle(indices)
        return cls(indices, 0)

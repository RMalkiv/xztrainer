import random
from typing import Sized, Iterator, List, Optional

from torch.utils.data import Sampler


class ReusableSequentialSampler(Sampler[int]):
    _indices: List[int]

    def __init__(self, indices: List[int]) -> None:
        super().__init__(indices)
        self._indices = indices

    def __iter__(self) -> Iterator[int]:
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)

    def save_state(self, batch_i: int, batch_size: int) -> List[int]:
        return self._indices[(batch_i + 1) * batch_size:]

    @classmethod
    def from_state(cls, state: List[int]):
        return cls(state)

    @classmethod
    def new(cls, data: Sized, shuffle: bool):
        indices = list(range(len(data)))
        if shuffle:
            random.shuffle(indices)
        return cls(indices)

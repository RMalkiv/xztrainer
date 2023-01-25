from typing import Sized, Iterator

from torch.utils.data import Sampler


class ReusableSequentialSampler(Sampler[int]):
    _data_len: int

    def __init__(self, data: Sized, start_from_i: int) -> None:
        super().__init__(data)
        self._data_len = len(data)
        self._start_from_i = start_from_i

    def __iter__(self) -> Iterator[int]:
        return iter(range(self._start_from_i, self._data_len))

    def __len__(self) -> int:
        return self._data_len - self._start_from_i

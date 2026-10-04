import itertools
from typing import Iterator, Optional

import torch
from torch.utils.data import Dataset, Sampler


class FixedLengthSampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        samples_per_gpu: int = 1,
        iters_per_epoch: int = 5000,
        seed: int = 0,
        dataset_already_sharded: bool = False,
        global_size: Optional[int] = None,
        local_rank: Optional[int] = None,
    ):
        if dataset_already_sharded:
            self.global_size = 1
            self.local_rank = 0
        else:
            self.global_size = global_size or 1
            self.local_rank = local_rank or 0
            print(f"FixedLengthSampler: global_size={self.global_size}, local_rank={self.local_rank}")

        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.iters_per_epoch = iters_per_epoch
        self.seed = seed
        self.epoch = 0

        assert iters_per_epoch is not None and iters_per_epoch > 0
        self.num_samples = int(iters_per_epoch) * self.samples_per_gpu

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)
        print(
            f"[{self.__class__.__name__}] [local_rank={self.local_rank}/{self.global_size}] set seed to {self.epoch + self.seed}"
        )

        def global_index_stream() -> Iterator[int]:
            while True:
                yield torch.randint(0, len(self.dataset), (1,), generator=g).item() # type: ignore[arg-type]

        start = self.local_rank if self.global_size and self.local_rank is not None else 0
        print(f"[{self.__class__.__name__}] [local_rank={self.local_rank}/{self.global_size}] sample start = {start}")
        step = self.global_size if self.global_size else 1
        return itertools.islice(global_index_stream(), start, start + self.num_samples * step, step)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        print(f"[{self.__class__.__name__}] set epoch => {epoch}")
        self.epoch = epoch

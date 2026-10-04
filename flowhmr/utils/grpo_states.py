from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class GRPOTrainingStates:

    iters_per_group: int
    group_size: int
    max_timesteps: int
    cur_timestep: int = 0
    cur_iter_in_group: int = 0
    sample_strategy: str = "progressive"
    prog_overlap: bool = False
    prog_overlap_step: int = 1
    max_iters_per_group: int = None
    min_iters_per_group: int = None
    roll_back: bool = False
    exp_decay_thre_timestep: int = 13
    exp_decay_k: float = 0.1

    def set_params(self, params: dict):
        for key, value in params.items():
            setattr(self, key, value)

    def __post_init__(self):
        if self.sample_strategy == "decay":
            if self.max_iters_per_group is None:
                self.max_iters_per_group = self.iters_per_group
            if self.min_iters_per_group is None:
                self.min_iters_per_group = max(1, self.iters_per_group // 4)
        self.init_timestep = self.cur_timestep

    def get_dynamic_iters_per_group(self) -> int:
        if self.sample_strategy != "decay":
            return self.iters_per_group
        progress = self.cur_timestep / self.max_timesteps
        current_iters = int(
            self.max_iters_per_group * (1 - progress) + self.min_iters_per_group * progress
        )
        return max(self.min_iters_per_group, current_iters)

    def get_exp_decay_iters_per_group(self) -> int:
        if self.sample_strategy != "exp_decay":
            return self.iters_per_group
        relu_value = max(0, self.cur_timestep - self.exp_decay_thre_timestep)
        decay_value = self.iters_per_group * np.exp(-self.exp_decay_k * relu_value)
        return int(np.ceil(decay_value))

    def _advance_window(self) -> None:
        if self.prog_overlap:
            self.cur_timestep += self.prog_overlap_step
        else:
            self.cur_timestep += self.group_size
        if self.cur_timestep > self.max_timesteps - 1:
            if self.roll_back:
                self.roll_back_start()
            else:
                self.cur_timestep = self.max_timesteps - 1

    def update_iteration(self, seed=None) -> None:
        if self.sample_strategy == "progressive":
            self.cur_iter_in_group += 1
            if self.cur_iter_in_group >= self.iters_per_group:
                self.cur_iter_in_group = 0
                self._advance_window()
        elif self.sample_strategy == "random":
            rng = np.random.default_rng(seed)
            self.cur_timestep = int(rng.integers(0, self.max_timesteps - self.group_size + 1))
        elif self.sample_strategy == "decay":
            self.cur_iter_in_group += 1
            if self.cur_iter_in_group >= self.get_dynamic_iters_per_group():
                self.cur_iter_in_group = 0
                self._advance_window()
        elif self.sample_strategy == "exp_decay":
            self.cur_iter_in_group += 1
            if self.cur_iter_in_group >= self.get_exp_decay_iters_per_group():
                self.cur_iter_in_group = 0
                self._advance_window()
        else:
            raise ValueError(f"Invalid sample strategy: {self.sample_strategy}")

    def roll_back_start(self) -> None:
        self.cur_timestep = self.init_timestep
        self.cur_iter_in_group = 0

    def get_current_timesteps(self) -> List[int]:
        return list(range(self.cur_timestep, min(self.cur_timestep + self.group_size, self.max_timesteps)))

    def is_training_complete(self) -> bool:
        if self.sample_strategy in ["progressive", "decay"]:
            return self.cur_timestep >= self.max_timesteps
        return False

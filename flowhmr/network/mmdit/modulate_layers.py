from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .bricks import get_activation_layer


class ModulateDiT(nn.Module):
    def __init__(self, feat_dim: int, factor: int, act_type: str = "silu", with_bottleneck: bool = False):
        super().__init__()
        self.act = get_activation_layer(act_type)()
        self.with_bottleneck = with_bottleneck
        if with_bottleneck:
            down_sample_ratio = 8
            assert feat_dim % down_sample_ratio == 0, f"feat_dim {feat_dim} must be divisible by {down_sample_ratio}"
            self.down_proj = nn.Linear(feat_dim, feat_dim // down_sample_ratio, bias=True)
            self.linear = nn.Linear(feat_dim // down_sample_ratio, factor * feat_dim, bias=True)
        else:
            self.linear = nn.Linear(feat_dim, factor * feat_dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        if self.with_bottleneck:
            return self.linear(self.act(self.down_proj(self.act(x))))
        else:
            return self.linear(self.act(x))


def modulate(x: Tensor, shift: Optional[Tensor] = None, scale: Optional[Tensor] = None) -> Tensor:
    if shift is not None and scale is not None:
        assert (
            len(x.shape) == len(shift.shape) == len(scale.shape)
        ), f"x, shift, scale must have the same number of dimensions, but got {x.shape}, {shift.shape} and {scale.shape}"

    orig_dtype = x.dtype
    x = x.float()
    if shift is not None:
        shift = shift.float()
    if scale is not None:
        scale = scale.float()

    if shift is not None and scale is not None:
        return (x * (1 + scale) + shift).to(dtype=orig_dtype)
    elif shift is not None:
        return (x + shift).to(dtype=orig_dtype)
    elif scale is not None:
        return (x * (1 + scale)).to(dtype=orig_dtype)
    else:
        return x.to(dtype=orig_dtype)


def apply_gate(x: Tensor, gate: Optional[Tensor] = None, tanh: bool = False) -> Tensor:
    if gate is not None:
        assert len(x.shape) == len(
            gate.shape
        ), f"x, gate must have the same number of dimensions, but got {x.shape} and {gate.shape}"
    if gate is None:
        return x
    if tanh:
        return x * gate.tanh()
    else:
        return x * gate

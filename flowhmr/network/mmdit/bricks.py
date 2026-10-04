from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def get_activation_layer(act_type: str) -> Callable[[], nn.Module]:
    if act_type == "gelu":
        return lambda: nn.GELU()
    elif act_type == "gelu_tanh":
        return lambda: nn.GELU(approximate="tanh")
    elif act_type == "relu":
        return nn.ReLU
    elif act_type == "silu":
        return nn.SiLU
    else:
        raise ValueError(f"Unknown activation type: {act_type}")


def get_norm_layer(norm_type: Optional[str]):
    if norm_type == "layer":
        return LayerNormFP32
    elif norm_type == "rms":
        return RMSNorm
    elif norm_type == "none" or norm_type is None:
        return nn.Identity
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, elementwise_affine=True, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        norm_in = x.to(dtype=torch.float32) if x.dtype in (torch.float16, torch.bfloat16) else x
        output = self._norm(norm_in).to(dtype=x.dtype)
        if hasattr(self, "weight"):
            output = output * self.weight
        return output


class LayerNormFP32(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        x_fp32 = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        y = F.layer_norm(x_fp32, self.normalized_shape, weight, bias, self.eps)
        return y.to(dtype=x.dtype)

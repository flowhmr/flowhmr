from typing import Any, Optional

import torch
from torch import nn


def get_module_device(module: nn.Module) -> torch.device:
    try:
        next(module.parameters())
    except StopIteration:
        raise ValueError("The input module should contain parameters.")

    if next(module.parameters()).is_cuda:
        return torch.device(next(module.parameters()).get_device())

    return torch.device("cpu")


def cast_to_module_dtype(x: Any, module: torch.nn.Module, device: Optional[torch.device] = None) -> Any:
    if not torch.is_tensor(x):
        return x

    ref = None
    for param in module.parameters(recurse=True):
        ref = param
        break
    if ref is None:
        for buffer in module.buffers(recurse=True):
            ref = buffer
            break

    if ref is None:
        return x.to(device=device) if device is not None else x

    target_device = device if device is not None else ref.device
    target_dtype = ref.dtype if x.is_floating_point() else x.dtype
    return x.to(device=target_device, dtype=target_dtype)

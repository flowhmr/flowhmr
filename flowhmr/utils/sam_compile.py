"""Optional compilation of the SAM backbone; decoder hooks remain eager."""

from functools import wraps
import os
import warnings

import torch


def resolve_sam_compile(device, use_compile=None):
    """Resolve the explicit option or environment, validating before model load."""
    if use_compile is None:
        value = os.environ.get("FLOWHMR_SAM_COMPILE", "0").strip().lower()
        if value in ("0", "false", "no", "off", ""):
            use_compile = False
        elif value in ("1", "true", "yes", "on"):
            use_compile = True
        else:
            raise ValueError("FLOWHMR_SAM_COMPILE must be 0/1, false/true, no/yes or off/on")
    elif not isinstance(use_compile, bool):
        raise TypeError("use_compile must be a bool or None")
    if use_compile:
        if torch.device(device).type != "cuda":
            raise ValueError("SAM backbone compilation requires a CUDA device")
        if not callable(getattr(torch, "compile", None)):
            raise RuntimeError("SAM backbone compilation requires torch.compile")
    return use_compile


def _contains_out_of_memory(error):
    # Dynamo may wrap an OOM in BackendCompilerFailed or attach it as a cause.
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, (torch.OutOfMemoryError, MemoryError)):
            return True
        pending.extend((current.__cause__, current.__context__,
                        getattr(current, "inner_exception", None)))
    return False


def enable_sam_compile(sam_model, device):
    """Compile only ``backbone.forward`` in place; return whether it was enabled.

    Compilation is lazy. ``backbone._flowhmr_compile_state['status']`` becomes
    ``active`` only after a successful compiled call, or ``failed`` after an
    eager fallback. A failed compilation is not retried on subsequent calls.
    Runtime/model errors and out-of-memory errors continue to the caller.
    """
    resolve_sam_compile(device, True)
    from torch._dynamo.exc import (
        BackendCompilerFailed, InternalTorchDynamoError, InvalidBackend, Unsupported,
    )

    compilation_errors = (BackendCompilerFailed, InternalTorchDynamoError, InvalidBackend, Unsupported)
    backbone = sam_model.backbone
    previous = getattr(backbone, "_flowhmr_compile_state", None)
    if previous is not None:
        return previous["status"] != "failed"
    eager_forward = backbone.forward
    state = {"status": "pending", "error": None}
    backbone._flowhmr_compile_state = state

    def fallback(error):
        if _contains_out_of_memory(error):
            raise error
        detail = f"{type(error).__name__}: {error}"[:500]
        state.update(status="failed", error=detail)
        backbone.forward = eager_forward
        warnings.warn(
            f"SAM backbone torch.compile failed; using eager inference. {detail}",
            RuntimeWarning, stacklevel=3,
        )

    try:
        # In PyTorch 2.5, mode and options are mutually exclusive. Omitting
        # mode uses default mode; explicitly disable CUDA graph capture.
        compiled_forward = torch.compile(
            eager_forward, fullgraph=True, dynamic=False,
            options={"triton.cudagraphs": False},
        )
    except compilation_errors as error:
        fallback(error)
        return False

    @wraps(eager_forward)
    def forward(*args, **kwargs):
        if state["status"] == "failed":
            return eager_forward(*args, **kwargs)
        try:
            result = compiled_forward(*args, **kwargs)
        except compilation_errors as error:
            fallback(error)
            return eager_forward(*args, **kwargs)
        state["status"] = "active"
        return result

    backbone.forward = forward
    return True

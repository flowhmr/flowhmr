from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    import torch
except Exception: # pragma: no cover
    torch = None


def _global_joints_from_params(
    body_model: Any,
    rot6d: "torch.Tensor",
    shapes: "torch.Tensor",
    trans: "torch.Tensor",
    joint_num: int = 52,
) -> np.ndarray:
    device = rot6d.device
    if hasattr(body_model, "parameters"):
        _p = next(body_model.parameters(), None)
        if _p is not None:
            device = _p.device
    rot6d = rot6d.to(device=device, dtype=torch.float32)
    trans = trans.to(device=device, dtype=torch.float32)
    shapes = shapes.to(device=device, dtype=torch.float32)
    T, J = rot6d.shape[0], rot6d.shape[1]
    if shapes.ndim == 1:
        shapes_flat = shapes.unsqueeze(0).expand(T, -1)
    elif shapes.shape[0] == 1:
        shapes_flat = shapes.expand(T, -1)
    else:
        shapes_flat = shapes
    with torch.no_grad():
        out = body_model({
            "rot6d": rot6d.reshape(T, J, 6),
            "shapes": shapes_flat.reshape(T, -1),
            "trans": trans.reshape(T, 3),
        })
    return out["keypoints3d"].reshape(T, -1, 3)[:, :joint_num, :].detach().float().cpu().numpy()


def compute_mpjpe_reward_from_params(
    body_model: Any,
    output: dict,
    gt_joints: np.ndarray,
    *,
    length: Optional[int] = None,
    joint_range: Optional[tuple] = None,
    score_scale: float = 0.15,
    rot_key: str = "rot6d",
    trans_key: str = "trans",
) -> dict:
    rot6d = output[rot_key]
    trans = output[trans_key]
    shapes = output["shapes"]
    if rot6d.ndim == 4:
        rot6d, trans, shapes = rot6d[0], trans[0], shapes[0]
    if length is not None:
        rot6d, trans = rot6d[:length], trans[:length]
        gt = gt_joints[:length]
    else:
        gt = gt_joints[: rot6d.shape[0]]

    pred = _global_joints_from_params(body_model, rot6d, shapes, trans)
    if joint_range is not None:
        lo, hi = joint_range
        pred, gt = pred[:, lo:hi], gt[:, lo:hi]
    mpjpe = float(np.linalg.norm(pred - gt, axis=-1).mean())
    return {"mpjpe_m": mpjpe, "reward": float(np.exp(-mpjpe / score_scale))}

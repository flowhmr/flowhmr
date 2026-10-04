"""Rotation helpers and the world -> camera root transform.

The axis-angle / quaternion / matrix conversions are adapted from PyTorch3D
(https://github.com/facebookresearch/pytorch3d, BSD-3-Clause license).
"""

import torch
import torch.nn.functional as F


def _standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """Unit quaternion with a non-negative real part."""
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """sqrt(max(0, x)) with a zero subgradient where x is 0."""
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) rotation matrices -> (..., 4) quaternions, real part first."""
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # floor at 0.1: if q_abs is small, that candidate is never picked anyway
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # pick the best-conditioned candidate (largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return _standardize_quaternion(out)


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """(..., 4) quaternions, real part first -> (..., 3) axis-angle."""
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for small x, sin(x/2)/x ~= 1/2 - x^2/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) rotation matrices -> (..., 3) axis-angle."""
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """(..., 4) quaternions, real part first -> (..., 3, 3) rotation matrices."""
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """(..., 3) axis-angle -> (..., 4) quaternions, real part first."""
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for small x, sin(x/2)/x ~= 1/2 - x^2/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """(..., 3) axis-angle -> (..., 3, 3) rotation matrices."""
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def get_c_rootparam(global_orient, transl, T_w2c, offset):
    """Express the SMPL root orientation/translation in camera coordinates.

    SMPL rotates around the root joint, not the origin, so the rest-pose
    root position `offset` is added before and removed after the transform.

    Args:
        global_orient: (F, 3) world-space root axis-angle
        transl: (F, 3) world-space root translation
        T_w2c: (F, 3|4, 4) or (3|4, 4) world-to-camera extrinsics
        offset: (3,) rest-pose root joint position

    Returns:
        R_c: (F, 3) camera-space root axis-angle
        t_c: (F, 3) camera-space root translation
    """
    assert global_orient.shape == transl.shape and len(global_orient.shape) == 2
    R_w = axis_angle_to_matrix(global_orient)  # (F, 3, 3)
    t_w = transl  # (F, 3)

    R_w2c = T_w2c[..., :3, :3]  # (*, 3, 3)
    t_w2c = T_w2c[..., :3, 3]  # (*, 3)
    if len(R_w2c.shape) == 2:
        R_w2c = R_w2c[None].expand(R_w.size(0), -1, -1)  # (F, 3, 3)
        t_w2c = t_w2c[None].expand(t_w.size(0), -1)

    R_c = matrix_to_axis_angle(R_w2c @ R_w)  # (F, 3)
    t_c = torch.einsum("fij,fj->fi", R_w2c, t_w + offset) + t_w2c - offset  # (F, 3)
    return R_c, t_c

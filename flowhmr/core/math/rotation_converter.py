from typing import Optional

import numpy as np
import torch
from torch import Tensor

from .geometry import angle_axis_to_rotation_matrix

# TODO: the quaternion here is ALL REAL FIRST, fix the conflict later
"""
Note: rotation_converter.py builds the matrix as torch.cat((x, y, z), -1), i.e. columns
[x, y, z] with y = _cross_product(z, x); this 6D layout differs from the one in
geometry.py (rot6d_to_rotation_matrix).
"""


def get_local_transl_vel(transl: Tensor, global_orient: Tensor) -> Tensor:
    assert len(transl.shape) == len(global_orient.shape)
    global_orient_R = angle_axis_to_rotation_matrix(global_orient)
    transl_vel = transl[..., 1:, :] - transl[..., :-1, :]
    transl_vel = torch.cat([transl_vel, transl_vel[..., [-1], :]], dim=-2)

    # v_local = R^T @ v_global
    local_transl_vel = torch.einsum("...lij,...li->...lj", global_orient_R, transl_vel)
    return local_transl_vel


def rollout_local_transl_vel(
    local_transl_vel: Tensor, global_orient: Tensor, transl_0: Optional[Tensor] = None
) -> Tensor:
    global_orient_R = angle_axis_to_rotation_matrix(global_orient)
    transl_vel = torch.einsum("...lij,...lj->...li", global_orient_R, local_transl_vel)

    # set start point
    if transl_0 is None:
        transl_0 = transl_vel[..., :1, :].clone().detach().zero_()
    transl_ = torch.cat([transl_0, transl_vel[..., :-1, :]], dim=-2)

    # rollout from start point
    transl = torch.cumsum(transl_, dim=-2)
    return transl


def _rot_mat2trans_mat(rot_mat: Tensor) -> Tensor:
    trans_mat = torch.eye(4, device=rot_mat.device, dtype=rot_mat.dtype)
    trans_mat = torch.tile(trans_mat, (*rot_mat.shape[:-2], 1, 1))
    trans_mat[..., :3, :3] = rot_mat
    return trans_mat


def _trans2trans_mat(trans: Tensor) -> Tensor:
    assert trans.shape[-1] == 3
    trans_mat = torch.eye(4, device=trans.device, dtype=trans.dtype)
    trans_mat = torch.tile(trans_mat, (*trans.shape[:-1], 1, 1))
    trans_mat[..., :3, 3] = trans
    return trans_mat


def _scaling2trans_mat(scaling: Tensor) -> Tensor:
    batch_shape = scaling.shape[:-1]
    trans_mat = torch.eye(4, device=scaling.device, dtype=scaling.dtype)
    trans_mat = trans_mat.expand(*batch_shape, 4, 4).clone()
    trans_mat[..., :3, :3] = torch.diag_embed(scaling)
    return trans_mat


def to_transform_mat(
    R: Optional[Tensor] = None,
    t: Optional[Tensor] = None,
    s: Optional[Tensor] = None,
) -> Tensor:
    device = None
    dtype = None
    for tensor in [R, t, s]:
        if tensor is not None:
            device = tensor.device
            dtype = tensor.dtype
            break

    if device is None:
        device = torch.device("cpu")
        dtype = torch.float32

    if R is not None:
        R = _rot_mat2trans_mat(R)
    else:
        R = torch.eye(4, device=device, dtype=dtype)
    if t is not None:
        t = _trans2trans_mat(t)
    else:
        t = torch.eye(4, device=device, dtype=dtype)
    if s is not None:
        s = _scaling2trans_mat(s)
    else:
        s = torch.eye(4, device=device, dtype=dtype)

    transform_mat = t @ R @ s
    return transform_mat


def _sqrt_positive_part(x: Tensor) -> Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def _axis_angle_rotation(axis: str, angle: Tensor) -> Tensor:

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def _angle_from_tan(axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool) -> Tensor:

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")


def _standardize_quaternion(quaternions: Tensor) -> Tensor:
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_to_axis_angle(quaternions: Tensor) -> Tensor:
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., 0:1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles] * angles[small_angles]) / 48
    return quaternions[..., 1:] / sin_half_angles_over_angles


def axis_angle_to_quaternion(axis_angle: Tensor) -> Tensor:
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles] * angles[small_angles]) / 48
    quaternions = torch.cat([torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1)
    return quaternions


def quaternion_to_matrix(quaternions: Tensor) -> Tensor:
    w, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * w),
            two_s * (i * k + j * w),
            two_s * (i * j + k * w),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * w),
            two_s * (i * k - j * w),
            two_s * (j * k + i * w),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix: Tensor) -> Tensor:
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

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

    # we produce the desired quaternion multiplied by each of w, i, j, k
    quat_by_wijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = Tensor([0.1]).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_wijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(
        batch_dim + (4,)
    )
    return _standardize_quaternion(out)


def quaternion_normalize(quaternions: Tensor) -> Tensor:
    return quaternions / torch.norm(quaternions, dim=-1, keepdim=True)


def quaternion_between_vectors(v1: Tensor, v2: Tensor) -> Tensor:
    """compute the quaternion rotation from v1 to v2"""
    # ensure the vectors are unit vectors
    v1 = v1 / torch.norm(v1, dim=-1, keepdim=True)
    v2 = v2 / torch.norm(v2, dim=-1, keepdim=True)
    # compute the cross product and dot product
    cross_product = torch.cross(v1, v2, dim=-1)
    dot_product = torch.sum(v1 * v2, dim=-1, keepdim=True)
    # handle the case of parallel vectors
    parallel_mask = torch.abs(dot_product.squeeze(-1)) > 0.9999
    # for parallel vectors, return the unit quaternion
    quaternion = torch.zeros(*v1.shape[:-1], 4, device=v1.device, dtype=v1.dtype)
    quaternion[..., 0] = 1.0 # set the w component to 1
    # for non-parallel vectors, compute the quaternion
    non_parallel_mask = ~parallel_mask
    if non_parallel_mask.any():
        w = 1.0 + dot_product[non_parallel_mask].squeeze(-1)
        quaternion[non_parallel_mask, 1:] = cross_product[non_parallel_mask]
        quaternion[non_parallel_mask, 0] = w
        # standardize the quaternion
        quaternion[non_parallel_mask] = quaternion[non_parallel_mask] / torch.norm(
            quaternion[non_parallel_mask], dim=-1, keepdim=True
        )
    return quaternion


def quaternion_rotate_vector(q: Tensor, v: Tensor) -> Tensor:
    """quaternion rotate vector (w, x, y, z)"""
    assert q.shape[-1] == 4, f"Quaternion should have 4 components, got {q.shape[-1]}"
    assert v.shape[-1] == 3, f"Vector should have 3 components, got {v.shape[-1]}"
    assert q.shape[:-1] == v.shape[:-1], f"Batch dimensions should match: {q.shape[:-1]} vs {v.shape[:-1]}"

    original_shape = list(v.shape)

    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    qw = q[:, 0:1]

    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)

    rotated_v = v + 2 * (qw * uv + uuv)

    return rotated_v.view(original_shape)


def quaternion_multiply(q1: Tensor, q2: Tensor) -> Tensor:
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def quaternion_inverse(q: Tensor) -> Tensor:
    # for unit quaternions, the inverse is equal to the conjugate
    q_conj = q.clone()
    q_conj[..., 1:] = -q_conj[..., 1:] # the imaginary part is negative
    # if the quaternion is not a unit quaternion, divide by the square of the norm
    norm_squared = torch.sum(q * q, dim=-1, keepdim=True)
    return q_conj / norm_squared


def quaternion_fix_continuity(q: Tensor) -> Tensor:
    """
    force quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.
    """
    assert q.ndim in (
        2,
        3,
    ), f"Expected 3D tensor (L, J, 4), or 2D tensor (L, 4), but got shape {q.shape}"
    assert q.shape[-1] == 4, f"Last dimension should be 4 for quaternions, got {q.shape[-1]}"
    if q.shape[0] <= 1:
        return q.clone() # single frame or empty sequence, no need to process

    result = q.clone()
    # compute the dot product between consecutive frames (L-1, J) or (L-1)
    dot_products = torch.sum(q[1:] * q[:-1], dim=-1)
    # find the negative dot product (indicates need to flip sign)
    flip_mask = dot_products < 0
    # accumulate the flip mask, ensure consistency
    # if a frame needs to be flipped, all subsequent frames need to be flipped the same number of times
    flip_mask = (torch.cumsum(flip_mask.int(), dim=0) % 2).bool()
    # flip the sign of the frames that need to be flipped
    result[1:][flip_mask] *= -1
    return result


def degrees_to_radians(angles: Tensor) -> Tensor:
    return angles * (torch.pi / 180.0)


def radians_to_degrees(angles: Tensor) -> Tensor:
    return angles * (180.0 / torch.pi)


def euler_angles_to_matrix(euler_angles: Tensor, convention: str) -> Tensor:
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [_axis_angle_rotation(c, e) for c, e in zip(convention, torch.unbind(euler_angles, -1))]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def matrix_to_euler_angles(matrix: Tensor, convention: str) -> Tensor:
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0))
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(convention[0], convention[1], matrix[..., i2], False, tait_bryan),
        central_angle,
        _angle_from_tan(convention[2], convention[1], matrix[..., i0, :], True, tait_bryan),
    )
    return torch.stack(o, -1)


def ortho6d_to_matrix(ortho6d: Tensor) -> Tensor:
    if ortho6d.shape[-1] != 6:
        raise ValueError(f"Expected last dimension of ortho6d to be 6, but got {ortho6d.shape[-1]}")

    original_shape = ortho6d.shape[:-1]
    flattened_ortho6d = ortho6d.reshape(-1, 6)

    x_raw = flattened_ortho6d[..., 0:3]
    y_raw = flattened_ortho6d[..., 3:6]

    x = _normalize_vector(x_raw)
    z = _normalize_vector(_cross_product(x, y_raw))
    y = _cross_product(z, x)

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), -1)
    matrix = matrix.reshape(*original_shape, 3, 3)
    return matrix


def _normalize_vector(v: Tensor, return_mag: bool = False) -> Tensor:
    norm = torch.norm(v, p=2, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=1e-8)
    return v / norm


def _cross_product(u: Tensor, v: Tensor) -> Tensor:
    u1, u2, u3 = u[..., 0], u[..., 1], u[..., 2]
    v1, v2, v3 = v[..., 0], v[..., 1], v[..., 2]

    result = torch.stack([u2 * v3 - u3 * v2, u3 * v1 - u1 * v3, u1 * v2 - u2 * v1], dim=-1)
    return result


def matrix_to_ortho6d(matrix: Tensor) -> Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(
            f"Expected last two dimensions of the rotation matrix to be (3, 3), but got {matrix.shape[-2:]}"
        )
    x_raw = matrix[..., 0]
    y_raw = matrix[..., 1]

    eps = 1e-6
    eye_matrix = torch.eye(3, device=matrix.device, dtype=matrix.dtype)
    if matrix.dim() > 2:
        eye_matrix = eye_matrix.expand_as(matrix)

    is_orthogonal = torch.allclose(
        torch.matmul(matrix.transpose(-2, -1), matrix),
        eye_matrix,
        atol=eps,
    )

    if not is_orthogonal:
        x = _normalize_vector(x_raw)
        y = _normalize_vector(y_raw)
    else:
        x = x_raw
        y = y_raw
    ortho6d = torch.cat((x, y), dim=-1)
    return ortho6d


def slice_seq_with_padding(whole_seq: np.ndarray, middle_idx: int, length: int) -> np.ndarray:
    whole_seq_padded = whole_seq.copy()
    if middle_idx - length // 2 < 0:
        # need padding
        l_pad_len = length // 2 - middle_idx
        whole_seq_padded = np.concatenate([np.stack([whole_seq_padded[0]] * l_pad_len), whole_seq_padded], axis=0)
    else:
        l_pad_len = 0
    if middle_idx + length - length // 2 > len(whole_seq):
        r_pad_len = middle_idx + length - length // 2 - len(whole_seq)
        whole_seq_padded = np.concatenate([whole_seq_padded, np.stack([whole_seq_padded[-1]] * r_pad_len)], axis=0)
    else:
        r_pad_len = 0
    assert len(whole_seq_padded) == len(whole_seq) + l_pad_len + r_pad_len
    middle_idx_padded = middle_idx + l_pad_len
    assert middle_idx_padded - length // 2 >= 0
    assert middle_idx_padded + length - length // 2 <= len(whole_seq_padded)
    return whole_seq_padded[middle_idx_padded - length // 2 : middle_idx_padded - length // 2 + length]


def gaussian_kernel1d(sigma: float, order: int, radius: int) -> np.ndarray:
    """
    Computes a 1D Gaussian convolution kernel. (from scipy)
    """
    if order < 0:
        raise ValueError("order must be non-negative")
    exponent_range = np.arange(order + 1)
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius + 1)
    phi_x = np.exp(-0.5 / sigma2 * x**2)
    phi_x = phi_x / phi_x.sum()

    if order == 0:
        return phi_x
    else:
        # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
        # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
        # p'(x) = -1 / sigma ** 2
        # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
        # coefficients of q(x)
        q = np.zeros(order + 1)
        q[0] = 1
        D = np.diag(exponent_range[1:], 1)
        P = np.diag(np.ones(order) / -sigma2, -1)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        return q * phi_x


def wavg_quaternion_markley(Q: np.ndarray, weights: np.ndarray) -> np.ndarray:

    # Form the symmetric accumulator matrix
    A = np.zeros((4, 4))
    M = Q.shape[0]
    wSum = 0

    for i in range(M):
        q = Q[i, :]
        w_i = weights[i]
        if q[0] < 0:
            # handle the antipodal configuration
            q = -q
        A += w_i * (np.outer(q, q)) # rank 1 update
        wSum += w_i

    # scale
    A /= wSum

    # Get the eigenvector corresponding to largest eigen value
    return np.linalg.eigh(A)[1][:, -1]

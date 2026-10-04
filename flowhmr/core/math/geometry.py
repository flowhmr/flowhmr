import torch
import torch.nn.functional as F


def get_T_w2c_from_wcparams(global_orient_w, transl_w, global_orient_c, transl_c, offset):
    assert global_orient_w.shape == transl_w.shape and len(global_orient_w.shape) == 2
    assert global_orient_c.shape == transl_c.shape and len(global_orient_c.shape) == 2

    R_w = angle_axis_to_rotation_matrix(global_orient_w)
    t_w = transl_w
    R_c = angle_axis_to_rotation_matrix(global_orient_c)
    t_c = transl_c

    R_w2c = R_c @ R_w.transpose(-1, -2)
    t_w2c = t_c + offset - torch.einsum("fij,fj->fi", R_w2c, t_w + offset)  # (F, 3)
    T_w2c = torch.eye(4, device=global_orient_w.device).repeat(R_w.size(0), 1, 1)
    T_w2c[..., :3, :3] = R_w2c
    T_w2c[..., :3, 3] = t_w2c
    return T_w2c


def get_R_c2gv(R_w2c, axis_gravity_in_w=[0, 0, -1]):
    if isinstance(axis_gravity_in_w, list):
        axis_gravity_in_w = torch.tensor(axis_gravity_in_w).float() # gravity direction in world coord
    axis_z_in_c = torch.tensor([0, 0, 1]).float()

    # get gv-coord axes in in c-coord
    axis_y_of_gv = R_w2c @ axis_gravity_in_w
    axis_x_of_gv = axis_y_of_gv.cross(axis_z_in_c.expand_as(axis_y_of_gv), dim=-1)
    # normalize
    axis_x_of_gv_norm = axis_x_of_gv.norm(dim=-1, keepdim=True)
    axis_x_of_gv = axis_x_of_gv / (axis_x_of_gv_norm + 1e-5)
    axis_x_of_gv[axis_x_of_gv_norm.squeeze(-1) < 1e-5] = torch.tensor([1.0, 0.0, 0.0]) # use cam x-axis as axis_x_of_gv
    axis_z_of_gv = axis_x_of_gv.cross(axis_y_of_gv, dim=-1)

    R_gv2c = torch.stack([axis_x_of_gv, axis_y_of_gv, axis_z_of_gv], dim=-1)
    R_c2gv = R_gv2c.transpose(-1, -2)
    return R_c2gv


def get_c_rootparam(global_orient, transl, T_w2c, offset):
    assert global_orient.shape == transl.shape and len(global_orient.shape) == 2
    R_w = angle_axis_to_rotation_matrix(global_orient)
    t_w = transl

    R_w2c = T_w2c[..., :3, :3]
    t_w2c = T_w2c[..., :3, 3]
    if len(R_w2c.shape) == 2:
        R_w2c = R_w2c[None].expand(R_w.size(0), -1, -1)
        t_w2c = t_w2c[None].expand(t_w.size(0), -1)

    R_c = rotation_matrix_to_angle_axis(R_w2c @ R_w)
    t_c = torch.einsum("fij,fj->fi", R_w2c, t_w + offset) + t_w2c - offset  # (F, 3)
    return R_c, t_c


def compute_cam_angvel(R_w2c, padding_last=True):
    # R @ R0 = R1, so R = R1 @ R0^T
    cam_angvel = rotation_matrix_to_rot6d(R_w2c[1:] @ R_w2c[:-1].transpose(-1, -2))
    # cam_angvel = (cam_angvel - torch.tensor([[1, 0, 0, 0, 1, 0]])) * FPS
    assert padding_last
    cam_angvel = torch.cat([cam_angvel, cam_angvel[-1:]], dim=0)
    return cam_angvel.float()


def rot6d_to_rotation_matrix(rot6d):
    # x = rot6d.view(-1, 3, 2)
    x = rot6d.view(*rot6d.shape[:-1], 3, 2)
    a1 = x[..., 0]
    a2 = x[..., 1]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - torch.einsum("...i,...i->...", b1, a2).unsqueeze(-1) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def rotation_matrix_to_rot6d(rotation_matrix):
    v1 = rotation_matrix[..., 0:1]
    v2 = rotation_matrix[..., 1:2]
    rot6d = torch.cat([v1, v2], dim=-1).reshape(*v1.shape[:-2], 6)
    return rot6d


def quaternion_to_rotation_matrix(quaternion):

    norm_quaternion = quaternion
    norm_quaternion = norm_quaternion / norm_quaternion.norm(p=2, dim=-1, keepdim=True)
    w, x, y, z = norm_quaternion[..., 0], norm_quaternion[..., 1], norm_quaternion[..., 2], norm_quaternion[..., 3]

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotation_matrix = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=-1,
    )
    rotation_matrix = rotation_matrix.view(*quaternion.shape[:-1], 3, 3)
    return rotation_matrix


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}".format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0, torch.atan2(-sin_theta, -cos_theta), torch.atan2(sin_theta, cos_theta)
    )

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(rotation_matrix)))

    # Save original shape and reshape if needed
    origin_shape = rotation_matrix.shape[:-2]
    flat_rot = rotation_matrix.reshape(-1, *rotation_matrix.shape[-2:])

    if not flat_rot.shape[-2:] == (3, 4):
        hom = (
            torch.tensor([0, 0, 1], dtype=rotation_matrix.dtype, device=rotation_matrix.device)
            .reshape(1, 3, 1)
            .expand(flat_rot.shape[0], -1, -1)
        )
        flat_rot = torch.cat([flat_rot, hom], dim=-1)

    rotation_matrix = flat_rot

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack(
        [rmat_t[:, 1, 2] - rmat_t[:, 2, 1], t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0], rmat_t[:, 2, 0] + rmat_t[:, 0, 2]],
        -1,
    )
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack(
        [rmat_t[:, 2, 0] - rmat_t[:, 0, 2], rmat_t[:, 0, 1] + rmat_t[:, 1, 0], t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]],
        -1,
    )
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack(
        [rmat_t[:, 0, 1] - rmat_t[:, 1, 0], rmat_t[:, 2, 0] + rmat_t[:, 0, 2], rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2],
        -1,
    )
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack(
        [t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1], rmat_t[:, 2, 0] - rmat_t[:, 0, 2], rmat_t[:, 0, 1] - rmat_t[:, 1, 0]],
        -1,
    )
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 + t2_rep * mask_c2 + t3_rep * mask_c3)
    q *= 0.5

    # Reshape back to original shape if needed
    q = q.reshape(*origin_shape, 4)
    return q


def rotation_matrix_to_angle_axis(rotation_matrix):
    origin_shape = rotation_matrix.shape[:-2]
    flat_rot = rotation_matrix.reshape(-1, *rotation_matrix.shape[-2:])
    if flat_rot.shape[1:] == (3, 3):
        rot_mat = flat_rot
        hom = (
            torch.tensor([0, 0, 1], dtype=rotation_matrix.dtype, device=rotation_matrix.device)
            .reshape(1, 3, 1)
            .expand(rot_mat.shape[0], -1, -1)
        )
        flat_rot = torch.cat([rot_mat, hom], dim=-1)

    quaternion = rotation_matrix_to_quaternion(flat_rot)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    aa = aa.reshape(*origin_shape, 3)
    return aa


def quat_to_rotmat(quat):
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=1,
    ).view(B, 3, 3)
    return rotMat


def angle_axis_to_rotation_matrix(theta):
    origin_shape = theta.shape[:-1]
    flat_theta = theta.reshape(-1, 3)
    l1norm = torch.norm(flat_theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(flat_theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)
    rot_mat = quat_to_rotmat(quat)
    return rot_mat.reshape(*origin_shape, 3, 3)


def rotation_matrix_to_euler_angles(rotation_matrix):
    """
    Convert 3x3 rotation matrix to Euler angles.
    """
    is_torch = False
    if isinstance(rotation_matrix, torch.Tensor):
        is_torch = True
        device = rotation_matrix.device
        rotation_matrix = rotation_matrix.cpu().numpy()
    from scipy.spatial.transform import Rotation

    rot_flat = rotation_matrix.reshape(-1, 3, 3)
    euler_angles = Rotation.from_matrix(rot_flat).as_euler("xyz", degrees=True)
    if is_torch:
        return torch.from_numpy(euler_angles).to(device)
    return euler_angles


def euler_angles_to_rotation_matrix(euler_angles, degrees=True):
    from scipy.spatial.transform import Rotation

    orig_shape = euler_angles.shape[:-1]
    euler_flat = euler_angles.reshape(-1, 3)
    rot_flat = Rotation.from_euler("xyz", euler_flat, degrees=degrees).as_matrix()
    return rot_flat.reshape(*orig_shape, 3, 3)


def get_local_transl_vel(transl, global_orient_R, fps=30):
    transl_vel = transl[..., 1:, :] - transl[..., :-1, :]
    transl_vel = torch.cat([torch.zeros_like(transl_vel[:1]), transl_vel], dim=-2)
    transl_vel = transl_vel * fps

    # v_local = R^T @ v_global
    local_transl_vel = torch.einsum("...lij,...li->...lj", global_orient_R, transl_vel)
    return local_transl_vel


def compute_transl_full_cam(pred_cam, bbx_xys, K_fullimg):
    s, tx, ty = pred_cam[..., 0], pred_cam[..., 1], pred_cam[..., 2]
    focal_length = K_fullimg[..., 0, 0]

    icx = K_fullimg[..., 0, 2]
    icy = K_fullimg[..., 1, 2]
    sb = s * bbx_xys[..., 2]
    cx = 2 * (bbx_xys[..., 0] - icx) / (sb + 1e-9)
    cy = 2 * (bbx_xys[..., 1] - icy) / (sb + 1e-9)
    tz = 2 * focal_length / (sb + 1e-9)

    cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)
    return cam_t

from typing import Dict

import torch

from flowhmr.core.math.geometry import rot6d_to_rotation_matrix, rotation_matrix_to_rot6d
from flowhmr.core.motion.motion_process import get_foot_detect
from flowhmr.core.motion.smooth_root import get_smooth_root_pos


# Module-level constants

# SMPLH joint parent indices (52 joints, root has parent -1)
# fmt: off
_SMPLH_PARENTS = torch.tensor([
    -1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,
     9,  9,  9, 12, 13, 14, 16, 17, 18, 19, 20, 22,
    23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34,
    35, 21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46,
    47, 21, 49, 50
], dtype=torch.long)
# fmt: on


# Internal helpers

def _run_fk(body_model, root_rot6d: torch.Tensor, body_rot6d: torch.Tensor,
            shapes: torch.Tensor, trans: torch.Tensor):
    B, T = root_rot6d.shape[:2]
    # concat root + body joints → (B*T, 52, 6)
    rot6d_BT52 = torch.cat(
        [root_rot6d.reshape(B * T, 1, 6), body_rot6d.reshape(B * T, 51, 6)],
        dim=1,
    )
    shapes_BT = shapes.reshape(B * T, -1)
    trans_BT = trans.reshape(B * T, 3)

    params = {"rot6d": rot6d_BT52, "shapes": shapes_BT, "trans": trans_BT}
    out = body_model(params)

    keypoints3d = out["keypoints3d"].reshape(B, T, 52, 3)      # (B, T, 52, 3)
    transforms = out["transforms"].reshape(B, T, 52, 4, 4)     # (B, T, 52, 4, 4)
    return keypoints3d, transforms


def _compute_heading_angle(global_joints_positions: torch.Tensor) -> torch.Tensor:
    l_hip = global_joints_positions[:, :, 1, :]
    r_hip = global_joints_positions[:, :, 2, :]
    # heading: θ = atan2(r_hip_z - l_hip_z, -(r_hip_x - l_hip_x))
    dz = r_hip[..., 2] - l_hip[..., 2]
    dx = -(r_hip[..., 0] - l_hip[..., 0])
    theta = torch.atan2(dz, dx)
    return theta


def _global_rots_to_local_rots(global_rot_mats: torch.Tensor,
                                parents: torch.Tensor) -> torch.Tensor:
    J = global_rot_mats.shape[2]
    local_rot_mats = global_rot_mats.clone()
    for j in range(1, J):
        p = parents[j].item()
        # R_local[j] = R_global[parent]^T @ R_global[j]
        local_rot_mats[:, :, j] = (
            global_rot_mats[:, :, p].transpose(-1, -2) @ global_rot_mats[:, :, j]
        )
    return local_rot_mats


def _decode_motion_base_style(motion_dict: Dict, body_model=None, fps: float = 30.0) -> Dict:
    global_rot_data = motion_dict["global_rot_data"]          # (B, T, 52, 6)
    smooth_root_pos = motion_dict["smooth_root_pos"]          # (B, T, 3)
    local_joints_positions = motion_dict["local_joints_positions"]  # (B, T, N, 3)
    shapes_raw = motion_dict["shapes"]                        # (B, T, 16)

    unbatched = global_rot_data.ndim == 3
    if unbatched:
        global_rot_data = global_rot_data.unsqueeze(0)
        smooth_root_pos = smooth_root_pos.unsqueeze(0)
        local_joints_positions = local_joints_positions.unsqueeze(0)
        shapes_raw = shapes_raw.unsqueeze(0)

    B, T, J = global_rot_data.shape[:3]

    global_rot_mats = rot6d_to_rotation_matrix(
        global_rot_data.reshape(B * T * J, 6)
    ).reshape(B, T, J, 3, 3)

    local_rot_mats = _global_rots_to_local_rots(
        global_rot_mats, _SMPLH_PARENTS.to(global_rot_data.device)
    )

    root_rot_mats = local_rot_mats[:, :, 0]
    body_rot_mats = local_rot_mats[:, :, 1:]

    root_rot6d = rotation_matrix_to_rot6d(root_rot_mats)
    body_rot6d = rotation_matrix_to_rot6d(body_rot_mats)

    # 4. Recover pelvis world position from local_joints_positions[joint 0]
    # Encoding subtracted smooth_root from XZ, so: pelvis_world = local_joints_positions[:,:,0] + smooth_root (XZ only)
    pelvis_local = local_joints_positions[:, :, 0, :]
    pelvis_world = pelvis_local.clone()
    pelvis_world[:, :, 0] += smooth_root_pos[:, :, 0] # restore world X
    pelvis_world[:, :, 2] += smooth_root_pos[:, :, 2] # restore world Z

    # 5. Compute trans
    if body_model is not None:
        shapes_mean = shapes_raw.mean(dim=1)
        j_shaped = body_model.compute_j_shaped(shapes_mean)
        pelvis_rest = j_shaped[:, 0, :].unsqueeze(1)
        trans = pelvis_world - pelvis_rest
    else:
        trans = pelvis_world

    shapes = shapes_raw

    result = {
        "root_rot6d":       root_rot6d,        # (B, T, 6)
        "body_rot6d":       body_rot6d,         # (B, T, 51, 6)
        "trans":            trans,              # (B, T, 3)
        "shapes":           shapes,             # (B, T, 16)
        "pelvis_world":     pelvis_world,       # (B, T, 3)
    }

    if unbatched:
        result = {k: v.squeeze(0) for k, v in result.items()}

    return result


def _encode_motion_base_style(target_dict: Dict, body_model, fps: float = 30.0,
                             n_joints_pos: int = 52) -> Dict:
    root_rot6d = target_dict["root_rot6d"]
    body_rot6d = target_dict["body_rot6d"]
    trans = target_dict["trans"]
    shapes = target_dict["shapes"]

    # ---- auto-batch (B=1) if unbatched ----
    unbatched = root_rot6d.ndim == 2
    if unbatched:
        root_rot6d = root_rot6d.unsqueeze(0)
        body_rot6d = body_rot6d.unsqueeze(0)
        trans = trans.unsqueeze(0)
        shapes = shapes.unsqueeze(0)

    B, T = root_rot6d.shape[:2]

    # 1. FK → global joint positions + transforms -------------------------
    keypoints3d, transforms = _run_fk(body_model, root_rot6d, body_rot6d, shapes, trans)

    global_rot_mats = transforms[:, :, :, :3, :3]

    # NOTE: SMPLH `trans` is NOT the pelvis world position.
    # The actual pelvis world position = keypoints3d[:,:,0,:] (FK joint 0 + trans).
    # Smoothing must be done on the real pelvis trajectory, not on trans.
    pelvis_pos = keypoints3d[:, :, 0, :]
    smooth_root = get_smooth_root_pos(pelvis_pos)

    theta = _compute_heading_angle(keypoints3d)
    global_root_heading = torch.stack(
        [torch.cos(theta), torch.sin(theta)], dim=-1
    )

    local_joints_positions = keypoints3d[:, :, :n_joints_pos, :].clone() # (B, T, n_joints_pos, 3)
    local_joints_positions[..., [0, 2]] -= smooth_root.unsqueeze(2)[..., [0, 2]]

    global_rot_data = rotation_matrix_to_rot6d(global_rot_mats)

    foot_contacts_list = []
    for b in range(B):
        kpts_b = keypoints3d[b]
        _, contact_b = get_foot_detect(
            kpts_b,
            joint_ids=(7, 10, 8, 11), # [L_ankle, L_toe, R_ankle, R_toe]
            vel_thr_per_frame=0.15 / fps,
            use_height=True,
            toe_height_thr_m=0.10,
            ankle_height_thr_m=0.10,
        )
        foot_contacts_list.append(contact_b.float().unsqueeze(0))
    foot_contacts = torch.cat(foot_contacts_list, dim=0)

    local_rot6d = torch.cat([root_rot6d[:, :, None, :], body_rot6d], dim=2)

    smooth_root_vel_tail = (smooth_root[:, 1:] - smooth_root[:, :-1]) * fps
    smooth_root_vel = torch.cat([smooth_root_vel_tail[:, :1], smooth_root_vel_tail], dim=1)

    result = {
        # core fields
        "smooth_root_pos":          smooth_root,              # (B, T, 3)
        "smooth_root_vel":          smooth_root_vel,          # (B, T, 3)
        "global_root_heading":      global_root_heading,      # (B, T, 2)
        "local_joints_positions":   local_joints_positions,   # (B, T, n_joints_pos, 3)
        "global_rot_data":          global_rot_data,          # (B, T, 52, 6)
        "local_rot_data":           local_rot6d,              # (B, T, 52, 6)
        "foot_contacts":            foot_contacts,            # (B, T, 4)
        # Auxiliary / debug fields
        "trans":                    trans,                    # (B, T, 3)
        "shapes":                   shapes,                   # (B, T, 16)
        "global_joints_positions":  keypoints3d,              # (B, T, 52, 3)
    }

    if unbatched:
        result = {k: v.squeeze(0) for k, v in result.items()}

    return result


def encode_motion_v0(target_dict: Dict, body_model, fps: float = 30.0) -> Dict:
    # Get the base result (all 52 joints, no velocities)
    result = _encode_motion_base_style(target_dict, body_model, fps=fps, n_joints_pos=52)

    # Compute velocities from global joint positions (keypoints3d)
    # Handle unbatched case: _encode_motion_base_style squeezes output if unbatched,
    # so we need to track the original batching state to compute velocities correctly.
    unbatched = target_dict["root_rot6d"].ndim == 2

    keypoints3d = result["global_joints_positions"]   # (B, T, 52, 3) or (T, 52, 3) if unbatched
    if unbatched:
        keypoints3d = keypoints3d.unsqueeze(0)

    vel_tail = (keypoints3d[:, 1:] - keypoints3d[:, :-1]) * fps
    velocities = torch.cat([vel_tail[:, :1], vel_tail], dim=1)

    if unbatched:
        velocities = velocities.squeeze(0)

    result["velocities"] = velocities
    return result


def _run_fk_rotmat(body_model, root_rotation: torch.Tensor, body_rotations: torch.Tensor,
                   shapes: torch.Tensor, trans: torch.Tensor):
    B, T = root_rotation.shape[:2]
    # concat root + body → (B*T, 52, 3, 3)
    rot_mats_BT52 = torch.cat(
        [root_rotation.reshape(B * T, 1, 3, 3), body_rotations.reshape(B * T, 51, 3, 3)],
        dim=1,
    )
    shapes_BT = shapes.reshape(B * T, -1)
    trans_BT = trans.reshape(B * T, 3)

    params = {"rot_mats": rot_mats_BT52, "shapes": shapes_BT, "trans": trans_BT}
    out = body_model(params)

    keypoints3d = out["keypoints3d"].reshape(B, T, 52, 3)
    transforms = out["transforms"].reshape(B, T, 52, 4, 4)
    return keypoints3d, transforms


def encode_motion_v0_rotmat(target_dict: Dict, body_model, fps: float = 30.0) -> Dict:
    root_rotation = target_dict["root_rotation"]
    body_rotations = target_dict["body_rotations"]
    trans = target_dict["trans"]
    shapes = target_dict["shapes"]

    unbatched = root_rotation.ndim == 3 # (T, 3, 3) vs (B, T, 3, 3)
    if unbatched:
        root_rotation = root_rotation.unsqueeze(0)
        body_rotations = body_rotations.unsqueeze(0)
        trans = trans.unsqueeze(0)
        shapes = shapes.unsqueeze(0)

    B, T = root_rotation.shape[:2]

    # 1. FK
    keypoints3d, transforms = _run_fk_rotmat(body_model, root_rotation, body_rotations, shapes, trans)

    # 2. Global rotations
    global_rot_mats = transforms[:, :, :, :3, :3]

    # 3. Smooth root
    pelvis_pos = keypoints3d[:, :, 0, :]
    smooth_root = get_smooth_root_pos(pelvis_pos)

    # 4. Heading
    theta = _compute_heading_angle(keypoints3d)
    global_root_heading = torch.stack(
        [torch.cos(theta), torch.sin(theta)], dim=-1
    )

    # 5. Local joint positions (all 52 joints)
    local_joints_positions = keypoints3d.clone()
    local_joints_positions[..., [0, 2]] -= smooth_root.unsqueeze(2)[..., [0, 2]]

    # 6. Global rotation 6D
    global_rot_data = rotation_matrix_to_rot6d(global_rot_mats)

    # 7. Foot contacts
    foot_contacts_list = []
    for b in range(B):
        kpts_b = keypoints3d[b]
        _, contact_b = get_foot_detect(
            kpts_b,
            joint_ids=(7, 10, 8, 11),
            vel_thr_per_frame=0.15 / fps,
            use_height=True,
            toe_height_thr_m=0.10,
            ankle_height_thr_m=0.10,
        )
        foot_contacts_list.append(contact_b.float().unsqueeze(0))
    foot_contacts = torch.cat(foot_contacts_list, dim=0)

    # 8. Velocities
    vel_tail = (keypoints3d[:, 1:] - keypoints3d[:, :-1]) * fps
    velocities = torch.cat([vel_tail[:, :1], vel_tail], dim=1)

    # 9. Local rotation 6D (from input rotation matrices)
    local_rot_mats = torch.cat(
        [root_rotation[:, :, None, :, :], body_rotations], dim=2
    )
    local_rot_data = rotation_matrix_to_rot6d(local_rot_mats)

    # 10. Smooth root velocity
    smooth_root_vel_tail = (smooth_root[:, 1:] - smooth_root[:, :-1]) * fps
    smooth_root_vel = torch.cat([smooth_root_vel_tail[:, :1], smooth_root_vel_tail], dim=1)

    result = {
        "smooth_root_pos":          smooth_root,
        "smooth_root_vel":          smooth_root_vel,
        "global_root_heading":      global_root_heading,
        "local_joints_positions":   local_joints_positions,
        "global_rot_data":          global_rot_data,
        "local_rot_data":           local_rot_data,
        "velocities":               velocities,
        "foot_contacts":            foot_contacts,
        "trans":                    trans,
        "shapes":                   shapes,
        "global_joints_positions":  keypoints3d,
    }

    if unbatched:
        result = {k: v.squeeze(0) for k, v in result.items()}

    return result


def decode_motion_v0(motion_dict: Dict, body_model=None, fps: float = 30.0) -> Dict:
    return _decode_motion_base_style(motion_dict, body_model=body_model, fps=fps)


def encode_motion_v1(target_dict: Dict, body_model, fps: float = 30.0) -> Dict:
    return _encode_motion_base_style(target_dict, body_model, fps=fps, n_joints_pos=52)


def decode_motion_v1(motion_dict: Dict, body_model=None, fps: float = 30.0) -> Dict:
    return _decode_motion_base_style(motion_dict, body_model=body_model, fps=fps)


def encode_motion_v2(target_dict: Dict, body_model, fps: float = 30.0) -> Dict:
    return _encode_motion_base_style(target_dict, body_model, fps=fps, n_joints_pos=22)


def decode_motion_v2(motion_dict: Dict, body_model=None, fps: float = 30.0) -> Dict:
    return _decode_motion_base_style(motion_dict, body_model=body_model, fps=fps)


def encode_motion_v3(target_dict: Dict, body_model, fps: float = 30.0) -> Dict:
    return _encode_motion_base_style(target_dict, body_model, fps=fps, n_joints_pos=1)


def decode_motion_v3(motion_dict: Dict, body_model=None, fps: float = 30.0) -> Dict:
    return _decode_motion_base_style(motion_dict, body_model=body_model, fps=fps)

from typing import List, Optional

import torch


def process_r_t(R_transform, root_rotation, transl, j_shaped):
    root_rotation_new = R_transform[None] @ root_rotation
    transl_new = (
        R_transform[None] @ (j_shaped[..., None] + torch.FloatTensor(transl[..., None]))
    ).reshape(-1, 3) - j_shaped
    return root_rotation_new, transl_new


def get_local_transl_vel(transl, global_orient_R, fps):
    transl_vel = transl[1:] - transl[:-1]
    transl_vel = torch.cat([transl_vel, transl_vel[-1:]], dim=0)
    transl_vel = transl_vel * fps
    local_transl_vel = torch.einsum("tij,ti->tj", global_orient_R, transl_vel)
    return local_transl_vel


def compute_wv_rotation(camera_R0: torch.Tensor) -> torch.Tensor:
    axis_z_in_c = torch.tensor([0, 0, 1], dtype=torch.float32)
    axis_z_in_w = camera_R0.t() @ axis_z_in_c
    axis_up_in_w = torch.tensor([0, 1, 0], dtype=torch.float32)

    axis_newx = torch.cross(axis_up_in_w, axis_z_in_w, dim=-1)
    axis_newx = axis_newx / axis_newx.norm(dim=-1, keepdim=True)

    axis_newz = torch.cross(axis_newx, axis_up_in_w, dim=-1)
    axis_newz = axis_newz / axis_newz.norm(dim=-1, keepdim=True)

    return torch.stack([axis_newx, axis_up_in_w, axis_newz], dim=-1).t()


def compute_camera_features(camera_RT_wv: torch.Tensor):
    camera_R0 = camera_RT_wv[0, :3, :3]
    R_to_first_frame = camera_RT_wv[:, :3, :3] @ camera_R0.t()[None]

    T_cam = torch.einsum(
        "tij,tj->ti",
        camera_RT_wv[:, :3, :3].transpose(1, 2),
        -camera_RT_wv[:, :3, -1],
    )
    center_velocity = T_cam[1:] - T_cam[:-1]
    center_velocity = torch.cat([center_velocity, center_velocity[-1:]], dim=0)

    return R_to_first_frame, center_velocity


def compute_bbox_info(
    bbox_center: torch.Tensor, bbox_scale: torch.Tensor, K: torch.Tensor
) -> torch.Tensor:
    bbox_center_homo = torch.cat([bbox_center, torch.ones_like(bbox_center[:, :1])], dim=-1)
    bbox_center_norm = torch.inverse(K) @ bbox_center_homo.reshape(-1, 3, 1)
    bbox_center_norm = bbox_center_norm.squeeze(-1)[:, :2]
    bbox_scale_norm = bbox_scale * 2 / (K[:, 0, 0] + K[:, 1, 1]).unsqueeze(-1)
    return torch.cat([bbox_center_norm, bbox_scale_norm], dim=-1)


def padding_or_clip(
    data: dict,
    max_len: int,
    round_frames: int = 1,
    keys: Optional[List[str]] = None,
) -> dict:
    if keys is None:
        keys = list(data.keys())

    for key in keys:
        if key not in data:
            continue
        val = data[key]
        if not isinstance(val, torch.Tensor):
            continue

        length = val.shape[0]
        length = length // round_frames * round_frames

        if max_len is None:
            pass
        elif length > max_len:
            data[key] = val[:max_len]
        else:
            padding = torch.zeros(max_len - length, *val.shape[1:]) + val[-1:]
            data[key] = torch.cat([val[:length], padding], dim=0)

    return data

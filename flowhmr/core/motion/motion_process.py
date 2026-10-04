import math
from typing import List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import torch
import torch.nn.functional as F
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from torch import Tensor

from ..math.geometry import rot6d_to_rotation_matrix
from ..math.rotation_converter import (
    axis_angle_to_quaternion,
    gaussian_kernel1d,
    matrix_to_quaternion,
    quaternion_fix_continuity,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_normalize,
    quaternion_rotate_vector,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
)
from ..visualization.visualize2d import COLORS, HML3D_KINEMATIC_CHAINS


def _angle_diff(a: Tensor, b: Tensor) -> Tensor:
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))


def _unwrap_angle_seq(yaw: Tensor) -> Tensor:
    if yaw.numel() <= 1:
        return yaw
    dy = _angle_diff(yaw[1:], yaw[:-1])
    dy = (dy + math.pi) % (2 * math.pi) - math.pi
    out = yaw.clone()
    out[1:] = out[0] + torch.cumsum(dy, dim=0)
    return out


def _get_heading_single_frame(root_rotations_mat: Tensor) -> Tensor:
    if root_rotations_mat.dim() == 2:
        root_rot = root_rotations_mat.unsqueeze(0)
    elif root_rotations_mat.dim() == 3:
        root_rot = root_rotations_mat
    else:
        raise ValueError(f"Expect (3,3) or (B,3,3), got {root_rotations_mat.shape}")

    device = root_rot.device
    local_x_world = root_rot[..., 0]
    local_z_world = root_rot[..., 2]

    fwd_main = F.normalize(local_z_world[..., [0, 2]], dim=-1, eps=1e-8)
    yaw_main = torch.atan2(fwd_main[..., 0], fwd_main[..., 1])

    fwd_fb = F.normalize(torch.stack([-local_x_world[..., 2], local_x_world[..., 0]], dim=-1), dim=-1, eps=1e-8)
    yaw_fb = torch.atan2(fwd_fb[..., 0], fwd_fb[..., 1])

    world_y = torch.tensor([0.0, 1.0, 0.0], device=device).view(1, 3)
    dot_abs = torch.abs((local_z_world * world_y).sum(dim=-1))
    use_fb = dot_abs > 0.50

    yaw = torch.where(use_fb, yaw_fb, yaw_main)
    return yaw


def _get_heading(
    root_rotations_mat: Tensor,
    rel_hysteresis: float = 0.05,
    max_deg_per_frame: float = 60.0,
    min_reliability: float = 1e-3,
) -> Tensor:
    device = root_rotations_mat.device

    is_batch = root_rotations_mat.dim() == 4
    if not is_batch:
        root_rot = root_rotations_mat.unsqueeze(0)
    else:
        root_rot = root_rotations_mat
    B, L = root_rot.shape[0], root_rot.shape[1]
    local_x_world = root_rot[..., 0]
    local_z_world = root_rot[..., 2]

    fwd_main = torch.nn.functional.normalize(local_z_world[..., [0, 2]], dim=-1, eps=1e-8)
    fwd_fb = torch.nn.functional.normalize(
        torch.stack([-local_x_world[..., 2], local_x_world[..., 0]], dim=-1),
        dim=-1,
        eps=1e-8,
    )
    yaw_main = torch.atan2(fwd_main[..., 0], fwd_main[..., 1])
    yaw_fb = torch.atan2(fwd_fb[..., 0], fwd_fb[..., 1])
    world_y_axis = torch.tensor([0.0, 1.0, 0.0], device=device).view(1, 1, 3).expand_as(local_z_world)
    dot_product_abs = torch.abs(torch.sum(local_z_world * world_y_axis, dim=-1))
    r_main = (1.0 - dot_product_abs).clamp(0.0, 1.0)
    r_fb = dot_product_abs

    out = torch.empty((B, L), device=device, dtype=yaw_main.dtype)
    prev_yaw = None
    prev_pick = 0 # 0:main, 1:fb
    max_delta = math.radians(max_deg_per_frame)
    for t in range(L):
        ym, yf = yaw_main[:, t], yaw_fb[:, t]
        rm, rf = r_main[:, t], r_fb[:, t]
        if t == 0:
            pick = rm < rf
            yaw = torch.where(pick, yf, ym)
        else:
            assert prev_yaw is not None
            keep_main = (~prev_pick) & (rf <= rm + rel_hysteresis)
            keep_fb = (prev_pick) & (rm <= rf + rel_hysteresis)
            d_m = torch.abs(_angle_diff(ym, prev_yaw))
            d_f = torch.abs(_angle_diff(yf, prev_yaw))
            choose_fb = d_f < d_m
            pick = torch.where(
                keep_main,
                torch.zeros_like(choose_fb),
                torch.where(keep_fb, torch.ones_like(choose_fb), choose_fb),
            )

            yaw_cand = torch.where(pick, yf, ym)
            r_cand = torch.where(pick, rf, rm)
            low_rel = r_cand < min_reliability

            delta = _angle_diff(yaw_cand, prev_yaw).clamp(-max_delta, max_delta)
            yaw = prev_yaw + delta
            yaw = torch.where(low_rel, prev_yaw, yaw)
            pick = torch.where(low_rel, prev_pick, pick)

        out[:, t] = yaw
        prev_yaw, prev_pick = yaw, pick

    out = _unwrap_yaw(out)
    if not is_batch:
        out = out.squeeze(0)
    return out


def get_root_vel(root_rotations_mat: Tensor, root_transl: Tensor) -> Tuple[Tensor, Tensor]:
    # root_rotations_mat shape of (L, 3, 3)
    # root_transl shape of (L, 3)


    yaw = _get_heading(root_rotations_mat)
    cos_yaw = torch.cos(yaw[:-1])
    sin_yaw = torch.sin(yaw[:-1])

    root_vel_world = root_transl[1:] - root_transl[:-1]
    root_vel_x_world, root_vel_z_world = root_vel_world[:, 0], root_vel_world[:, 2]
    root_vel_x_body = cos_yaw * root_vel_x_world - sin_yaw * root_vel_z_world
    root_vel_z_body = sin_yaw * root_vel_x_world + cos_yaw * root_vel_z_world
    root_xz_vel_body = torch.zeros(root_transl.size(0), 2, device=root_transl.device)
    root_xz_vel_body[:-1, 0] = root_vel_x_body
    root_xz_vel_body[:-1, 1] = root_vel_z_body

    yaw_delta = yaw[1:] - yaw[:-1]
    yaw_delta = (yaw_delta + torch.pi) % (2 * torch.pi) - torch.pi # wrap
    root_ry_vel_world = torch.zeros_like(yaw)
    root_ry_vel_world[1:] = yaw_delta
    root_ry_vel_world = root_ry_vel_world.unsqueeze(-1)

    return root_ry_vel_world, root_xz_vel_body


def get_rifke(kpts: Tensor, root_rotations_mat: Tensor, mode: str = "yaw") -> Tensor:
    # Root-Invariant Forward Kinematics Encoding (RIFKE)
    # kpts shape of ([B,] L, J, 3)
    # root_rotations_mat shape of ([B,] L, 3, 3)
    assert kpts.dim() == root_rotations_mat.dim()
    if kpts.dim() == 3:
        kpts = kpts.unsqueeze(0)
        root_rotations_mat = root_rotations_mat.unsqueeze(0)
        is_batch = False
    else:
        is_batch = True

    if mode == "yaw":
        kpts_local = kpts.clone()
        kpts_local[..., 0] -= kpts[..., 0:1, 0]
        kpts_local[..., 2] -= kpts[..., 0:1, 2]
        yaw = _get_heading(root_rotations_mat)
        cos_yaw = torch.cos(yaw).to(kpts.device).unsqueeze(-1)
        sin_yaw = torch.sin(yaw).to(kpts.device).unsqueeze(-1)
        x, z = kpts_local[..., 0].clone(), kpts_local[..., 2].clone()
        kpts_local[..., 0] = cos_yaw * x - sin_yaw * z
        kpts_local[..., 2] = sin_yaw * x + cos_yaw * z
    elif mode == "so3":
        root = kpts[..., 0:1, :].clone()
        centered = kpts - root
        R_inv = root_rotations_mat.transpose(-1, -2)
        kpts_local = torch.einsum("blij,blkj->blki", R_inv, centered)
    elif mode == "no_translation":
        root = kpts[..., 0:1, :].clone()
        kpts_local = kpts - root
    else:
        raise ValueError(f"Unknown mode: {mode}")
    if not is_batch:
        kpts_local = kpts_local.squeeze(0)
        root_rotations_mat = root_rotations_mat.squeeze(0)
    return kpts_local


def estimate_floor_height_global(
    kpts: Tensor,
    joint_ids: tuple = (7, 10, 8, 11),
    floor_vel_thr_per_frame: float = 0.005,
    floor_height_offset_m: float = 0.01,
    use_dbscan: bool = True,
    cluster_eps_m: float = 0.005,
    cluster_min_samples: int = 3,
) -> Tensor:
    idx = torch.tensor(joint_ids, device=kpts.device)
    pts = kpts[:, idx, :]
    spd = torch.linalg.norm(pts[1:] - pts[:-1], dim=-1)[..., [0, 2]].norm(dim=-1, keepdim=True) # (L-1, 4, 1) horizontal (XZ) speed
    y_all = pts[..., 1]
    return _estimate_floor_height_series(
        y_all,
        spd,
        floor_vel_thr_per_frame,
        floor_height_offset_m,
    )


def _tv1d_denoise_l2(y: Tensor, lam: float = 0.02, n_iter: int = 60) -> Tensor:
    # 1D ROF: min_x 0.5*||x - y||_2^2 + lam * TV(x)
    assert y.dim() == 1, "y must be 1D tensor"
    L = y.shape[0]
    if L <= 1:
        return y.clone()
    device = y.device
    dtype = y.dtype

    x = y.clone()
    x_bar = x.clone()
    p = torch.zeros(L - 1, device=device, dtype=dtype)

    tau = 0.5
    sigma = 0.5
    theta = 1.0

    for _ in range(n_iter):
        # p^{k+1} = proj_{|p| <= lam}(p^k + sigma * D x_bar)
        g = x_bar[1:] - x_bar[:-1]
        p = p + sigma * g
        p = torch.clamp(p, -lam, lam)

        # x^{k+1} = (x^k + tau*(-D^T p) + tau*y) / (1+tau)
        div_p = torch.zeros_like(x)
        div_p[0] = -p[0]
        if L > 2:
            div_p[1:-1] = p[:-1] - p[1:]
        div_p[-1] = p[-1]

        x_prev = x
        x = (x + tau * div_p + tau * y) / (1.0 + tau)
        x_bar = x + theta * (x - x_prev)

    return x


def _estimate_floor_height_series(
    y_all: Tensor,
    spd_xz: Tensor,
    floor_vel_thr_per_frame: float,
    floor_height_offset_m: float,
    win: int = 15,
) -> Tensor:
    L, K = y_all.shape
    device = y_all.device
    dtype = y_all.dtype

    if spd_xz.numel() == 0:
        slow_L = torch.ones((L, K), dtype=torch.bool, device=device)
    else:
        if spd_xz.shape[0] == L:
            slow_L = spd_xz < floor_vel_thr_per_frame
        else:
            slow = spd_xz < floor_vel_thr_per_frame
            slow_L = torch.cat([slow[:1], slow], dim=0)
    if slow_L.shape[1] != K:
        if slow_L.shape[1] > K:
            slow_L = slow_L[:, :K]
        else:
            pad = torch.ones((L, K - slow_L.shape[1]), dtype=slow_L.dtype, device=device)
            slow_L = torch.cat([slow_L, pad], dim=1)

    y_obs = torch.empty(L, device=device, dtype=dtype)
    q = 0.2
    for t in range(L):
        mask = slow_L[t]
        if mask.any():
            vals = y_all[t, mask]
            y_obs[t] = torch.quantile(vals, q)
        else:
            t0, t1 = max(0, t - win), min(L, t + 1)
            mask_win = slow_L[t0:t1].any(dim=1)
            if mask_win.any():
                vals = y_all[t0:t1][slow_L[t0:t1]]
                y_obs[t] = torch.quantile(vals, q)
            else:
                y_obs[t] = y_obs[t - 1] if t > 0 else torch.quantile(y_all[t], 0.1)

    if L >= 3:
        x = y_obs.view(1, 1, L)
        y_obs = torch.nn.functional.avg_pool1d(x, kernel_size=3, stride=1, padding=1).view(-1)

    lambda_tv = 0.02
    floor_h_t = _tv1d_denoise_l2(y_obs, lam=lambda_tv, n_iter=60)

    return floor_h_t - floor_height_offset_m


def _hysteresis_and_morph(
    prob: Tensor,
    on_thr: float = 0.7,
    off_thr: float = 0.5,
    morph_min_len: int = 3,
    morph_max_gap: int = 2,
) -> Tensor:
    L, K = prob.shape
    device = prob.device
    contact = torch.zeros_like(prob, dtype=torch.bool)
    prev = torch.zeros((K,), dtype=torch.bool, device=device)
    for t in range(L):
        on = prob[t] > on_thr
        off = prob[t] < off_thr
        prev = torch.where(on, torch.ones_like(prev, dtype=torch.bool), prev)
        prev = torch.where(off, torch.zeros_like(prev, dtype=torch.bool), prev)
        contact[t] = prev

    def morph_clean(x: Tensor, min_len: int = 3, max_gap: int = 2) -> Tensor:
        x = x.clone()
        cnt = 0
        for tt in range(L):
            if x[tt]:
                cnt += 1
            if (not x[tt]) or tt == L - 1:
                if 0 < cnt < min_len:
                    x[tt - cnt : tt] = False
                cnt = 0
        gap = 0
        last_on = -1
        for tt in range(L):
            if x[tt]:
                if 0 < gap <= max_gap and last_on >= 0:
                    x[last_on + 1 : tt] = True
                last_on = tt
                gap = 0
            else:
                gap += 1
        return x

    return torch.stack(
        [morph_clean(contact[:, j], morph_min_len, morph_max_gap) for j in range(K)],
        dim=1,
    )


def get_contact_prob(
    kpts: Tensor,
    joint_ids: tuple,
    vxz_thr: float,
    height_thr_per_joint: torch.Tensor,
    use_height: bool = True,
    floor_vel_thr_per_frame: float = 0.005,
    floor_height_offset_m: float = 0.0,
    use_dbscan: bool = False,
    cluster_eps_m: float = 0.005,
    cluster_min_samples: int = 3,
    floor_h_override: Optional[Tensor] = None,
    on_thr: float = 0.7,
    off_thr: float = 0.5,
    morph_min_len: int = 3,
    morph_max_gap: int = 2,
    use_vertical_cues: bool = False,
    vy_ext_thr: float = 1e-4,
    dvy_thr: float = 0.030,
    vy_static_thr: float = 5e-4,
):
    def _smooth1d_avg(x: Tensor, win: int = 3) -> Tensor:
        if win <= 1 or x.shape[0] < 3:
            return x
        pad_total = win - 1
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        xT = x.permute(1, 0).unsqueeze(1)
        x_pad = F.pad(xT, (pad_left, pad_right), mode="replicate")
        y = F.avg_pool1d(x_pad, kernel_size=win, stride=1, padding=0)
        return y.squeeze(1).permute(1, 0)

    L = kpts.shape[0]
    device = kpts.device
    idx = torch.tensor(joint_ids, device=device)
    pts = kpts[:, idx, :]
    K = pts.shape[1]
    y_all = pts[..., 1]
    if L >= 2:
        disp = pts[1:] - pts[:-1]
        spd_xz = torch.linalg.norm(disp[..., [0, 2]], dim=-1)
        spd_xz_L = torch.empty((L, K), device=device, dtype=pts.dtype)
        spd_xz_L[:-1] = spd_xz
        spd_xz_L[-1] = spd_xz_L[-2]
    else:
        disp = torch.empty((0, K, 3), device=device, dtype=pts.dtype)
        spd_xz = torch.zeros((L, K), device=device, dtype=pts.dtype)
        spd_xz_L = torch.zeros((L, K), device=device, dtype=pts.dtype)

    if floor_h_override is None:
        floor_h = _estimate_floor_height_series(
            y_all,
            (spd_xz if L >= 2 else torch.empty((0, K), device=device, dtype=pts.dtype)),
            floor_vel_thr_per_frame,
            floor_height_offset_m,
        )
    else:
        floor_h = floor_h_override.clone().detach()
    y_rel = y_all - floor_h[:, None]
    h_thr = height_thr_per_joint.to(device).view(1, -1).expand_as(y_rel)

    eps = torch.tensor(1e-6, device=device, dtype=pts.dtype)
    if use_height:
        height_band = torch.tensor(0.02, device=device, dtype=pts.dtype)
        s_height = torch.clamp((h_thr + height_band - y_rel) / height_band, 0.0, 1.0)
    else:
        s_height = torch.ones_like(y_rel, dtype=pts.dtype)
    vxz = torch.tensor(vxz_thr, device=device, dtype=pts.dtype)
    vel_band = torch.clamp(vxz * 0.5, min=eps)
    s_hspd = torch.clamp((vxz + vel_band - spd_xz_L) / vel_band, 0.0, 1.0)

    if use_vertical_cues:
        y_s = _smooth1d_avg(y_all, win=3)
        vy = torch.zeros((L, K), device=device, dtype=pts.dtype)
        if L >= 2:
            vy[1:] = y_s[1:] - y_s[:-1]
            vy[0] = vy[1]
        dvy = torch.zeros((L, K), device=device, dtype=pts.dtype)
        if L >= 2:
            dvy[1:] = vy[1:] - vy[:-1]
            dvy[0] = dvy[1]
        plate_win = 5
        plate_eps = 5e-5
        plate_dwell = 3
        x = y_s.permute(1, 0).unsqueeze(1)
        roll_min = (-F.max_pool1d(-x, kernel_size=plate_win, stride=1, padding=plate_win // 2)).squeeze(1).permute(1, 0)
        near_min = y_s <= (roll_min + plate_eps)
        run = torch.zeros((L, K), dtype=torch.int32, device=y_s.device)
        run[0] = near_min[0].int()
        for t in range(1, L):
            run[t] = torch.where(
                near_min[t],
                run[t - 1] + 1,
                torch.tensor(0, device=y_s.device, dtype=torch.int32),
            )
        y_minima = run >= plate_dwell
        vy_prev = torch.zeros_like(vy)
        if L >= 2:
            vy_prev[1:] = vy[:-1]
            vy_prev[0] = vy[0]
        not_rising = (vy_prev < vy_ext_thr) & (vy <= vy_ext_thr)
        vertical_on = not_rising & (y_minima | (dvy > dvy_thr))
        vertical_on = vertical_on & (y_rel < (h_thr + 0.01))
        sustain_mem = 5
        if sustain_mem > 1:
            x = vertical_on.float().permute(1, 0).unsqueeze(1)
            y = F.max_pool1d(x, kernel_size=sustain_mem, stride=1, padding=0)
            y = F.pad(y, (sustain_mem - 1, 0))
            vertical_on_recent = y.squeeze(1).permute(1, 0) > 0
        else:
            vertical_on_recent = vertical_on
        s_vertical = vertical_on_recent.float()
        boost = torch.tensor(0.7, device=device, dtype=pts.dtype)
    else:
        s_vertical = torch.zeros_like(y_rel, dtype=pts.dtype)
        boost = torch.tensor(0.0, device=device, dtype=pts.dtype)

    score_base = s_hspd * s_height
    prob = torch.maximum(score_base, (boost * s_vertical) * s_height)
    prob = _smooth1d_avg(prob, win=5)
    contact_hys = _hysteresis_and_morph(prob, on_thr, off_thr, morph_min_len, morph_max_gap)
    return prob, contact_hys


def get_foot_detect(
    kpts: Tensor,
    joint_ids: tuple = (7, 10, 8, 11), # [L_ankle, L_toe, R_ankle, R_toe]
    vel_thr_per_frame: float = 0.005,
    use_height: bool = True,
    toe_height_thr_m: float = 0.04,
    ankle_height_thr_m: float = 0.08,
    floor_h_override: Optional[Tensor] = None,
):
    height_thr = torch.tensor(
        [ankle_height_thr_m, toe_height_thr_m, ankle_height_thr_m, toe_height_thr_m],
        dtype=torch.float32,
    )
    return get_contact_prob(
        kpts=kpts,
        joint_ids=joint_ids,
        vxz_thr=vel_thr_per_frame,
        height_thr_per_joint=height_thr,
        use_height=use_height,
        floor_h_override=floor_h_override,
        on_thr=0.6,
        off_thr=0.4,
        morph_min_len=2,
        morph_max_gap=3,
    )


def get_foot_detect_deprecated(
    kpts: Tensor,
    fid_l: List[int] = [7, 10],
    fid_r: List[int] = [8, 11],
    thres: float = 2.5e-5,
):
    feet_l_x = (kpts[1:, fid_l, 0] - kpts[:-1, fid_l, 0]) ** 2
    feet_l_y = (kpts[1:, fid_l, 1] - kpts[:-1, fid_l, 1]) ** 2
    feet_l_z = (kpts[1:, fid_l, 2] - kpts[:-1, fid_l, 2]) ** 2
    feet_l = ((feet_l_x + feet_l_y + feet_l_z) < thres).float()
    feet_l = torch.cat([feet_l, torch.zeros_like(feet_l[:1])], dim=0)

    feet_r_x = (kpts[1:, fid_r, 0] - kpts[:-1, fid_r, 0]) ** 2
    feet_r_y = (kpts[1:, fid_r, 1] - kpts[:-1, fid_r, 1]) ** 2
    feet_r_z = (kpts[1:, fid_r, 2] - kpts[:-1, fid_r, 2]) ** 2
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < thres)).float()
    feet_r = torch.cat([feet_r, torch.zeros_like(feet_r[:1])], dim=0)
    return feet_l, feet_r


def get_hand_detect(
    kpts: Tensor,
    joint_ids: tuple = (20, 21), # [L_wrist, R_wrist]
    vel_thr_per_frame: float = 0.020,
    use_height: bool = False,
    hand_height_thr_m: float = 0.12,
    floor_h_override: Optional[Tensor] = None,
):
    height_thr = torch.tensor([hand_height_thr_m, hand_height_thr_m], dtype=torch.float32)
    return get_contact_prob(
        kpts=kpts,
        joint_ids=joint_ids,
        vxz_thr=vel_thr_per_frame,
        use_height=use_height,
        height_thr_per_joint=height_thr,
        floor_h_override=floor_h_override,
        on_thr=0.6,
        off_thr=0.4,
        morph_min_len=2,
        morph_max_gap=3,
        use_vertical_cues=True,
    )


def get_knee_detect(
    kpts: Tensor,
    joint_ids: tuple = (4, 5), # [L_knee, R_knee]
    vel_thr_per_frame: float = 0.005,
    use_height: bool = True,
    knee_height_thr_m: float = 0.09,
    floor_h_override: Optional[Tensor] = None,
):
    height_thr = torch.tensor([knee_height_thr_m, knee_height_thr_m], dtype=torch.float32)
    return get_contact_prob(
        kpts=kpts,
        joint_ids=joint_ids,
        vxz_thr=vel_thr_per_frame,
        use_height=use_height,
        height_thr_per_joint=height_thr,
        floor_h_override=floor_h_override,
        on_thr=0.7,
        off_thr=0.5,
        morph_min_len=3,
        morph_max_gap=2,
    )


def _unwrap_yaw(yaw: Tensor) -> Tensor:
    # yaw: (B, L)
    dy = yaw[:, 1:] - yaw[:, :-1]
    dy = (dy + torch.pi) % (2 * torch.pi) - torch.pi
    return torch.cat([yaw[:, :1], yaw[:, :1] + torch.cumsum(dy, dim=1)], dim=1)


def recover_root_kpts(
    root_ry_vel_world: Tensor,
    root_xz_vel_body: Tensor,
    root_y_transl: Tensor,
    root_rotations_mat_init: Optional[Tensor] = None,
    root_transl_init: Optional[Tensor] = None,
    smooth: bool = True,
    full_root_rotations_mat: Optional[Tensor] = None,
) -> Tensor:
    # root_ry_vel_world shape of (L, 1) or (B, L, 1)
    # root_xz_vel_body shape of (L, 2) or (B, L, 2)
    # root_y_transl shape of (L, 1) or (B, L, 1)
    # root_rotations_mat_init shape of (3, 3) or (B, 3, 3) or None

    device = root_ry_vel_world.device

    if root_ry_vel_world.dim() == 3:
        B, L = root_ry_vel_world.shape[:2]
        is_batch = True
    else:
        L = root_ry_vel_world.shape[0]
        B = 1
        is_batch = False
        root_ry_vel_world = root_ry_vel_world.unsqueeze(0)
        root_xz_vel_body = root_xz_vel_body.unsqueeze(0)
        root_y_transl = root_y_transl.unsqueeze(0)

    if root_rotations_mat_init is None:
        root_rotations_mat_init = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
    else:
        if root_rotations_mat_init.dim() == 2:
            root_rotations_mat_init = root_rotations_mat_init.unsqueeze(0).expand(B, -1, -1)
        assert root_rotations_mat_init.shape == (
            B,
            3,
            3,
        ), f"Shape error: {root_rotations_mat_init.shape}, should be (B, 3, 3) or (3, 3)"

    if root_transl_init is None:
        root_transl_init = torch.zeros(B, 3, device=device)
    else:
        if root_transl_init.dim() == 1:
            root_transl_init = root_transl_init.unsqueeze(0).expand(B, -1)
        assert root_transl_init.shape == (
            B,
            3,
        ), f"Shape error: {root_transl_init.shape}, should be (B, 3) or (3,)"

    if full_root_rotations_mat is not None:
        full_yaw_seq = _get_heading(full_root_rotations_mat)
        if full_yaw_seq.dim() == 1:
            yaw_init = full_yaw_seq[0].view(1)
        else:
            yaw_init = full_yaw_seq[:, 0]
    elif root_rotations_mat_init is not None:
        yaw_init = _get_heading_single_frame(root_rotations_mat_init)
    else:
        yaw_init = torch.zeros(B, device=device)

    yaw_delta = root_ry_vel_world.squeeze(-1)
    yaw_delta = torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta)) # wrap to [-pi, pi]
    if smooth and yaw_delta.shape[1] >= 3:
        max_delta = math.radians(60.0) # max 60° per frame
        yaw_delta = torch.clamp(yaw_delta, -max_delta, max_delta)
        yd = yaw_delta.unsqueeze(1)
        yd = F.avg_pool1d(F.pad(yd, (1, 1), mode="replicate"), kernel_size=3, stride=1)
        yaw_delta = yd.squeeze(1)

    yaw_seq = yaw_init.unsqueeze(1) + torch.cumsum(yaw_delta, dim=1)
    yaw_seq = _unwrap_yaw(yaw_seq)

    cos_yaw = torch.cos(yaw_seq[:, :-1])
    sin_yaw = torch.sin(yaw_seq[:, :-1])

    root_xz_vel_body_s = root_xz_vel_body
    if smooth and root_xz_vel_body.shape[1] >= 3:
        vxz = root_xz_vel_body.transpose(1, 2)
        vxz = F.avg_pool1d(F.pad(vxz, (1, 1), mode="replicate"), kernel_size=3, stride=1)
        root_xz_vel_body_s = vxz.transpose(1, 2)

    root_x_vel_body = root_xz_vel_body_s[:, :-1, 0]
    root_z_vel_body = root_xz_vel_body_s[:, :-1, 1]

    root_x_vel_world = cos_yaw * root_x_vel_body + sin_yaw * root_z_vel_body
    root_z_vel_world = -sin_yaw * root_x_vel_body + cos_yaw * root_z_vel_body
    root_xz_vel_world = torch.stack([root_x_vel_world, root_z_vel_world], dim=-1)

    # xyz
    root_xz_transl = torch.zeros(B, L, 2, device=device)
    root_xz_transl[:, 0] = root_transl_init[:, [0, 2]]
    root_xz_transl[:, 1:] = torch.cumsum(root_xz_vel_world, dim=1) + root_xz_transl[:, 0:1]
    reconstructed_xyz = torch.cat(
        [root_xz_transl[:, :, 0:1], root_y_transl, root_xz_transl[:, :, 1:2]], dim=-1
    )
    return reconstructed_xyz


def recover_root_rot_pos(data: Tensor) -> Tuple[Tensor, Tensor]:
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    """Get Y-axis rotation from rotation velocity"""
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)
    # construct quaternion in (x,y,z,w) format
    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    # FIXME: here is inconsistent with the original code
    r_rot_quat[..., 1] = torch.sin(r_rot_ang / 2)
    r_rot_quat[..., 3] = torch.cos(r_rot_ang / 2)
    """Get root position"""
    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    r_pos = quaternion_rotate_vector(quaternion_inverse(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_ric(hml263: Tensor, joints_num: int) -> Tensor:
    root_rot_quat, root_pos = recover_root_rot_pos(hml263)

    positions = hml263[..., 4 : 4 + (joints_num - 1) * 3]
    positions = positions.view(positions.shape[:-1] + (-1, 3))
    """Add Y-axis rotation to local joints"""
    positions = quaternion_rotate_vector(
        quaternion_inverse(root_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)),
        positions,
    )
    """Add root XZ to joints"""
    positions[..., 0] += root_pos[..., 0:1]
    positions[..., 2] += root_pos[..., 2:3]
    """Concate root and joints"""
    positions = torch.cat([root_pos.unsqueeze(-2), positions], dim=-2)
    return positions


def correct_translation_with_contact(
    k3d: Tensor,
    transl: Tensor,
    prob: Tensor,
    joint_ids: List[int] = [7, 10, 8, 11],
    on_thr: float = 0.50,
    off_thr: float = 0.30,
    morph_min_len: int = 3,
    morph_max_gap: int = 2,
    eps: float = 1e-8,
) -> Tensor:
    if k3d.dim() == 3:
        k3d = k3d.unsqueeze(0)
    if transl.dim() == 2:
        transl = transl.unsqueeze(0)
    B, L, J, _ = k3d.shape
    K = len(joint_ids)
    if prob.dim() == 2:
        contact = _hysteresis_and_morph(prob, on_thr, off_thr, morph_min_len, morph_max_gap)
        contact = contact.unsqueeze(0).expand(B, -1, -1)
        prob_b = prob.unsqueeze(0).expand(B, -1, -1)
    elif prob.dim() == 3:
        contact_list = []
        prob_b = prob
        for b in range(prob.shape[0]):
            contact_list.append(_hysteresis_and_morph(prob[b], on_thr, off_thr, morph_min_len, morph_max_gap))
        contact = torch.stack(contact_list, dim=0)
    else:
        raise ValueError("prob must be (L,K) or (B,L,K)")
    pair_contact = contact[:, 1:] & contact[:, :-1]
    pred_j3d_static = k3d[:, :, joint_ids, :]
    pred_j3d_static_disp = pred_j3d_static[:, 1:] - pred_j3d_static[:, :-1]
    w = 0.5 * (prob_b[:, 1:] + prob_b[:, :-1])
    w = w * pair_contact.float()
    w_sum = w.sum(dim=2, keepdim=True).clamp_min(eps)
    drift = (pred_j3d_static_disp * w.unsqueeze(-1)).sum(dim=2) / w_sum
    drift[..., 1] = 0.0
    w_disp = transl[:, 1:] - transl[:, :-1]
    w_disp_new = w_disp - drift
    transl_fixed = torch.zeros_like(transl)
    transl_fixed[:, 0] = transl[:, 0]
    transl_fixed[:, 1:] = transl_fixed[:, :1] + torch.cumsum(w_disp_new, dim=1)
    return transl_fixed.squeeze(0) if transl_fixed.shape[0] == 1 else transl_fixed


def smooth_quats(quats: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    from .rotation_converter import (
        gaussian_kernel1d,
        quaternion_fix_continuity,
        slice_seq_with_padding,
        wavg_quaternion_markley,
    )

    if len(quats) == 0 or sigma <= 0:
        return quats.copy()

    q_all = quaternion_fix_continuity(torch.from_numpy(quats)).numpy()

    results = q_all.copy()
    truncate = 4.0
    order = 0
    lw = int(truncate * float(sigma) + 0.5)
    weights = gaussian_kernel1d(sigma=sigma, order=order, radius=lw)[::-1]
    kernel_len = len(weights)

    for fr in range(len(q_all)):
        cur_quats = slice_seq_with_padding(q_all, fr, kernel_len)
        ref = cur_quats[kernel_len // 2 : kernel_len // 2 + 1]
        dots = (cur_quats * ref).sum(axis=-1, keepdims=True)
        cur_quats = np.where(dots < 0.0, -cur_quats, cur_quats)

        results[fr, :] = wavg_quaternion_markley(cur_quats, weights)

    return results.copy()


def smooth_rotation(
    quats: np.ndarray,
    joint_names: List[str],
    smooth_joints: List[str],
    sigma: float = 1.0,
) -> np.ndarray:
    if quats.ndim == 4:
        is_batch = True
    else:
        is_batch = False
        quats = quats[None, ...]
    for b in range(quats.shape[0]):
        for j in smooth_joints:
            j_idx = joint_names.index(j)
            cur_quats = quats[b, :, j_idx].copy()
            cur_quats_t = quaternion_fix_continuity(torch.from_numpy(cur_quats)).numpy()
            quats[b, :, j_idx] = smooth_quats(cur_quats_t, sigma=sigma)
    if not is_batch:
        quats = quats.squeeze(0)
    return quats


def unwrap_euler_over_time(xyz: torch.Tensor) -> torch.Tensor:
    # xyz: (B, L, J, 3)
    # y[t] = y[0] + cumsum(wrap(Δy))
    y = xyz.clone()
    dy = torch.atan2(torch.sin(y[:, 1:] - y[:, :-1]), torch.cos(y[:, 1:] - y[:, :-1]))
    y[:, 1:] = y[:, :1] + torch.cumsum(dy, dim=1)
    return y

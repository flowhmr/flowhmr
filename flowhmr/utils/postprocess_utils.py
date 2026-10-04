import numpy as np
import torch

class TrajectoryCorrector:

    def __init__(self, margins, w_pos=0.01, w_vel=1.0, w_acc=10.0,
                 hard_weight=1e6):
        if torch.is_tensor(margins):
            margins = margins.detach().cpu().numpy()
        self.margins = np.asarray(margins, dtype=np.float64)
        self.w_pos = float(w_pos)
        self.w_vel = float(w_vel)
        self.w_acc = float(w_acc)
        self.hard_weight = float(hard_weight)

    def interpolate(self, x_orig, obs):
        device = x_orig.device
        dtype = torch.float64
        T, C = x_orig.shape

        x_o = x_orig.to(dtype)
        ob = obs.to(dtype)

        m_pos = torch.from_numpy((self.margins >= 0.0).astype(np.float64)).to(device)
        w_pos_diag = m_pos * self.hard_weight + (1.0 - m_pos) * self.w_pos

        D1 = torch.zeros(T - 1, T, device=device, dtype=dtype)
        idx = torch.arange(T - 1, device=device)
        D1[idx, idx] = -1.0
        D1[idx, idx + 1] = 1.0

        D2 = torch.zeros(T - 2, T, device=device, dtype=dtype)
        idx2 = torch.arange(T - 2, device=device)
        D2[idx2, idx2] = 1.0
        D2[idx2, idx2 + 1] = -2.0
        D2[idx2, idx2 + 2] = 1.0

        D1tD1 = D1.transpose(0, 1) @ D1
        D2tD2 = D2.transpose(0, 1) @ D2

        # A = diag(w_pos_diag) + w_v * D1^T D1 + w_a * D2^T D2
        A = torch.diag(w_pos_diag) + self.w_vel * D1tD1 + self.w_acc * D2tD2

        A = A + 1e-9 * torch.eye(T, device=device, dtype=dtype)

        # rhs: (T, C)
        pos_target = m_pos.unsqueeze(-1) * ob + (1.0 - m_pos).unsqueeze(-1) * x_o
        rhs = w_pos_diag.unsqueeze(-1) * pos_target + self.w_vel * (D1tD1 @ x_o) + self.w_acc * (D2tD2 @ x_o)

        x_fixed = torch.linalg.solve(A, rhs)
        return x_fixed.to(x_orig.dtype)


def _forward_smpl_vertices(smpl_mesh, output: dict):
    """Dense SMPL mesh FK → (B, L, V, 3) vertices."""
    rot6d = output["rot6d"]
    B, L = rot6d.shape[:2]
    J = rot6d.shape[2]
    rot6d_flat = rot6d.reshape(B * L, J, 6)
    shapes = output["shapes"].expand(B, L, -1).reshape(B * L, -1)
    trans_flat = output["trans"].reshape(B * L, 3)
    out = smpl_mesh({"rot6d": rot6d_flat, "shapes": shapes, "trans": trans_flat})
    verts = out["vertices"]  # (B*L, V, 3)
    V = verts.shape[1]
    return verts.reshape(B, L, V, 3)

def _forward_smpl_joints(body_model, output):
    rot6d = output["rot6d"]
    B, T, J = rot6d.shape[:3]
    rot6d_flat = rot6d.reshape(B * T, J, 6)
    shapes = output["shapes"].expand(B, T, -1).reshape(B * T, -1)
    trans_flat = output["trans"].reshape(B * T, 3)
    out = body_model({"rot6d": rot6d_flat, "shapes": shapes, "trans": trans_flat})
    keypoints3d = out["keypoints3d"].reshape(B, T, J, 3)
    transforms = out["transforms"].reshape(B, T, J, 4, 4)
    return keypoints3d, transforms


def _find_segments(mask_np):
    padded = np.concatenate([[0], mask_np.astype(np.int32), [0]])
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [[int(s), int(e)] for s, e in zip(starts, ends)]

def _split_interval_by_position_range(positions, start, end, range_threshold):
    if end - start <= 1:
        return [(start, end)]
    pos = positions[start:end]
    full_range = pos.max(axis=0) - pos.min(axis=0)
    if np.all(full_range <= range_threshold):
        return [(start, end)]

    sub_intervals = []
    cur_start = start
    while cur_start < end:
        cur_min = positions[cur_start].copy()
        cur_max = positions[cur_start].copy()
        cur_end = cur_start + 1
        while cur_end < end:
            new_min = np.minimum(cur_min, positions[cur_end])
            new_max = np.maximum(cur_max, positions[cur_end])
            if np.any(new_max - new_min > range_threshold):
                break
            cur_min, cur_max = new_min, new_max
            cur_end += 1
        sub_intervals.append((cur_start, cur_end))
        cur_start = cur_end
    return sub_intervals


def find_foot_contact_interval(foot_contacts, foot_contact_positions, joint_names,
    threshold=0.5, velocity_threshold=0.15 * 3, shrink_frame=3,
    position_range_threshold=0.1):
    foot_contact_mask = foot_contacts.cpu().numpy() > threshold
    contact_intervals = {}
    if isinstance(position_range_threshold, dict):
        _range_thresholds = position_range_threshold
    else:
        _range_thresholds = {}
    _default_range_threshold = 0.1 if isinstance(position_range_threshold, dict) else float(position_range_threshold)

    assert len(joint_names) == foot_contact_mask.shape[-1]
    for i, jname in enumerate(joint_names):
        jnt_range_threshold = _range_thresholds.get(jname, _default_range_threshold)
        velocity = torch.norm(foot_contact_positions[1:, i, :] - foot_contact_positions[:-1, i, :], dim=-1) * 30
        velocity = torch.cat([velocity, velocity[-1:]], dim=0).cpu().numpy()
        flag_static = velocity < velocity_threshold
        contact_interval = _find_segments(foot_contact_mask[..., i] & flag_static)

        positions_np = foot_contact_positions[:, i, :].cpu().numpy()

        # Supplementary: no contact detected but velocity low AND position range small
        non_contact_static = (~foot_contact_mask[..., i]) & (velocity < velocity_threshold / 3)
        extra_intervals = _find_segments(non_contact_static)
        extra_intervals = [
            (s, e) for s, e in extra_intervals
            if (e - s > shrink_frame * 2 + 1)
            and np.all(positions_np[s:e].max(axis=0) - positions_np[s:e].min(axis=0) <= jnt_range_threshold)
        ]
        contact_interval = sorted(contact_interval + extra_intervals, key=lambda x: x[0])
        contact_interval = [(s, e) for s, e in contact_interval if e - s > shrink_frame * 2 + 1]

        split_intervals = []
        for s, e in contact_interval:
            split_intervals.extend(_split_interval_by_position_range(positions_np, s, e, jnt_range_threshold))
        split_intervals_ = []
        for s, e in split_intervals:
            if s != 0:
                s = s + shrink_frame
            if e != foot_contact_positions.shape[0]:
                e = e - shrink_frame
            split_intervals_.append((s, e))
        split_intervals = split_intervals_
        split_intervals = [(s, e) for s, e in split_intervals if e - s > shrink_frame * 2 + 1]
        contact_intervals[jname] = split_intervals
    return contact_intervals

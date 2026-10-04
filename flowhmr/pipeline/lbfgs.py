import torch
from typing import Callable
from ..core.math.geometry import rot6d_to_rotation_matrix, rotation_matrix_to_rot6d


def get_optimizer(opt_params):
    optimizer = torch.optim.LBFGS(
        opt_params, lr=1.0, max_iter=20,
        line_search_fn="strong_wolfe",
    )
    return optimizer


def smooth_loss(x, fps=30):
    assert x.shape[0] > 2
    diff = (x[1:] - x[:-1]) * fps
    jitter = (diff[1:] - diff[:-1]) * fps
    return (jitter ** 2).mean()


def get_trans_closure(optimizer, body_model, input_params, indices, target_positions, weight_dict={'mse': 10000.0, 'smooth': 0.01}):
    def closure(return_dict=False):
        optimizer.zero_grad()

        joints_all = body_model(
            {
                "rot6d": input_params["rot6d"][0],
                "shapes": input_params["shapes"][0],
                "trans": input_params["trans"][0],
            }
        )["keypoints3d"]  # (L, J, 3)
        # foot target loss
        joints_selected = joints_all[..., indices, :]
        loss_mse = ((joints_selected - target_positions) ** 2).mean()

        loss_dict = {
            "mse": loss_mse,
            "smooth": smooth_loss(input_params["trans"][0]),
        }
        if return_dict:
            return {k: loss_dict[k] * weight_dict[k] for k in loss_dict}
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict)
        loss.backward()
        return loss
    return closure

ROT6D_SMOOTH_WEIGHT = 0.001
REG_WEIGHT = 1.0

def get_chain_closure(optimizer, body_model, input_params, indices, target_positions, weight_dict={'mse': 10000.0, 'smooth': ROT6D_SMOOTH_WEIGHT, 'reg': REG_WEIGHT}):
    rot6d = input_params["rot6d"]
    shapes = input_params["shapes"]
    trans = input_params["trans"]
    rot6d_opt = input_params["rot6d_opt"]
    rot6d_indices = input_params["rot6d_indices"]
    rot6d_opt_init = rot6d_opt.detach().clone()
    def closure(return_dict=False):
        optimizer.zero_grad()

        rot6d_full = rot6d.clone()
        rot6d_full[:, :, rot6d_indices] = rot6d_opt

        joints_all = body_model(
            {
                "rot6d": rot6d_full[0],
                "shapes": shapes[0].detach(),
                "trans": trans[0],
            }
        )["keypoints3d"]  # (L, J, 3)
        joints_selected = joints_all[..., indices, :]

        # foot target loss
        loss_mse = ((joints_selected - target_positions) ** 2).mean()


        loss_dict = {
            "mse": loss_mse,
            "smooth": smooth_loss(rot6d_opt[0]),
            "reg": ((rot6d_opt - rot6d_opt_init) ** 2).mean(),
        }
        if return_dict:
            return {k: loss_dict[k] * weight_dict[k] for k in loss_dict}
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict)
        loss.backward()
        return loss
    return closure


def _build_Rx(angle):
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    zeros = torch.zeros_like(cos_a)
    ones = torch.ones_like(cos_a)
    return torch.stack([
        torch.stack([ones, zeros, zeros], dim=-1),
        torch.stack([zeros, cos_a, -sin_a], dim=-1),
        torch.stack([zeros, sin_a, cos_a], dim=-1),
    ], dim=-2)


def get_ankle_chain_closure(
    optimizer, body_model, input_params, target_joint_idx, target_positions,
    weight_dict={
        'mse': 10000.0,
        'hip_smooth': ROT6D_SMOOTH_WEIGHT, 'hip_reg': REG_WEIGHT,
        'knee_smooth': 0.01, 'knee_reg': 10.0,
    },
):
    rot6d = input_params["rot6d"]
    shapes = input_params["shapes"]
    trans = input_params["trans"]
    hip_rot6d_opt = input_params["hip_rot6d_opt"]
    knee_flex_opt = input_params["knee_flex_opt"]
    hip_joint_idx = input_params["hip_joint_idx"]
    knee_joint_idx = input_params["knee_joint_idx"]
    knee_rot6d_orig = input_params["knee_rot6d_orig"]

    hip_rot6d_init = hip_rot6d_opt.detach().clone()
    knee_R_orig = rot6d_to_rotation_matrix(knee_rot6d_orig)

    def closure(return_dict=False):
        optimizer.zero_grad()

        # knee: R_orig @ Rx(delta_flex) → rot6d
        Rx = _build_Rx(knee_flex_opt[..., 0])
        knee_R_new = knee_R_orig @ Rx
        knee_rot6d_new = rotation_matrix_to_rot6d(knee_R_new)

        rot6d_full = rot6d.clone()
        rot6d_full[:, :, hip_joint_idx] = hip_rot6d_opt
        rot6d_full[:, :, knee_joint_idx] = knee_rot6d_new

        joints_all = body_model({
            "rot6d": rot6d_full[0],
            "shapes": shapes[0].detach(),
            "trans": trans[0],
        })["keypoints3d"]  # (T, J, 3)
        joints_selected = joints_all[..., target_joint_idx, :]

        loss_mse = ((joints_selected - target_positions) ** 2).mean()

        hip_smooth = smooth_loss(hip_rot6d_opt[0])
        knee_smooth = smooth_loss(knee_flex_opt[0])
        hip_reg = ((hip_rot6d_opt - hip_rot6d_init) ** 2).mean()
        knee_reg = (knee_flex_opt ** 2).mean()

        loss_dict = {
            "mse": loss_mse,
            "hip_smooth": hip_smooth,
            "hip_reg": hip_reg,
            "knee_smooth": knee_smooth,
            "knee_reg": knee_reg,
        }
        if return_dict:
            return {k: loss_dict[k] * weight_dict[k] for k in loss_dict}
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict)
        loss.backward()
        return loss
    return closure


def get_mesh_closure(optimizer, smpl_mesh, input_params, weight_dict={'mse': 10000.0, 'static': 100.0, 'reg': 100.0, 'smooth': ROT6D_SMOOTH_WEIGHT}):
    rot6d = input_params["rot6d"]
    shapes = input_params["shapes"]
    trans = input_params["trans"]
    rot6d_opt = input_params["rot6d_opt"]
    rot6d_opt_init = rot6d_opt.detach().clone()
    rot6d_indices = input_params["rot6d_indices"]
    contact_intervals = input_params["contact_intervals"]
    part_indices = input_params["part_indices"]
    contact_mask = torch.zeros(rot6d.shape[1], device=rot6d.device, dtype=torch.bool)
    for s, e in contact_intervals:
        contact_mask[s:e] = True

    vertices = smpl_mesh(
        {
            "rot6d": rot6d[0],
            "shapes": shapes[0].detach(),
            "trans": trans[0],
        },
        sample_indices=part_indices,
    )["vertices"]  # (L, N_verts, 3)
    _, min_y_idx = vertices[..., 1].min(dim=-1)


    def closure(return_dict=False):
        optimizer.zero_grad()

        rot6d_full = rot6d.clone()
        rot6d_full[:, :, rot6d_indices] = rot6d_opt

        vertices = smpl_mesh(
            {
                "rot6d": rot6d_full[0],
                "shapes": shapes[0].detach(),
                "trans": trans[0],
            },
            sample_indices=part_indices,
        )["vertices"]  # (L, N_verts, 3)
        min_y = vertices[torch.arange(vertices.shape[0]), min_y_idx, 1]
        diff = (vertices[1:] - vertices[:-1]) * 30
        diff_min_y = diff[torch.arange(diff.shape[0]), min_y_idx[:-1]]
        diff_min_y = torch.cat([diff_min_y, diff_min_y[-1:]], dim=0)

        loss_mse = (min_y ** 2 * contact_mask).sum()
        loss_static = ((diff_min_y ** 2).sum(dim=-1) * contact_mask).sum()
        if contact_mask.any():
            loss_mse = loss_mse / contact_mask.sum()
            loss_static = loss_static / contact_mask.sum()
        loss_dict = {
            "mse": loss_mse,
            "static": loss_static,
            "reg": ((rot6d_opt - rot6d_opt_init) ** 2).mean(),
            "smooth": smooth_loss(rot6d_opt[0]),
        }
        if return_dict:
            return {k: loss_dict[k] * weight_dict[k] for k in loss_dict}
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict)
        loss.backward()
        return loss
    return closure


def get_ground_correction_closure(
    optimizer, body_model, smpl_mesh, input_params,
    weight_dict=None,
):
    if weight_dict is None:
        weight_dict = {
            'ground': 50000.0, 'float': 50000.0,
            'mesh_ground': 50000.0,
            'vel_preserve': 100.0,
            'rot_vel_preserve': 100.0,
            'trans_smooth': 0.01, 'trans_reg': 10.0,
            'hip_reg': REG_WEIGHT,
            'knee_reg': 10.0,
            'ankle_reg': REG_WEIGHT,
        }

    rot6d = input_params["rot6d"]
    shapes = input_params["shapes"]
    trans = input_params["trans"]
    trans_y_opt = input_params["trans_y_opt"]
    trans_y_init = trans_y_opt.detach().clone()

    sides = ['L', 'R']
    hip_rot6d_opt = {s: input_params[f"hip_rot6d_opt_{s}"] for s in sides}
    knee_flex_opt = {s: input_params[f"knee_flex_opt_{s}"] for s in sides}
    ankle_rot6d_opt = {s: input_params[f"ankle_rot6d_opt_{s}"] for s in sides}
    hip_idx = {s: input_params[f"hip_idx_{s}"] for s in sides}
    knee_idx = {s: input_params[f"knee_idx_{s}"] for s in sides}
    ankle_idx = {s: input_params[f"ankle_idx_{s}"] for s in sides}
    knee_rot6d_orig = {s: input_params[f"knee_rot6d_orig_{s}"] for s in sides}
    knee_R_orig = {s: rot6d_to_rotation_matrix(knee_rot6d_orig[s]) for s in sides}
    contact_mask = {s: input_params[f"contact_mask_{s}"] for s in sides}
    foot_vids = {'L': input_params["left_foot_vids"], 'R': input_params["right_foot_vids"]}
    contact_joint_ids = input_params["contact_joint_ids"]

    hip_rot6d_init = {s: hip_rot6d_opt[s].detach().clone() for s in sides}
    ankle_rot6d_init = {s: ankle_rot6d_opt[s].detach().clone() for s in sides}

    # non_leg_min_y ≈ trans_y + offset → offset = non_leg_min_y - trans_y
    non_leg_min_y_offset = input_params["non_leg_min_y_offset"].detach()  # (T,)

    float_margin = 0.005 # 5mm

    with torch.no_grad():
        ref_fk = body_model({
            "rot6d": rot6d[0], "shapes": shapes[0].detach(), "trans": trans[0],
        })
        ref_joint_pos = ref_fk["keypoints3d"][:, contact_joint_ids, :]  # (T, 4, 3)
        ref_joint_vel = (ref_joint_pos[1:] - ref_joint_pos[:-1]) * 30

    L_ankle_idx_val = ankle_idx['L']
    R_ankle_idx_val = ankle_idx['R']
    with torch.no_grad():
        ref_transforms = body_model({
            "rot6d": rot6d[0], "shapes": shapes[0].detach(), "trans": trans[0],
        })["transforms"]  # (T, J, 4, 4)
        ref_global_L = ref_transforms[:, L_ankle_idx_val, :3, :3]
        ref_global_R = ref_transforms[:, R_ankle_idx_val, :3, :3]
        ref_global_rel_L = ref_global_L[:-1].transpose(-1, -2) @ ref_global_L[1:]
        ref_global_rel_R = ref_global_R[:-1].transpose(-1, -2) @ ref_global_R[1:]

    def closure(return_dict=False):
        optimizer.zero_grad()

        rot6d_full = rot6d.clone()
        trans_full = trans.clone()
        trans_full[:, :, 1:2] = trans_y_opt

        for s in sides:
            rot6d_full[:, :, hip_idx[s]] = hip_rot6d_opt[s]
            Rx = _build_Rx(knee_flex_opt[s][..., 0])
            knee_R_new = knee_R_orig[s] @ Rx
            rot6d_full[:, :, knee_idx[s]] = rotation_matrix_to_rot6d(knee_R_new)
            rot6d_full[:, :, ankle_idx[s]] = ankle_rot6d_opt[s]

        # FK → joints + mesh
        fk_out = body_model({
            "rot6d": rot6d_full[0], "shapes": shapes[0].detach(), "trans": trans_full[0],
        })
        keypoints3d = fk_out["keypoints3d"]  # (T, J, 3)

        all_foot_vids = torch.cat([foot_vids['L'], foot_vids['R']])
        vertices = smpl_mesh({
            "rot6d": rot6d_full[0], "shapes": shapes[0].detach(), "trans": trans_full[0],
        }, sample_indices=all_foot_vids)["vertices"]  # (T, N_foot, 3)

        n_L = foot_vids['L'].shape[0]
        verts_L = vertices[:, :n_L, :]
        verts_R = vertices[:, n_L:, :]

        min_y_L_all = verts_L[:, :, 1].min(dim=-1).values
        min_y_R_all = verts_R[:, :, 1].min(dim=-1).values
        loss_ground = (torch.relu(-min_y_L_all).pow(2).mean() + torch.relu(-min_y_R_all).pow(2).mean()) / 2

        loss_float = torch.tensor(0.0, device=rot6d.device)
        n_contact = 0
        for s, verts_s in [('L', verts_L), ('R', verts_R)]:
            mask = contact_mask[s]
            if not mask.any():
                continue
            min_y_s = verts_s[mask, :, 1].min(dim=-1).values
            loss_float = loss_float + torch.relu(min_y_s - float_margin).pow(2).mean()
            n_contact += 1
        if n_contact > 0:
            loss_float = loss_float / n_contact

        approx_non_leg_min_y = trans_y_opt[0, :, 0] + non_leg_min_y_offset
        loss_mesh_ground = torch.relu(-approx_non_leg_min_y).pow(2).mean()

        joint_pos = keypoints3d[:, contact_joint_ids, :]
        joint_vel = (joint_pos[1:] - joint_pos[:-1]) * 30
        loss_vel = ((joint_vel - ref_joint_vel) ** 2).mean()

        cur_transforms = fk_out["transforms"]  # (T, J, 4, 4)
        cur_global_L = cur_transforms[:, L_ankle_idx_val, :3, :3]
        cur_global_R = cur_transforms[:, R_ankle_idx_val, :3, :3]
        cur_global_rel_L = cur_global_L[:-1].transpose(-1, -2) @ cur_global_L[1:]
        cur_global_rel_R = cur_global_R[:-1].transpose(-1, -2) @ cur_global_R[1:]
        loss_rot_vel = (((cur_global_rel_L - ref_global_rel_L) ** 2).mean()
                        + ((cur_global_rel_R - ref_global_rel_R) ** 2).mean()) / 2

        # --- smooth + reg ---
        loss_trans_smooth = smooth_loss(trans_y_opt[0])
        loss_trans_reg = ((trans_y_opt - trans_y_init) ** 2).mean()

        loss_hip_reg = sum(((hip_rot6d_opt[s] - hip_rot6d_init[s]) ** 2).mean() for s in sides) / 2
        loss_knee_reg = sum((knee_flex_opt[s] ** 2).mean() for s in sides) / 2
        loss_ankle_reg = sum(((ankle_rot6d_opt[s] - ankle_rot6d_init[s]) ** 2).mean() for s in sides) / 2

        loss_dict = {
            "ground": loss_ground, "float": loss_float,
            "mesh_ground": loss_mesh_ground,
            "vel_preserve": loss_vel,
            "rot_vel_preserve": loss_rot_vel,
            "trans_smooth": loss_trans_smooth, "trans_reg": loss_trans_reg,
            "hip_reg": loss_hip_reg,
            "knee_reg": loss_knee_reg,
            "ankle_reg": loss_ankle_reg,
        }
        if return_dict:
            return {k: loss_dict[k] * weight_dict[k] for k in loss_dict}
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict)
        loss.backward()
        return loss
    return closure


def _run_lbfgs(
    optimizer: torch.optim.LBFGS,
    closure_fn: Callable,
    max_outer_iters: int = 50,
    stage_name: str = "lbfgs",
    conv_tol: float = 1e-6,
    conv_patience: int = 3,
):
    prev_loss = None
    patience_cnt = 0
    for it in range(max_outer_iters):
        with torch.no_grad():
            loss_dict = closure_fn(return_dict=True)
            loss_sum = sum(loss_dict.values())

            if prev_loss is not None and prev_loss > 0:
                rel_change = abs(prev_loss - loss_sum) / prev_loss
                if rel_change < conv_tol:
                    patience_cnt += 1
                    if patience_cnt >= conv_patience:
                        break
                else:
                    patience_cnt = 0
            prev_loss = loss_sum
        optimizer.step(closure_fn)
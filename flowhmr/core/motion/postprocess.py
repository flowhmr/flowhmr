import torch
from torch.cuda.amp import autocast
import numpy as np
import os
from datetime import datetime

from ..math.geometry import rotation_matrix_to_rot6d, rot6d_to_rotation_matrix

from . import matrix
from .ccd_ik import CCD_IK
from .ccd_ik_full import CCDIKFull, Activation
from ..utils.net_utils import gaussian_smooth
from ..bodymodels.fk_utils import get_joints_from_smpl_params, get_fkmat_from_smpl_params


def _save_ik_debug_comparison(fk_j3d_original, post_target_j3d, fk_j3d_after_ik, joint_ids, joint_names, debug_dir, timestamp):
    import matplotlib.pyplot as plt

    os.makedirs(debug_dir, exist_ok=True)
    axis_names = ['X', 'Y', 'Z']

    data_orig = fk_j3d_original[0].cpu().numpy()
    data_target = post_target_j3d[0].cpu().numpy()
    data_after = fk_j3d_after_ik[0].cpu().numpy()

    num_joints = len(joint_ids)
    fig, axes = plt.subplots(3, num_joints, figsize=(num_joints * 3, 9), squeeze=False)
    fig.suptitle('End-Effector Trajectories: Original FK vs IK Target vs Post-IK FK', fontsize=13)

    for col_idx, (jid, jname) in enumerate(zip(joint_ids, joint_names)):
        for axis_idx in range(3):
            ax = axes[axis_idx, col_idx]
            ax.plot(data_orig[:, jid, axis_idx], linewidth=0.8, label='Original FK', alpha=0.7)
            ax.plot(data_target[:, jid, axis_idx], linewidth=0.8, label='IK Target', linestyle='--')
            ax.plot(data_after[:, jid, axis_idx], linewidth=0.8, label='Post-IK FK', alpha=0.7)
            if axis_idx == 0:
                ax.set_title(f'{jname} (j{jid})', fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f'{axis_names[axis_idx]} axis', fontsize=10)
            if axis_idx == 2:
                ax.set_xlabel('Frame', fontsize=9)
            ax.grid(True, alpha=0.3)
            if axis_idx == 0 and col_idx == num_joints - 1:
                ax.legend(fontsize=7)

    # unify ylim across all joints for each axis (each row shares the same y range)
    for axis_idx in range(3):
        y_min = min(axes[axis_idx, col].get_ylim()[0] for col in range(num_joints))
        y_max = max(axes[axis_idx, col].get_ylim()[1] for col in range(num_joints))
        for col in range(num_joints):
            axes[axis_idx, col].set_ylim(y_min, y_max)

    plt.tight_layout()
    save_path = os.path.join(debug_dir, f'{timestamp}_ik_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[process_ik debug] Saved IK comparison to {save_path}")


def save_fk_end_effector_debug(body_model, output, name="fk_end_effector", debug_dir="debug/process_ik",
                               joint_ids=None, joint_names=None):
    import matplotlib.pyplot as plt

    if joint_ids is None:
        joint_ids = [7, 10, 8, 11, 20, 21]
        joint_names = ["L_Ankle", "L_Foot", "R_Ankle", "R_Foot", "L_Wrist", "R_Wrist"]
    if joint_names is None:
        joint_names = [f"Joint_{jid}" for jid in joint_ids]

    fk_j3d = get_joints_from_smpl_params(body_model, output, joint_num=52)["global_joints"]
    data = fk_j3d[0].detach().cpu().numpy()
    axis_names = ['X', 'Y', 'Z']

    num_joints = len(joint_ids)
    fig, axes = plt.subplots(3, num_joints, figsize=(num_joints * 3, 9), squeeze=False)
    fig.suptitle(f'{name}: End-Effector Positions over Time', fontsize=13)

    for col_idx, (jid, jname) in enumerate(zip(joint_ids, joint_names)):
        for axis_idx in range(3):
            ax = axes[axis_idx, col_idx]
            ax.plot(data[:, jid, axis_idx], linewidth=0.8)
            if axis_idx == 0:
                ax.set_title(f'{jname} (j{jid})', fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f'{axis_names[axis_idx]} axis', fontsize=10)
            if axis_idx == 2:
                ax.set_xlabel('Frame', fontsize=9)
            ax.grid(True, alpha=0.3)

    # unify ylim across all joints for each axis
    for axis_idx in range(3):
        y_min = min(axes[axis_idx, col].get_ylim()[0] for col in range(num_joints))
        y_max = max(axes[axis_idx, col].get_ylim()[1] for col in range(num_joints))
        for col in range(num_joints):
            axes[axis_idx, col].set_ylim(y_min, y_max)

    os.makedirs(debug_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(debug_dir, f'{timestamp}_{name}.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[debug] Saved FK end-effector visualization to {save_path}")


@autocast(enabled=False)
def pp_static_joint_from_stationary(
    body_model,
    output,
    static_threshold=0.5,
    replace_ground_y=True,
    smooth_xz=True,
    smooth_y_sigma=1.0,
):
    """Correct world root translation using predicted stationary confidences.

    ``stationary_target[t, j]`` describes whether end effector ``j`` should be
    static over transition ``t -> t + 1``. Four-channel inputs use
    [L_Ankle, L_Foot, R_Ankle, R_Foot]; six-channel inputs additionally use
    [L_Wrist, R_Wrist].
    """
    stationary = output.get("stationary_target")
    if stationary is None:
        stationary = output.get("stationary")
    if stationary is None:
        raise KeyError("postprocess requires stationary_target or stationary")
    if stationary.ndim != 3 or stationary.shape[-1] not in (4, 6):
        raise ValueError(f"expected stationary shape (B,T,4|6), got {tuple(stationary.shape)}")

    fk_j3d = get_joints_from_smpl_params(body_model, output, joint_num=52)["global_joints"]
    joint_ids = [7, 10, 8, 11, 20, 21][:stationary.shape[-1]]
    fk_end_j3d = fk_j3d[:, :, joint_ids]
    fk_end_vel = fk_end_j3d[:, 1:] - fk_end_j3d[:, :-1]

    trans = output["trans"].clone()
    root_vel = trans[:, 1:] - trans[:, :-1]
    static_mask = stationary[:, :-1] > static_threshold
    static_count = static_mask.sum(dim=-1, keepdim=True)
    correction = -(fk_end_vel * static_mask[..., None]).sum(dim=-2)
    correction = correction / static_count.clamp_min(1).to(correction.dtype)
    correction = torch.where(static_count > 0, correction, torch.zeros_like(correction))

    root_vel_new = root_vel + correction
    updated_trans = torch.cumsum(torch.cat([trans[:, :1], root_vel_new], dim=1), dim=1)

    if smooth_xz:
        updated_trans[..., 0] = gaussian_smooth(updated_trans[..., 0], dim=-1)
        updated_trans[..., 2] = gaussian_smooth(updated_trans[..., 2], dim=-1)

    if replace_ground_y:
        updated_fk_j3d = fk_j3d - trans.unsqueeze(-2) + updated_trans.unsqueeze(-2)
        ground_y = updated_fk_j3d[..., 1].flatten(-2).min(dim=-1)[0]
        updated_trans[..., 1] -= ground_y[:, None]

    if smooth_y_sigma > 0:
        updated_trans[..., 1] = gaussian_smooth(
            updated_trans[..., 1], sigma=smooth_y_sigma, dim=-1
        )

    return updated_trans


@autocast(enabled=False)
def pp_static_joint(body_model, output, fps=30, l_thres=1e-2, u_thres=5e-2, replace_ground_y=True):
    # fowrward kinematics to get global end effector joints
    fk_j3d = get_joints_from_smpl_params(body_model, output, joint_num=52)["global_joints"]
    L = fk_j3d.shape[1]
    joint_ids = [7, 10, 8, 11, 20, 21] # [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist]
    fk_end_j3d = fk_j3d.clone()[:, :, joint_ids]

    # calculate end effector velocity of fk and prediction, also the root translation velocity
    fk_end_vel = fk_end_j3d[:, 1:] - fk_end_j3d[:, :-1]
    pred_end_vel = output["end_effector_vel"].clone()[:, :-1] / fps  # (B, L-1, 6, 3)
    trans = output["trans"].clone()  # (B, L, 3)
    root_vel = trans[:, 1:] - trans[:, :-1]

    # determine static and dynamic frames by thresholding predicted end effector velocity
    static_label_ = pred_end_vel.norm(2, dim=-1) < l_thres
    dynamic_label_ = pred_end_vel.norm(2, dim=-1) > u_thres

    # for static frames (< l_thres), zero-out predicted end effector velocity
    # for dynamic frames (> u_thres), ignore the fk_pred_diff
    pred_end_vel = pred_end_vel - (static_label_[..., None] * pred_end_vel)
    fk_pred_diff = pred_end_vel - fk_end_vel
    fk_pred_diff = fk_pred_diff - (dynamic_label_[..., None] * fk_pred_diff)

    non_dynamic_cnt = (~dynamic_label_).float().sum(dim=-1, keepdim=True)
    valid_avg_diff = fk_pred_diff.sum(dim=-2) / torch.clamp(
        non_dynamic_cnt, min=1.0
    )
    total_joints = fk_pred_diff.shape[-2]
    confidence = non_dynamic_cnt / total_joints
    fk_pred_diff = valid_avg_diff * confidence

    # update root translation to make fk close to predicted end effector velocity
    root_vel_new = root_vel + fk_pred_diff
    updated_trans = torch.cumsum(torch.cat([trans[:, :1], root_vel_new], dim=1), dim=1)

    # gaussian smooth x and z of updated root translation
    updated_trans[..., 0] = gaussian_smooth(updated_trans[..., 0], dim=-1)
    updated_trans[..., 2] = gaussian_smooth(updated_trans[..., 2], dim=-1)

    # Put the sequence on the ground by -min(y), this does not consider foot height, for o3d vis
    updated_fk_j3d = fk_j3d - trans.unsqueeze(-2) + updated_trans.unsqueeze(-2)
    if replace_ground_y:
        ground_y = updated_fk_j3d[..., 1].flatten(-2).min(dim=-1)[0]
        updated_trans[..., 1] -= ground_y[:, None]

    return updated_trans


@autocast(enabled=False)
def pp_static_joint_footonly(body_model, output, fps=30, l_thres=1e-2, u_thres=5e-2, replace_ground_y=True):
    # fowrward kinematics to get global end effector joints
    fk_j3d = get_joints_from_smpl_params(body_model, output, joint_num=52)["global_joints"]
    L = fk_j3d.shape[1]
    joint_ids = [7, 10, 8, 11] # [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist]
    fk_end_j3d = fk_j3d.clone()[:, :, joint_ids]

    # calculate end effector velocity of fk and prediction, also the root translation velocity
    fk_end_vel = fk_end_j3d[:, 1:] - fk_end_j3d[:, :-1]
    pred_end_vel = output["end_effector_vel"].clone()[:, :-1, :4] / fps  # (B, L-1, 4, 3)
    trans = output["trans"].clone()  # (B, L, 3)
    root_vel = trans[:, 1:] - trans[:, :-1]

    # determine static and dynamic frames by thresholding predicted end effector velocity
    static_label_ = pred_end_vel.norm(2, dim=-1) < l_thres
    dynamic_label_ = pred_end_vel.norm(2, dim=-1) > u_thres

    # for static frames (< l_thres), zero-out predicted end effector velocity
    # for dynamic frames (> u_thres), ignore the fk_pred_diff
    pred_end_vel = pred_end_vel - (static_label_[..., None] * pred_end_vel)
    fk_pred_diff = pred_end_vel - fk_end_vel
    fk_pred_diff = fk_pred_diff - (dynamic_label_[..., None] * fk_pred_diff)

    non_dynamic_cnt = (~dynamic_label_).float().sum(dim=-1, keepdim=True)
    valid_avg_diff = fk_pred_diff.sum(dim=-2) / torch.clamp(
        non_dynamic_cnt, min=1.0
    )
    total_joints = fk_pred_diff.shape[-2]
    confidence = non_dynamic_cnt / total_joints
    fk_pred_diff = valid_avg_diff * confidence

    # update root translation to make fk close to predicted end effector velocity
    root_vel_new = root_vel + fk_pred_diff
    updated_trans = torch.cumsum(torch.cat([trans[:, :1], root_vel_new], dim=1), dim=1)

    # gaussian smooth x and z of updated root translation
    updated_trans[..., 0] = gaussian_smooth(updated_trans[..., 0], dim=-1)
    updated_trans[..., 2] = gaussian_smooth(updated_trans[..., 2], dim=-1)

    # Put the sequence on the ground by -min(y), this does not consider foot height, for o3d vis
    updated_fk_j3d = fk_j3d - trans.unsqueeze(-2) + updated_trans.unsqueeze(-2)
    if replace_ground_y:
        ground_y = updated_fk_j3d[..., 1].flatten(-2).min(dim=-1)[0]
        updated_trans[..., 1] -= ground_y[:, None]

    return updated_trans


@autocast(enabled=False)
def pp_static_joint_footonly_v2(body_model, output, fps=30, l_thres=1e-2, u_thres=5e-2, replace_ground_y=True):
    # forward kinematics to get global end effector joints
    fk_j3d = get_joints_from_smpl_params(body_model, output, joint_num=52)["global_joints"]
    joint_ids = [7, 10, 8, 11] # [L_Ankle, L_foot, R_Ankle, R_foot]
    fk_end_j3d = fk_j3d.clone()[:, :, joint_ids]

    # calculate end effector velocity of fk and prediction, also the root translation velocity
    fk_end_vel = fk_end_j3d[:, 1:] - fk_end_j3d[:, :-1]
    pred_end_vel = output["end_effector_vel"].clone()[:, :-1, :4] / fps  # (B, L-1, 4, 3)
    trans = output["trans"].clone()  # (B, L, 3)
    root_vel = trans[:, 1:] - trans[:, :-1]

    # determine static and dynamic frames by thresholding predicted end effector velocity
    static_label_ = pred_end_vel.norm(2, dim=-1) < l_thres
    dynamic_label_ = pred_end_vel.norm(2, dim=-1) > u_thres

    # for static frames (< l_thres), zero-out predicted end effector velocity
    # for dynamic frames (> u_thres), ignore the fk_pred_diff
    pred_end_vel = pred_end_vel - (static_label_[..., None] * pred_end_vel)
    fk_pred_diff = pred_end_vel - fk_end_vel
    fk_pred_diff = fk_pred_diff - (dynamic_label_[..., None] * fk_pred_diff)

    # average fk_pred_diff over non-dynamic joints only (dynamic joints are zeroed and excluded)
    non_dynamic_cnt = (~dynamic_label_).float().sum(dim=-1, keepdim=True)
    fk_pred_diff = fk_pred_diff.sum(dim=-2) / torch.clamp(non_dynamic_cnt, min=1.0)

    # update root translation to make fk close to predicted end effector velocity
    root_vel_new = root_vel + fk_pred_diff
    updated_trans = torch.cumsum(torch.cat([trans[:, :1], root_vel_new], dim=1), dim=1)

    # gaussian smooth x and z of updated root translation
    updated_trans[..., 0] = gaussian_smooth(updated_trans[..., 0], dim=-1)
    updated_trans[..., 2] = gaussian_smooth(updated_trans[..., 2], dim=-1)

    # Put the sequence on the ground by -min(y), this does not consider foot height, for o3d vis
    updated_fk_j3d = fk_j3d - trans.unsqueeze(-2) + updated_trans.unsqueeze(-2)
    if replace_ground_y:
        ground_y = updated_fk_j3d[..., 1].flatten(-2).min(dim=-1)[0]
        updated_trans[..., 1] -= ground_y[:, None]

    return updated_trans


@autocast(enabled=False)
def process_ik(body_model, output, fps=30, use_fk_vel=False,
               debug=False, force_align_to_first_frame=True, use_ccd_ik_full=True, optimize_wrist=False):
    # forward kinematics to get global joint positions and joint rotmat
    fk_j3d, local_rotmat, fk_rotmat = get_fkmat_from_smpl_params(body_model, output)

    if optimize_wrist:
        joint_ids = [7, 10, 8, 11, 20, 21] # [L_Ankle, L_Foot, R_Ankle, R_Foot, L_Wrist, R_Wrist]
        joint_names = ["L_Ankle", "L_Foot", "R_Ankle", "R_Foot", "L_Wrist", "R_Wrist"]
    else:
        joint_ids = [7, 10, 8, 11] # [L_Ankle, L_Foot, R_Ankle, R_Foot]
        joint_names = ["L_Ankle", "L_Foot", "R_Ankle", "R_Foot"]

    num_ee = len(joint_ids)

    # determine magnitude of predicted end effector velocity
    if use_fk_vel:
        fk_end_vel = fk_j3d[:, 1:, joint_ids] - fk_j3d[:, :-1, joint_ids]
        end_vel_mag = fk_end_vel.norm(2, dim=-1)
        pred_end_vel = fk_end_vel # for static_conf computation
    else:
        pred_end_vel = output["end_effector_vel"].clone()[:, :-1, :num_ee] / fps
        end_vel_mag = pred_end_vel.norm(2, dim=-1)

    # non-linear mapping from end vel to static confidence by transformed sigmoid
    static_conf = end_vel_to_static_conf(pred_end_vel)

    # save original FK for debug comparison (before any target smoothing)
    if debug:
        fk_j3d_original = fk_j3d.clone()

    post_target_j3d = fk_j3d.clone()
    for i in range(1, fk_j3d.size(1)):
        prev = post_target_j3d[:, i - 1, joint_ids]
        this = fk_j3d[:, i, joint_ids]
        c_prev = static_conf[:, i - 1, :, None].float()
        post_target_j3d[:, i, joint_ids] = prev * c_prev + this * (1.0 - c_prev)

    if force_align_to_first_frame:
        first_frame_l_ankle_y = fk_j3d[:, 0, 7, 1]
        first_frame_l_foot_y = fk_j3d[:, 0, 10, 1]
        curr_l_ankle_y = post_target_j3d[:, :, 7, 1]
        curr_l_foot_y = post_target_j3d[:, :, 10, 1]
        l_ankle_diff = torch.clamp(first_frame_l_ankle_y[:, None] - curr_l_ankle_y, min=0)
        l_foot_diff = torch.clamp(first_frame_l_foot_y[:, None] - curr_l_foot_y, min=0)
        l_shift_y = torch.max(l_ankle_diff, l_foot_diff)
        post_target_j3d[:, :, 7, 1] += l_shift_y # L_Ankle
        post_target_j3d[:, :, 10, 1] += l_shift_y # L_Foot

        first_frame_r_ankle_y = fk_j3d[:, 0, 8, 1]
        first_frame_r_foot_y = fk_j3d[:, 0, 11, 1]
        curr_r_ankle_y = post_target_j3d[:, :, 8, 1]
        curr_r_foot_y = post_target_j3d[:, :, 11, 1]
        r_ankle_diff = torch.clamp(first_frame_r_ankle_y[:, None] - curr_r_ankle_y, min=0)
        r_foot_diff = torch.clamp(first_frame_r_foot_y[:, None] - curr_r_foot_y, min=0)
        r_shift_y = torch.max(r_ankle_diff, r_foot_diff)
        post_target_j3d[:, :, 8, 1] += r_shift_y # R_Ankle
        post_target_j3d[:, :, 11, 1] += r_shift_y # R_Foot

    # ik
    global_rot = matrix.get_rotation(fk_rotmat)
    parents = body_model.parents[:22]
    left_leg_chain = [0, 1, 4, 7, 10]
    right_leg_chain = [0, 2, 5, 8, 11]
    left_hand_chain = [9, 13, 16, 18, 20]
    right_hand_chain = [9, 14, 17, 19, 21]

    def ik(local_mat, target_pos, target_rot, target_ind, chain):
        local_mat = local_mat.clone()
        if use_ccd_ik_full:
            IK_solver = CCDIKFull(
                local_mat,
                parents.tolist(),
                target_ind,
                target_pos=target_pos,
                target_rot=target_rot,
                kinematic_chain=chain,
                iterations=25,
                threshold=0.001,
                activation=Activation.LINEAR,
                pos_weight=1.0,
                rot_weight=0.0,
                debug=debug,
            )

            if debug:
                # --- DEBUG: before IK solve ---
                pre_ik_positions = []
                for ti_idx, ti in enumerate(target_ind):
                    ee_pos = IK_solver.get_global_position(ti)
                    tgt_pos = target_pos[..., ti_idx, :]
                    dist = (ee_pos - tgt_pos).norm(dim=-1).mean().item()
                    pre_ik_positions.append(dist)
                print(f"[DEBUG IK] chain={chain}, target_ind={target_ind}")
                print(f"[DEBUG IK]   BEFORE solve: ee-target distances = {['%.6f' % d for d in pre_ik_positions]}")

            chain_local_mat = IK_solver.solve()

            if debug:
                # --- DEBUG: after IK solve ---
                post_ik_positions = []
                for ti_idx, ti in enumerate(target_ind):
                    ee_pos = IK_solver.get_global_position(ti)
                    tgt_pos = target_pos[..., ti_idx, :]
                    dist = (ee_pos - tgt_pos).norm(dim=-1).mean().item()
                    post_ik_positions.append(dist)
                    print(f"[DEBUG IK]   AFTER  solve: target[{ti_idx}] (chain joint {ti}): "
                          f"ee_pos_mean={ee_pos.mean(dim=tuple(range(ee_pos.dim()-1))).tolist()}, "
                          f"tgt_pos_mean={tgt_pos.mean(dim=tuple(range(tgt_pos.dim()-1))).tolist()}, "
                          f"dist={dist:.6f}")
                print(f"[DEBUG IK]   AFTER  solve: ee-target distances = {['%.6f' % d for d in post_ik_positions]}")
                print(f"[DEBUG IK]   converged={IK_solver.is_converged()}, threshold={IK_solver.threshold}")
                print()

            chain_rotmat = matrix.get_rotation(chain_local_mat)
            local_mat[:, :, chain[1:], :-1, :-1] = chain_rotmat[:, :, 1:]
        else:
            IK_solver = CCD_IK(
                local_mat,
                parents,
                target_ind,
                target_pos,
                target_rot,
                kinematic_chain=chain,
                max_iter=2,
                reg_weight=0.,
            )
            chain_local_mat = IK_solver.solve()
            chain_rotmat = matrix.get_rotation(chain_local_mat)
            local_mat[:, :, chain[1:], :-1, :-1] = chain_rotmat[:, :, 1:]
        return local_mat

    # foot IK (always applied)
    local_rotmat = ik(local_rotmat, post_target_j3d[:, :, [7, 10]], global_rot[:, :, [7, 10]], [3, 4], left_leg_chain)
    local_rotmat = ik(local_rotmat, post_target_j3d[:, :, [8, 11]], global_rot[:, :, [8, 11]], [3, 4], right_leg_chain)

    # wrist IK (only when optimize_wrist is enabled)
    if optimize_wrist:
        local_rotmat = ik(local_rotmat, post_target_j3d[:, :, [20]], global_rot[:, :, [20]], [4], left_hand_chain)
        local_rotmat = ik(local_rotmat, post_target_j3d[:, :, [21]], global_rot[:, :, [21]], [4], right_hand_chain)

    body_pose = rotation_matrix_to_rot6d(matrix.get_rotation(local_rotmat[:, :, 1:]))
    new_pose = torch.cat([output['rot6d'][..., :1, :], body_pose], dim=-2)
    fk_j3d_after_ik, _, _ = get_fkmat_from_smpl_params(body_model, {
        "rot6d": new_pose,
        "trans": output["trans"],
        "shapes": output["shapes"],
    })

    # debug: save comparison of pre-IK and post-IK end-effector trajectories
    if debug:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_dir = os.path.join("debug", "process_ik")
        _save_ik_debug_comparison(
            fk_j3d_original, post_target_j3d, fk_j3d_after_ik,
            joint_ids, joint_names, debug_dir, timestamp,
        )

    return body_pose


@autocast(enabled=False)
def process_ik_mc(body_model, output, fps=30):
    # fowrward kinematics to get global joint positions and joint rotmat
    fk_j3d, local_rotmat, fk_rotmat = get_fkmat_from_smpl_params(body_model, output)
    joint_ids = [7, 10, 8, 11] # [L_Ankle, L_foot, R_Ankle, R_foot]

    # determine magnitude of predicted end effector velocity
    pred_end_vel = output["end_effector_vel"].clone()[:, :-1] / fps  # (B, L-1, 6, 3)
    end_vel_mag = pred_end_vel[:, :, : len(joint_ids)].norm(2, dim=-1)

    post_target_j3d = fk_j3d.clone()
    for i in range(1, fk_j3d.size(1)):
        prev = post_target_j3d[:, i - 1, joint_ids]
        this = fk_j3d[:, i, joint_ids]
        # print("prev: ", prev.shape)
        # print("end_vel_mag[:, i, None]: ", end_vel_mag[:, i][:, :, None].shape)
        # print("this: ", this.shape)
        post_target_j3d[:, i, joint_ids] = prev + end_vel_mag[:, i][:, :, None] * (this - prev)

    # ik
    global_rot = matrix.get_rotation(fk_rotmat)
    parents = body_model.parents[:22]
    left_leg_chain = [0, 1, 4, 7, 10]
    right_leg_chain = [0, 2, 5, 8, 11]
    left_hand_chain = [9, 13, 16, 18, 20]
    right_hand_chain = [9, 14, 17, 19, 21]

    def ik(local_mat, target_pos, target_rot, target_ind, chain):
        local_mat = local_mat.clone()
        IK_solver = CCD_IK(
            local_mat,
            parents,
            target_ind,
            target_pos,
            target_rot,
            kinematic_chain=chain,
            max_iter=2,
        )

        chain_local_mat = IK_solver.solve()
        chain_rotmat = matrix.get_rotation(chain_local_mat)
        local_mat[:, :, chain[1:], :-1, :-1] = chain_rotmat[:, :, 1:]
        return local_mat

    local_rotmat = ik(local_rotmat, post_target_j3d[:, :, [7, 10]], global_rot[:, :, [7, 10]], [3, 4], left_leg_chain)
    local_rotmat = ik(local_rotmat, post_target_j3d[:, :, [8, 11]], global_rot[:, :, [8, 11]], [3, 4], right_leg_chain)
    local_rotmat = ik(local_rotmat, post_target_j3d[:, :, [20]], global_rot[:, :, [20]], [4], left_hand_chain)
    local_rotmat = ik(local_rotmat, post_target_j3d[:, :, [21]], global_rot[:, :, [21]], [4], right_hand_chain)

    body_pose = rotation_matrix_to_rot6d(matrix.get_rotation(local_rotmat[:, :, 1:]))

    return body_pose


def end_vel_to_static_conf(end_vel, l_thres=1e-2, conf_percentile=80):
    end_vel_mag = end_vel.norm(2, dim=-1)

    static_conf = torch.sigmoid(end_vel_mag / l_thres)
    static_conf = 1.0 - static_conf
    static_conf = static_conf * 2
    nonzero_conf_mask = static_conf != 0
    if nonzero_conf_mask.any():
        nonzero_conf = static_conf[nonzero_conf_mask].flatten()
        perc_values = np.percentile(nonzero_conf.cpu().numpy(), conf_percentile)
        static_conf *= 1 / perc_values
        static_conf = torch.clamp(static_conf, max=1.0)

    return static_conf


def _find_segments(mask_np):
    padded = np.concatenate([[0], mask_np.astype(np.int32), [0]])
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts, ends))


def _find_anchor_frame(end_vel_mag_1d, seg_start, seg_end, L):
    vel_start = max(0, seg_start - 1)
    vel_end = min(L - 1, seg_end)
    if vel_start >= vel_end:
        return seg_start
    seg_vel = end_vel_mag_1d[vel_start:vel_end]
    anchor_offset = seg_vel.argmin().item()
    anchor = vel_start + anchor_offset
    return max(seg_start, min(anchor, seg_end - 1))


@autocast(enabled=False)
def process_ik_v2(body_model, output, fps=30, use_fk_vel=False,
                  debug=False, force_align_to_first_frame=True,
                  use_ccd_ik_full=True, transition_frames=3):
    # Step 1: FK + preparation
    fk_j3d, local_rotmat, fk_rotmat = get_fkmat_from_smpl_params(body_model, output)
    B, L = fk_j3d.shape[:2]

    joint_ids = [7, 10, 8, 11] # L_Ankle, L_Foot, R_Ankle, R_Foot
    num_ee = len(joint_ids)

    # end effector velocity
    if use_fk_vel:
        fk_end_vel = fk_j3d[:, 1:, joint_ids] - fk_j3d[:, :-1, joint_ids]
        end_vel_mag = fk_end_vel.norm(2, dim=-1)
        pred_end_vel = fk_end_vel
    else:
        pred_end_vel = output["end_effector_vel"].clone()[:, :-1, :num_ee] / fps
        end_vel_mag = pred_end_vel.norm(2, dim=-1)

    # Step 2: binary static mask
    static_conf = end_vel_to_static_conf(pred_end_vel)
    static_mask = static_conf > 0.5

    if debug:
        for ji, jn in enumerate(["L_Ankle", "L_Foot", "R_Ankle", "R_Foot"]):
            n_static = static_mask[0, :, ji].sum().item()
            print(f"[process_ik_v2] {jn} static frames: {n_static}/{L-1}")

    # Step 3: build frame-level static mask + detect segments + anchor
    post_target_j3d = fk_j3d.clone()

    # Extend transition-level mask (B, L-1, 4) to frame-level (B, L, 4).
    # A frame is static if either adjacent transition is static.
    frame_static = torch.zeros(B, L, num_ee, dtype=torch.bool, device=fk_j3d.device)
    frame_static[:, 0] = static_mask[:, 0]
    frame_static[:, -1] = static_mask[:, -1]
    frame_static[:, 1:-1] = static_mask[:, :-1] | static_mask[:, 1:]

    # Per end-effector: lock each static segment to anchor frame's FK position
    joint_names_v2 = ["L_Ankle", "L_Foot", "R_Ankle", "R_Foot"]
    for b in range(B):
        for ji in range(num_ee):
            mask_1d = frame_static[b, :, ji]
            if not mask_1d.any():
                continue
            segments = _find_segments(mask_1d.cpu().numpy())
            for seg_start, seg_end in segments:
                anchor = _find_anchor_frame(end_vel_mag[b, :, ji], seg_start, seg_end, L)
                anchor_pos = fk_j3d[b, anchor, joint_ids[ji]]
                post_target_j3d[b, seg_start:seg_end, joint_ids[ji]] = anchor_pos
            # Diagnostic: print segment info for this joint
            print(f"      [process_ik_v2] {joint_names_v2[ji]}: {len(segments)} static segments, "
                  f"total static frames={mask_1d.sum().item()}/{L}")

    # Diagnostic: IK target max single-frame displacement after anchoring (before transition smoothing)
    print(f"      [process_ik_v2] === IK target max displacement AFTER anchoring (before transition) ===")
    for ji in range(num_ee):
        jid = joint_ids[ji]
        target_vel = (post_target_j3d[0, 1:, jid] - post_target_j3d[0, :-1, jid]).norm(dim=-1)
        fk_vel = (fk_j3d[0, 1:, jid] - fk_j3d[0, :-1, jid]).norm(dim=-1)
        max_disp_t = target_vel.max().item()
        max_frame_t = target_vel.argmax().item()
        max_disp_fk = fk_vel.max().item()
        max_frame_fk = fk_vel.argmax().item()
        print(f"        {joint_names_v2[ji]}: target max_disp={max_disp_t:.6f} m at frame {max_frame_t}->{max_frame_t+1} "
              f"(FK original max_disp={max_disp_fk:.6f} at frame {max_frame_fk}->{max_frame_fk+1})")

    # Step 4: transition smoothing at dynamic gaps between static segments
    # Instead of applying independent leading/trailing transitions per segment
    # (which collide when the dynamic gap < 2*tf), we find each dynamic gap
    # and smoothly interpolate from the trailing anchor to the leading anchor
    # across the entire gap using smoothstep.
    tf = transition_frames
    if tf > 0:
        for b in range(B):
            for ji in range(num_ee):
                mask_1d = frame_static[b, :, ji]
                jid = joint_ids[ji]
                segments = _find_segments(mask_1d.cpu().numpy())
                if not segments:
                    continue

                # Handle leading dynamic region: [0, first_seg_start)
                first_seg_start = segments[0][0]
                if first_seg_start > 0:
                    n_lead = min(tf, first_seg_start)
                    lead_begin = first_seg_start - n_lead
                    for t in range(lead_begin, first_seg_start):
                        alpha_lin = (t - lead_begin + 1) / (n_lead + 1)
                        alpha = 3.0 * alpha_lin * alpha_lin - 2.0 * alpha_lin * alpha_lin * alpha_lin
                        post_target_j3d[b, t, jid] = (
                            fk_j3d[b, t, jid] * (1.0 - alpha)
                            + post_target_j3d[b, first_seg_start, jid] * alpha
                        )

                # Handle gaps between consecutive segments
                for si in range(len(segments) - 1):
                    prev_end = segments[si][1] # exclusive end of previous segment
                    next_start = segments[si + 1][0] # inclusive start of next segment
                    gap_len = next_start - prev_end
                    if gap_len <= 0:
                        continue
                    # Blend from prev segment's anchor to next segment's anchor
                    # across the entire gap, using smoothstep
                    anchor_prev = post_target_j3d[b, prev_end - 1, jid].clone() # last frame of prev segment
                    anchor_next = post_target_j3d[b, next_start, jid].clone() # first frame of next segment
                    for t in range(prev_end, next_start):
                        alpha_lin = (t - prev_end + 1) / (gap_len + 1)
                        alpha = 3.0 * alpha_lin * alpha_lin - 2.0 * alpha_lin * alpha_lin * alpha_lin
                        post_target_j3d[b, t, jid] = anchor_prev * (1.0 - alpha) + anchor_next * alpha

                # Handle trailing dynamic region: [last_seg_end, L)
                last_seg_end = segments[-1][1]
                if last_seg_end < L:
                    n_trail = min(tf, L - last_seg_end)
                    trail_end = last_seg_end + n_trail
                    for t in range(last_seg_end, trail_end):
                        alpha_lin = (t - last_seg_end + 1) / (n_trail + 1)
                        alpha = 3.0 * alpha_lin * alpha_lin - 2.0 * alpha_lin * alpha_lin * alpha_lin
                        post_target_j3d[b, t, jid] = (
                            post_target_j3d[b, last_seg_end - 1, jid] * (1.0 - alpha)
                            + fk_j3d[b, t, jid] * alpha
                        )

    # Diagnostic: IK target max single-frame displacement after transition smoothing
    print(f"      [process_ik_v2] === IK target max displacement AFTER transition smoothing ===")
    for ji in range(num_ee):
        jid = joint_ids[ji]
        target_vel = (post_target_j3d[0, 1:, jid] - post_target_j3d[0, :-1, jid]).norm(dim=-1)
        max_disp_t = target_vel.max().item()
        max_frame_t = target_vel.argmax().item()
        print(f"        {joint_names_v2[ji]}: target max_disp={max_disp_t:.6f} m ({max_disp_t*1000:.2f} mm) at frame {max_frame_t}->{max_frame_t+1}")

    # Step 5: force_align_to_first_frame (prevent feet below ground)
    if force_align_to_first_frame:
        for side_ankl, side_foot in [(7, 10), (8, 11)]:
            first_ankle_y = fk_j3d[:, 0, side_ankl, 1]
            first_foot_y = fk_j3d[:, 0, side_foot, 1]
            ankle_diff = torch.clamp(first_ankle_y[:, None] - post_target_j3d[:, :, side_ankl, 1], min=0)
            foot_diff = torch.clamp(first_foot_y[:, None] - post_target_j3d[:, :, side_foot, 1], min=0)
            shift_y = torch.max(ankle_diff, foot_diff)

            # Uniform shift within static segments to avoid re-introducing velocity
            ji_ankl = joint_ids.index(side_ankl)
            ji_foot = joint_ids.index(side_foot)
            foot_static = frame_static[:, :, ji_ankl] & frame_static[:, :, ji_foot]
            for b in range(B):
                for ss, se in _find_segments(foot_static[b].cpu().numpy()):
                    seg_shift = shift_y[b, ss:se].max()
                    shift_y[b, ss:se] = seg_shift

            post_target_j3d[:, :, side_ankl, 1] += shift_y
            post_target_j3d[:, :, side_foot, 1] += shift_y

    # Step 6: IK solve
    global_rot = matrix.get_rotation(fk_rotmat)
    parents = body_model.parents[:22]
    left_leg_chain = [0, 1, 4, 7, 10]
    right_leg_chain = [0, 2, 5, 8, 11]

    def ik(local_mat, target_pos, target_rot, target_ind, chain):
        local_mat = local_mat.clone()
        if use_ccd_ik_full:
            IK_solver = CCDIKFull(
                local_mat, parents.tolist(), target_ind,
                target_pos=target_pos, target_rot=target_rot,
                kinematic_chain=chain, iterations=25, threshold=0.001,
                activation=Activation.LINEAR, pos_weight=1.0, rot_weight=0.0,
                debug=False,
            )
            chain_local_mat = IK_solver.solve()
            chain_rotmat = matrix.get_rotation(chain_local_mat)
            local_mat[:, :, chain[1:], :-1, :-1] = chain_rotmat[:, :, 1:]
        else:
            IK_solver = CCD_IK(
                local_mat, parents, target_ind, target_pos, target_rot,
                kinematic_chain=chain, max_iter=2, reg_weight=0.,
            )
            chain_local_mat = IK_solver.solve()
            chain_rotmat = matrix.get_rotation(chain_local_mat)
            local_mat[:, :, chain[1:], :-1, :-1] = chain_rotmat[:, :, 1:]
        return local_mat

    # Left leg IK
    local_rotmat = ik(local_rotmat, post_target_j3d[:, :, [7, 10]],
                      global_rot[:, :, [7, 10]], [3, 4], left_leg_chain)
    # Right leg IK
    local_rotmat = ik(local_rotmat, post_target_j3d[:, :, [8, 11]],
                      global_rot[:, :, [8, 11]], [3, 4], right_leg_chain)

    # Step 7: extract body_pose and return
    body_pose = rotation_matrix_to_rot6d(matrix.get_rotation(local_rotmat[:, :, 1:]))

    if debug:
        new_pose = torch.cat([output['rot6d'][..., :1, :], body_pose], dim=-2)
        fk_j3d_after, _, _ = get_fkmat_from_smpl_params(body_model, {
            "rot6d": new_pose, "trans": output["trans"], "shapes": output["shapes"],
        })
        foot_vel_after = (fk_j3d_after[:, 1:, joint_ids] - fk_j3d_after[:, :-1, joint_ids]).norm(dim=-1)
        if static_mask.any():
            slide = foot_vel_after[static_mask].mean() * fps * 1000
            print(f"[process_ik_v2] static frame foot slide after: {slide:.2f} mm/s")

    return body_pose

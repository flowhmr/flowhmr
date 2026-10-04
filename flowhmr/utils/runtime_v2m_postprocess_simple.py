import torch
import numpy as np
from ..pipeline.lbfgs import get_optimizer, get_chain_closure, get_ankle_chain_closure, get_ground_correction_closure, _build_Rx, _run_lbfgs
from ..core.math.geometry import rot6d_to_rotation_matrix, rotation_matrix_to_rot6d
from .postprocess_utils import _forward_smpl_joints, _forward_smpl_vertices, TrajectoryCorrector, find_foot_contact_interval
def _smooth_motion(output, half_window=1, fps=30):
    rot6d = output["rot6d"]  # (B, T, J, 6)
    trans = output["trans"]  # (B, T, 3)
    B, T, J, _ = rot6d.shape
    assert B == 1

    weights = [0.25, 0.5, 0.25]
    W = 2 * half_window + 1
    assert len(weights) == W

    rot6d_smooth = torch.zeros_like(rot6d)
    for i, w in enumerate(weights):
        offset = i - half_window
        s_src = half_window + offset
        e_src = T - half_window + offset
        rot6d_smooth[:, half_window:T-half_window] += w * rot6d[:, s_src:e_src]
    rot6d_smooth[:, :half_window] = rot6d[:, :half_window]
    rot6d_smooth[:, T-half_window:] = rot6d[:, T-half_window:]
    R = rot6d_to_rotation_matrix(rot6d_smooth)
    output["rot6d"] = rotation_matrix_to_rot6d(R)

    trans_smooth = torch.zeros_like(trans)
    for i, w in enumerate(weights):
        offset = i - half_window
        s_src = half_window + offset
        e_src = T - half_window + offset
        trans_smooth[:, half_window:T-half_window] += w * trans[:, s_src:e_src]
    trans_smooth[:, :half_window] = trans[:, :half_window]
    trans_smooth[:, T-half_window:] = trans[:, T-half_window:]
    output["trans"] = trans_smooth

    return output


def _compute_angular_velocity_deg_per_s(rot6d, fps=30):
    R = rot6d_to_rotation_matrix(rot6d)
    R_rel = R[:, :-1].transpose(-1, -2) @ R[:, 1:]
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    angle_rad = torch.acos(cos_angle)
    return angle_rad * (180.0 / torch.pi) * fps # deg/s

def _smooth_rotation_outliers(output, threshold_deg_per_s=1350.0, max_rounds=3, fps=30):

    rot6d = output["rot6d"]  # (B, T, J, 6)
    B, T, J, _ = rot6d.shape
    assert B == 1, "Only support batch size 1"

    for round_idx in range(max_rounds):
        ang_vel = _compute_angular_velocity_deg_per_s(rot6d, fps=fps)
        outlier_per_frame = (ang_vel > threshold_deg_per_s).any(dim=-1)[0]

        if not outlier_per_frame.any():
            break

        outlier_indices = torch.where(outlier_per_frame)[0]
        if len(outlier_indices) == 0:
            break

        outlier_frames = (outlier_indices + 1).tolist()

        segments = []
        if outlier_frames:
            start = outlier_frames[0]
            end = outlier_frames[0]
            for t in outlier_frames[1:]:
                if t == end + 1:
                    end = t
                else:
                    segments.append((start, end))
                    start = t
                    end = t
            segments.append((start, end))

        for seg_start, seg_end in segments:
            left_boundary = seg_start - 1
            while left_boundary >= 0 and (left_boundary < len(outlier_per_frame) and outlier_per_frame[left_boundary]):
                left_boundary -= 1
            if left_boundary < 0:
                left_boundary = 0

            right_boundary = seg_end + 1
            while right_boundary < T and (right_boundary - 1 < len(outlier_per_frame) and outlier_per_frame[right_boundary - 1]):
                right_boundary += 1
            if right_boundary >= T:
                right_boundary = T - 1


            for t in range(seg_start, seg_end + 1):
                alpha_t = (t - left_boundary) / (right_boundary - left_boundary) if right_boundary > left_boundary else 0.5
                alpha_t = torch.tensor(alpha_t, device=rot6d.device, dtype=rot6d.dtype)

                rot6d_interp = (1.0 - alpha_t) * rot6d[:, left_boundary, :, :] + alpha_t * rot6d[:, right_boundary, :, :]
                R = rot6d_to_rotation_matrix(rot6d_interp)
                rot6d[:, t, :, :] = rotation_matrix_to_rot6d(R)

    output["rot6d"] = rot6d

    return output


JOINT_NAMES = ['Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Index1', 'L_Index2', 'L_Index3', 'L_Middle1', 'L_Middle2', 'L_Middle3', 'L_Pinky1', 'L_Pinky2', 'L_Pinky3', 'L_Ring1', 'L_Ring2', 'L_Ring3', 'L_Thumb1', 'L_Thumb2', 'L_Thumb3', 'R_Index1', 'R_Index2', 'R_Index3', 'R_Middle1', 'R_Middle2', 'R_Middle3', 'R_Pinky1', 'R_Pinky2', 'R_Pinky3', 'R_Ring1', 'R_Ring2', 'R_Ring3', 'R_Thumb1', 'R_Thumb2', 'R_Thumb3']


FOOT_CONTACT_NAMES = ['L_Ankle', 'L_Foot', 'R_Ankle', 'R_Foot']
FOOT_CONTACT_JOINT_IDS = [JOINT_NAMES.index(name) for name in FOOT_CONTACT_NAMES]

CHAINS_DICT = {
    "L_Ankle": ["L_Hip", "L_Knee"],
    "R_Ankle": ["R_Hip", "R_Knee"],
    "L_Foot": ["L_Ankle"],
    "R_Foot": ["R_Ankle"],
}

ANKLE_CHAIN_JOINTS = {
    "L_Ankle": {"hip": JOINT_NAMES.index("L_Hip"), "knee": JOINT_NAMES.index("L_Knee")},
    "R_Ankle": {"hip": JOINT_NAMES.index("R_Hip"), "knee": JOINT_NAMES.index("R_Knee")},
}

# SMPLH parent indices (52 joints)
_SMPLH_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 22,
    23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34,
    35, 21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46,
    47, 21, 49, 50
]

_LEFT_LEG = {"hip": 1, "knee": 4, "ankle": 7, "foot": 10}
_RIGHT_LEG = {"hip": 2, "knee": 5, "ankle": 8, "foot": 11}

def get_target_ankle_positions_by_foot_contacts(body_model, output, foot_contact_intervals, ankle_names=['L_Ankle', 'R_Ankle'], max_offset=0.05):
    keypoints3d, transforms = _forward_smpl_joints(body_model, output)
    keypoints3d = keypoints3d[0]
    T = keypoints3d.shape[0]
    source_ankle_positions = keypoints3d[:, [JOINT_NAMES.index(name) for name in ankle_names], :].clone()
    target_ankle_positions = source_ankle_positions.clone()

    for leg_idx, name in enumerate(ankle_names):
        intervals = foot_contact_intervals[name]
        flag = torch.zeros(T, device=keypoints3d.device, dtype=torch.bool)
        for s, e in intervals:
            mid = (s + e) // 2
            anchor = source_ankle_positions[mid, leg_idx, :]
            target_ankle_positions[s:e, leg_idx, :] = anchor.clone()
            flag[s:e] = True
        margins = flag.float() - 1.0
        corrector = TrajectoryCorrector(margins)
        target_ankle_positions[:, leg_idx, :] = corrector.interpolate(source_ankle_positions[:, leg_idx, :], target_ankle_positions[:, leg_idx, :])
    if max_offset is not None:
        delta = target_ankle_positions - source_ankle_positions
        target_ankle_positions = source_ankle_positions + delta.clamp(-max_offset, max_offset)
    return target_ankle_positions

def get_target_static_foot_positions(body_model, smpl_mesh, output, foot_contact_intervals, foot_names=['L_Foot', 'R_Foot']):
    keypoints3d, transforms = _forward_smpl_joints(body_model, output)
    keypoints3d = keypoints3d[0]
    source_foot_positions = keypoints3d[:, [JOINT_NAMES.index(name) for name in foot_names], :].clone()
    target_foot_positions = source_foot_positions.clone()
    ankle_map = {'L_Foot': 'L_Ankle', 'R_Foot': 'R_Ankle'}
    bone_info = {}
    for fname in foot_names:
        aname = ankle_map[fname]
        ankle_pos = keypoints3d[:, JOINT_NAMES.index(aname), :]
        foot_pos = keypoints3d[:, JOINT_NAMES.index(fname), :]
        bone_info[fname] = {'ankle_pos': ankle_pos, 'bone_len': (foot_pos - ankle_pos).norm(dim=-1)}

    for leg_idx, name in enumerate(foot_names):
        intervals = foot_contact_intervals[name]
        ankle_pos = bone_info[name]['ankle_pos']
        flag = torch.zeros(keypoints3d.shape[0], device=keypoints3d.device, dtype=torch.bool)
        for s, e in intervals:
            mid = (s + e) // 2
            anchor = source_foot_positions[mid, leg_idx, :]
            target_foot_positions[s:e, leg_idx, :] = anchor
            flag[s:e] = True
        margins = flag.float() - 1.0
        corrector = TrajectoryCorrector(margins)
        target_foot_positions[:, leg_idx, :] = corrector.interpolate(source_foot_positions[:, leg_idx, :], target_foot_positions[:, leg_idx, :])

    return target_foot_positions


class PostprocessPipeline:
    def __init__(self, use_lbfgs_for_two_bone=False):
        self.use_lbfgs_for_two_bone = use_lbfgs_for_two_bone
        self.body_model = None
        self.smpl_mesh = None
        self.processors = []
        self.static_vel_thres = 0.01
        self.static_threshold = 0.7

    def move_mesh_to_ground(self, output: dict):
        smpl_vertices = _forward_smpl_vertices(self.smpl_mesh, output)
        mesh_min_y = smpl_vertices[0, ..., 1].min(dim=-1).values.min().item()
        output['trans'][..., 1] -= mesh_min_y
        output['local_joints_positions'][..., 0, 1] -= mesh_min_y
        output['pelvis_world'][..., 1] -= mesh_min_y
        return output

    def move_mesh_to_ground_by_foot_contact(self, output, foot_contact_intervals):
        """foot_contact_intervals: dict[str, list[tuple]]"""
        smpl_vertices = _forward_smpl_vertices(self.smpl_mesh, output)
        vertices_left_leg_min = smpl_vertices[0, :, self.left_leg_vids, 1].min(dim=-1).values
        vertices_right_leg_min = smpl_vertices[0, :, self.right_leg_vids, 1].min(dim=-1).values
        contact_heights = []
        # left: L_Ankle + L_Foot
        for s, e in foot_contact_intervals['L_Ankle'] + foot_contact_intervals['L_Foot']:
            contact_heights.extend(vertices_left_leg_min[s:e].cpu().numpy().tolist())
        # right: R_Ankle + R_Foot
        for s, e in foot_contact_intervals['R_Ankle'] + foot_contact_intervals['R_Foot']:
            contact_heights.extend(vertices_right_leg_min[s:e].cpu().numpy().tolist())
        contact_heights = np.array(contact_heights)
        if contact_heights.size == 0:
            return output
        contact_height_mean = contact_heights.mean().item()
        output['trans'][..., 1] -= contact_height_mean
        output['local_joints_positions'][..., 0, 1] -= contact_height_mean
        output['pelvis_world'][..., 1] -= contact_height_mean
        return output

    def run(self, output: dict, **kwargs) -> dict:

        output = self.move_mesh_to_ground(output)
        output = _smooth_motion(output)
        output = _smooth_rotation_outliers(output)

        foot_contacts = output['foot_contacts']
        assert foot_contacts.shape[0] == 1, "Only support single batch"
        keypoints3d, transforms = _forward_smpl_joints(self.body_model, output)
        keypoints3d = keypoints3d[0]

        foot_contact_intervals = find_foot_contact_interval(
            foot_contacts[0], keypoints3d[:, FOOT_CONTACT_JOINT_IDS],
            joint_names=FOOT_CONTACT_NAMES, threshold=self.static_threshold,
            position_range_threshold={
                'L_Ankle': 0.1, 'R_Ankle': 0.1,
                'L_Foot': 0.05, 'R_Foot': 0.05,
            },
        )
        output = self.move_mesh_to_ground_by_foot_contact(output, foot_contact_intervals)

        total_contact_frames = sum(len(v) for v in foot_contact_intervals.values())
        if total_contact_frames == 0:
            return output

        keypoints3d, transforms = _forward_smpl_joints(self.body_model, output)
        keypoints3d = keypoints3d[0]

        # compute target ankle positions
        target_ankle_positions = get_target_ankle_positions_by_foot_contacts(
            self.body_model, output, foot_contact_intervals,
        )
        ankle_names = ['L_Ankle', 'R_Ankle']
        for idx, joint_name in enumerate(ankle_names):
            jids = ANKLE_CHAIN_JOINTS[joint_name]
            hip_idx, knee_idx = jids["hip"], jids["knee"]
            chain_joint_ids = [hip_idx, knee_idx]
            rot6d_before = output["rot6d"][:, :, chain_joint_ids].clone()
            hip_rot6d_opt = torch.nn.Parameter(output["rot6d"][:, :, hip_idx, :].clone())
            knee_flex_opt = torch.nn.Parameter(torch.zeros(*output["rot6d"].shape[:2], 1, device=output["rot6d"].device))
            optimizer = get_optimizer([hip_rot6d_opt, knee_flex_opt])
            closure = get_ankle_chain_closure(optimizer, self.partial_body_model, {
                "rot6d": output["rot6d"], "shapes": output["shapes"], "trans": output["trans"],
                "hip_rot6d_opt": hip_rot6d_opt, "knee_flex_opt": knee_flex_opt,
                "hip_joint_idx": hip_idx, "knee_joint_idx": knee_idx,
                "knee_rot6d_orig": output["rot6d"][:, :, knee_idx, :].detach().clone(),
            }, [JOINT_NAMES.index(joint_name)], target_ankle_positions[:, idx:idx+1, :])
            _run_lbfgs(optimizer, closure, stage_name=f'{joint_name}')
            output['rot6d'][:, :, hip_idx, :] = hip_rot6d_opt.detach()
            knee_R_orig = rot6d_to_rotation_matrix(output["rot6d"][:, :, knee_idx, :])
            knee_R_new = knee_R_orig @ _build_Rx(knee_flex_opt.detach()[..., 0])
            output['rot6d'][:, :, knee_idx, :] = rotation_matrix_to_rot6d(knee_R_new)
        keypoints3d, transforms = _forward_smpl_joints(self.body_model, output)
        keypoints3d = keypoints3d[0]

        if True: # optimize ankle to stabilize the foot joints
            target_foot_positions = get_target_static_foot_positions(
                self.body_model, self.smpl_mesh, output, foot_contact_intervals,
            )
            foot_names = ['L_Foot', 'R_Foot']
            for idx, joint_name in enumerate(foot_names):
                chain = CHAINS_DICT[joint_name]
                chain_joint_ids = [JOINT_NAMES.index(name) for name in chain]
                rot6d_before = output["rot6d"][:, :, chain_joint_ids].clone()
                rot6d_opt = torch.nn.Parameter(rot6d_before.clone())
                optimizer = get_optimizer([rot6d_opt])
                closure = get_chain_closure(optimizer, self.partial_body_model, {
                    "rot6d": output["rot6d"],
                    "shapes": output["shapes"],
                    "trans": output["trans"],
                    "rot6d_opt": rot6d_opt,
                    "rot6d_indices": chain_joint_ids,
                }, [JOINT_NAMES.index(joint_name)], target_foot_positions[:, idx:idx+1, :])
                _run_lbfgs(optimizer, closure, stage_name=f'fine_{joint_name}')
                output['rot6d'][:, :, chain_joint_ids] = rot6d_opt.detach()
            keypoints3d, transforms = _forward_smpl_joints(self.body_model, output)
            keypoints3d = keypoints3d[0]

        # ==================== Ground Correction ====================
        if True:
            T = output["rot6d"].shape[1]
            device = output["rot6d"].device
            contact_mask_L = torch.zeros(T, dtype=torch.bool, device=device)
            contact_mask_R = torch.zeros(T, dtype=torch.bool, device=device)
            for jname in ['L_Ankle', 'L_Foot']:
                for s, e in foot_contact_intervals.get(jname, []):
                    contact_mask_L[s:e] = True
            for jname in ['R_Ankle', 'R_Foot']:
                for s, e in foot_contact_intervals.get(jname, []):
                    contact_mask_R[s:e] = True

            if contact_mask_L.any() or contact_mask_R.any():
                L_hip_idx = JOINT_NAMES.index('L_Hip')
                R_hip_idx = JOINT_NAMES.index('R_Hip')
                L_knee_idx = JOINT_NAMES.index('L_Knee')
                R_knee_idx = JOINT_NAMES.index('R_Knee')
                L_ankle_idx = JOINT_NAMES.index('L_Ankle')
                R_ankle_idx = JOINT_NAMES.index('R_Ankle')

                with torch.no_grad():
                    vertices_all = _forward_smpl_vertices(self.smpl_mesh, output)[0]
                    all_leg_vids = torch.cat([self.left_leg_vids, self.right_leg_vids])
                    non_leg_mask = torch.ones(vertices_all.shape[1], dtype=torch.bool, device=device)
                    non_leg_mask[all_leg_vids] = False
                    non_leg_min_y = vertices_all[:, non_leg_mask, 1].min(dim=-1).values
                    non_leg_min_y_offset = non_leg_min_y - output['trans'][0, :, 1]  # (T,)

                trans_y_opt = torch.nn.Parameter(output["trans"][:, :, 1:2].clone())
                hip_rot6d_opt_L = torch.nn.Parameter(output["rot6d"][:, :, L_hip_idx, :].clone())
                hip_rot6d_opt_R = torch.nn.Parameter(output["rot6d"][:, :, R_hip_idx, :].clone())
                knee_flex_opt_L = torch.nn.Parameter(torch.zeros(output["rot6d"].shape[0], T, 1, device=device))
                knee_flex_opt_R = torch.nn.Parameter(torch.zeros(output["rot6d"].shape[0], T, 1, device=device))
                ankle_rot6d_opt_L = torch.nn.Parameter(output["rot6d"][:, :, L_ankle_idx, :].clone())
                ankle_rot6d_opt_R = torch.nn.Parameter(output["rot6d"][:, :, R_ankle_idx, :].clone())

                opt_params = [trans_y_opt, hip_rot6d_opt_L, hip_rot6d_opt_R,
                              knee_flex_opt_L, knee_flex_opt_R,
                              ankle_rot6d_opt_L, ankle_rot6d_opt_R]
                optimizer = get_optimizer(opt_params)

                contact_joint_ids = [L_ankle_idx, JOINT_NAMES.index('L_Foot'),
                                     R_ankle_idx, JOINT_NAMES.index('R_Foot')]

                closure = get_ground_correction_closure(optimizer, self.partial_body_model, self.smpl_mesh, {
                    "rot6d": output["rot6d"], "shapes": output["shapes"], "trans": output["trans"],
                    "trans_y_opt": trans_y_opt,
                    "hip_rot6d_opt_L": hip_rot6d_opt_L, "hip_rot6d_opt_R": hip_rot6d_opt_R,
                    "knee_flex_opt_L": knee_flex_opt_L, "knee_flex_opt_R": knee_flex_opt_R,
                    "ankle_rot6d_opt_L": ankle_rot6d_opt_L, "ankle_rot6d_opt_R": ankle_rot6d_opt_R,
                    "hip_idx_L": L_hip_idx, "hip_idx_R": R_hip_idx,
                    "knee_idx_L": L_knee_idx, "knee_idx_R": R_knee_idx,
                    "ankle_idx_L": L_ankle_idx, "ankle_idx_R": R_ankle_idx,
                    "knee_rot6d_orig_L": output["rot6d"][:, :, L_knee_idx, :].detach().clone(),
                    "knee_rot6d_orig_R": output["rot6d"][:, :, R_knee_idx, :].detach().clone(),
                    "contact_mask_L": contact_mask_L, "contact_mask_R": contact_mask_R,
                    "left_foot_vids": self.left_foot_vids, "right_foot_vids": self.right_foot_vids,
                    "contact_joint_ids": contact_joint_ids,
                    "non_leg_min_y_offset": non_leg_min_y_offset,
                })
                _run_lbfgs(optimizer, closure, stage_name='ground_correction')

                trans_y_delta = trans_y_opt.detach()[:, :, 0] - output['trans'][:, :, 1]
                output['trans'][:, :, 1:2] = trans_y_opt.detach()
                output['local_joints_positions'][..., 0, 1] += trans_y_delta
                output['pelvis_world'][..., 1] += trans_y_delta
                output['rot6d'][:, :, L_hip_idx, :] = hip_rot6d_opt_L.detach()
                output['rot6d'][:, :, R_hip_idx, :] = hip_rot6d_opt_R.detach()
                L_knee_R_orig = rot6d_to_rotation_matrix(output["rot6d"][:, :, L_knee_idx, :])
                L_knee_R_new = L_knee_R_orig @ _build_Rx(knee_flex_opt_L.detach()[..., 0])
                output['rot6d'][:, :, L_knee_idx, :] = rotation_matrix_to_rot6d(L_knee_R_new)
                R_knee_R_orig = rot6d_to_rotation_matrix(output["rot6d"][:, :, R_knee_idx, :])
                R_knee_R_new = R_knee_R_orig @ _build_Rx(knee_flex_opt_R.detach()[..., 0])
                output['rot6d'][:, :, R_knee_idx, :] = rotation_matrix_to_rot6d(R_knee_R_new)
                output['rot6d'][:, :, L_ankle_idx, :] = ankle_rot6d_opt_L.detach()
                output['rot6d'][:, :, R_ankle_idx, :] = ankle_rot6d_opt_R.detach()

        return output

    @classmethod
    def default(cls, body_model=None, smpl_mesh=None) -> "PostprocessPipeline":
        from ..core.bodymodels.smpl_skeleton import PartialSMPLSkeleton
        pipeline = cls()
        pipeline.body_model = body_model
        pipeline.smpl_mesh = smpl_mesh
        lower_body_joints = [
            JOINT_NAMES.index(n) for n in
            ('Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'L_Foot', 'R_Foot')
        ]
        pipeline.partial_body_model = PartialSMPLSkeleton(body_model, lower_body_joints)
        pipeline.partial_body_model.to(body_model.j_template.device)
        weights = smpl_mesh.lbs_weights
        max_indices = weights.argmax(dim=-1)
        left_leg_joints = [JOINT_NAMES.index(n) for n in ('L_Hip', 'L_Knee', 'L_Ankle', 'L_Foot')]
        right_leg_joints = [JOINT_NAMES.index(n) for n in ('R_Hip', 'R_Knee', 'R_Ankle', 'R_Foot')]
        left_leg_vids = torch.isin(max_indices, torch.tensor(left_leg_joints, device=max_indices.device)).nonzero(as_tuple=True)[0]
        right_leg_vids = torch.isin(max_indices, torch.tensor(right_leg_joints, device=max_indices.device)).nonzero(as_tuple=True)[0]
        pipeline.left_leg_vids = left_leg_vids
        pipeline.right_leg_vids = right_leg_vids
        left_foot_joints = [JOINT_NAMES.index(n) for n in ('L_Ankle', 'L_Foot')]
        right_foot_joints = [JOINT_NAMES.index(n) for n in ('R_Ankle', 'R_Foot')]
        pipeline.left_foot_vids = torch.isin(max_indices, torch.tensor(left_foot_joints, device=max_indices.device)).nonzero(as_tuple=True)[0]
        pipeline.right_foot_vids = torch.isin(max_indices, torch.tensor(right_foot_joints, device=max_indices.device)).nonzero(as_tuple=True)[0]
        return pipeline
import argparse
import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import torch
import joblib
from scipy.spatial.transform import Rotation as sRot

from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree


class FlowHMRToPHCConverter:
    """Reusable converter for batch jobs; the neutral skeleton is built once."""

    def __init__(self, fps=30, upright_start=False, yup2zup=True, max_frames=None):
        self.fps = int(fps)
        self.upright_start = bool(upright_start)
        self.yup2zup = bool(yup2zup)
        self.max_frames = int(max_frames) if max_frames else None
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.max_frames is not None and self.max_frames < 2:
            raise ValueError(f"max_frames must be >= 2, got {self.max_frames}")

        self.beta = np.zeros(16, dtype=np.float64)
        robot_cfg = {
            "mesh": False, "rel_joint_lm": True, "upright_start": self.upright_start,
            "remove_toe": False, "real_weight": True,
            "real_weight_porpotion_capsules": True, "real_weight_porpotion_boxes": True,
            "replace_feet": True, "masterfoot": False, "big_ankle": True,
            "freeze_hand": False, "box_body": False, "master_range": 50,
            "body_params": {}, "joint_params": {}, "geom_params": {}, "actuator_params": {},
            "model": "smpl",
        }
        robot = LocalRobot(robot_cfg)
        robot.load_from_skeleton(
            betas=torch.from_numpy(self.beta[None]), gender=[0], objs_info=None
        )
        xml_path = f"/tmp/phc_smpl_humanoid_{os.getpid()}.xml"
        robot.write_xml(xml_path)
        try:
            self.skeleton_tree = SkeletonTree.from_mjcf(xml_path)
        finally:
            if os.path.exists(xml_path):
                os.remove(xml_path)
        self.smpl_2_mujoco = [
            SMPL_BONE_ORDER_NAMES.index(name)
            for name in SMPL_MUJOCO_NAMES
            if name in SMPL_BONE_ORDER_NAMES
        ]

    @staticmethod
    def _normalize_input(data):
        poses = np.asarray(data["poses"], dtype=np.float64)
        trans = np.asarray(data["trans"], dtype=np.float64)
        if poses.ndim == 2 and poses.shape[1] % 3 == 0:
            poses = poses.reshape(poses.shape[0], -1, 3)
        if poses.ndim != 3 or poses.shape[1] < 22 or poses.shape[2] != 3:
            raise ValueError(f"invalid poses shape {poses.shape}; expected (T,J,3), J>=22")
        if trans.ndim != 2 or trans.shape[1] != 3:
            raise ValueError(f"invalid trans shape {trans.shape}; expected (T,3)")
        if len(poses) != len(trans) or len(poses) < 2:
            raise ValueError(f"frame mismatch/too short: poses={len(poses)} trans={len(trans)}")
        if not np.isfinite(poses).all() or not np.isfinite(trans).all():
            raise ValueError("poses/trans contains NaN or Inf")
        return poses, trans

    def convert_data(self, data):
        poses, trans = self._normalize_input(data)
        in_fps = int(np.asarray(data.get("mocap_framerate", 30)).reshape(-1)[0])
        if in_fps <= 0:
            raise ValueError(f"invalid mocap_framerate {in_fps}")
        skip = max(1, int(round(in_fps / self.fps)))
        poses = poses[::skip].copy()
        trans = trans[::skip].copy()
        if self.max_frames is not None and len(poses) > self.max_frames:
            poses = poses[: self.max_frames].copy()
            trans = trans[: self.max_frames].copy()
        num_frames = len(poses)

        if self.yup2zup:
            up_rotation = sRot.from_euler("x", 90, degrees=True)
            poses[:, 0] = (
                up_rotation * sRot.from_rotvec(poses[:, 0])
            ).as_rotvec()
            trans = trans @ up_rotation.as_matrix().T

        pose_body = poses[:, :22].reshape(num_frames, 66)
        pose_aa = np.concatenate(
            [pose_body, np.zeros((num_frames, 6), dtype=pose_body.dtype)], axis=-1
        )
        pose_aa_mj = pose_aa.reshape(num_frames, 24, 3)[:, self.smpl_2_mujoco]
        pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(
            num_frames, 24, 4
        )
        root_trans_offset = (
            torch.from_numpy(trans) + self.skeleton_tree.local_translation[0]
        )
        state = SkeletonState.from_rotation_and_root_translation(
            self.skeleton_tree,
            torch.from_numpy(pose_quat),
            root_trans_offset,
            is_local=True,
        )
        if self.upright_start:
            pose_quat_global = (
                sRot.from_quat(state.global_rotation.reshape(-1, 4).numpy())
                * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
            ).as_quat().reshape(num_frames, -1, 4)
            state = SkeletonState.from_rotation_and_root_translation(
                self.skeleton_tree,
                torch.from_numpy(pose_quat_global),
                root_trans_offset,
                is_local=False,
            )

        return {
            "pose_quat_global": state.global_rotation.numpy(),
            "pose_quat": state.local_rotation.numpy(),
            "trans_orig": trans,
            "root_trans_offset": root_trans_offset,
            "beta": self.beta.copy(),
            "gender": "neutral",
            "pose_aa": pose_aa,
            "fps": self.fps,
        }

    def convert_file(self, input_path):
        with np.load(input_path, allow_pickle=True) as source:
            data = {key: source[key] for key in source.files}
        return self.convert_data(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="FlowHMR/SMPL-H npz path")
    ap.add_argument("--output", type=str, required=True, help="output PHC motion pkl path")
    ap.add_argument("--motion_name", type=str, default=None, help="key in the pkl (default: input stem)")
    ap.add_argument("--fps", type=int, default=30, help="target fps (input is downsampled to this)")
    ap.add_argument("--upright_start", action="store_true", default=False,
                    help="apply PHC upright-start rotation (match the policy you eval with)")
    ap.add_argument("--yup2zup", action="store_true", default=True,
                    help="rotate Y-up (FlowHMR) motion to Z-up (PHC/AMASS). On by default.")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="truncate motion to this many frames after fps downsampling (e.g. 360)")
    ap.add_argument("--no-yup2zup", dest="yup2zup", action="store_false",
                    help="disable the Y-up->Z-up rotation (if input is already Z-up).")
    args = ap.parse_args()

    converter = FlowHMRToPHCConverter(
        fps=args.fps, upright_start=args.upright_start, yup2zup=args.yup2zup,
        max_frames=args.max_frames,
    )
    name = args.motion_name or os.path.splitext(os.path.basename(args.input))[0]
    motion = converter.convert_file(args.input)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    joblib.dump({name: motion}, args.output)
    print(f"[convert] {args.input}  ->  {args.output}")
    print(
        f"[convert] motion '{name}': frames={len(motion['pose_aa'])} "
        f"fps={args.fps} upright_start={args.upright_start}"
    )


if __name__ == "__main__":
    main()

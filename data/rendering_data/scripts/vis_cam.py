"""Visualize camera projections on a rendered video (sanity check).

Projects the SMPL-H 3D joints (from the motion .npz) onto the rendered
video frames using the camera .npz produced by the pipeline, and writes an
overlay video via an ffmpeg pipe.

Usage:
    python scripts/vis_cam.py \
        --motion output/task/motion.npz \
        --camera output/task/task_0000_camera.npz \
        --video output/task/task_0000.mp4 \
        --smpl_model_path body_models/smplh/neutral/model.npz \
        --output task_0000_vis.mp4
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import cv2
from decord import VideoReader

from utils.smpl_skeleton import SMPLSkeleton
from utils.geometry import get_c_rootparam
from utils.bbox_utils import project


def draw_joints_on_frame(frame, joints2d, color=(0, 0, 255), radius=7, thickness=5):
    """Draw circular markers for all joints on a frame."""
    frame_with_joints = frame
    for (x, y) in joints2d[:, :2]:
        x_int = int(round(x))
        y_int = int(round(y))
        cv2.circle(
            img=frame_with_joints,
            center=(x_int, y_int),
            radius=radius,
            color=color,
            thickness=thickness
        )
    return frame_with_joints


def vis_cam(motion_file, camera_file, video_file, smpl_model, output_file=None):
    """Project 3D joints to 2D and overlay them on the video frames."""
    motion = np.load(motion_file)
    camera = np.load(camera_file)
    video = VideoReader(video_file)
    poses_np = motion['poses'].reshape(motion['poses'].shape[0], -1, 3)
    if poses_np.shape[1] == 55:
        # SMPL-X -> SMPL-H joint layout: drop jaw and eyes (joints 22-24)
        poses_np = np.concatenate([poses_np[:, :22], poses_np[:, 25:]], axis=1)
    poses_np = poses_np.reshape(poses_np.shape[0], -1)
    poses = torch.from_numpy(poses_np).float()
    # betas may be stored as (B,) or (1, B); the skeleton uses the first 10
    betas = torch.from_numpy(motion['betas']).float().reshape(-1)[:10]

    global_orient_w = torch.from_numpy(poses_np[:, :3]).float()
    transl_w = torch.from_numpy(motion['trans']).float()

    T_w2c = torch.from_numpy(camera['RT']).float()
    offset = smpl_model.get_skeleton(betas)[0]
    global_orient_c, transl_c = get_c_rootparam(
        global_orient_w,
        transl_w,
        T_w2c,
        offset,
    )

    K = torch.from_numpy(camera['K']).float()

    # joints in camera space, for 2D projection
    poses_c = poses.clone()
    poses_c[:, :3] = global_orient_c
    params_c = {
        'shapes': betas[None].repeat(poses.shape[0], 1),
        'poses': poses_c,
        'trans': transl_c,
    }
    joints3d = smpl_model(params_c)['keypoints3d']

    # video writer via ffmpeg pipe
    ffmpeg_process = None
    if output_file:
        frame_sample = video[0].asnumpy()
        height, width = frame_sample.shape[:2]
        fps = video.get_avg_fps()

        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-i', '-',
            '-an',
            '-vcodec', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            output_file
        ]
        ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

        print(f"save video to: {output_file}")
        print(f"resolution: {width}x{height}, fps: {fps:.2f}, frames: {len(video)}")

    for frame_idx, frame in enumerate(video):
        frame = frame.asnumpy()
        # decord outputs RGB, OpenCV needs BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        joints2d = project(joints3d[frame_idx], K[frame_idx])
        frame_bgr = draw_joints_on_frame(frame_bgr, joints2d)

        if ffmpeg_process:
            ffmpeg_process.stdin.write(frame_bgr.tobytes())
            if (frame_idx + 1) % 30 == 0:
                print(f"progress: {frame_idx + 1}/{len(video)}")
        else:
            cv2.imshow('frame', frame_bgr)
            cv2.waitKey(1)

    if ffmpeg_process:
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
        print(f"video saved: {output_file}")
    else:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize camera projection results')
    parser.add_argument('--motion', type=str, required=True, help='SMPL motion data file (.npz)')
    parser.add_argument('--camera', type=str, required=True, help='Camera parameter file (.npz)')
    parser.add_argument('--video', type=str, required=True, help='Input video file')
    parser.add_argument('--smpl_model_path', type=str, default='body_models/smplh/neutral/model.npz',
                        help='Path to the SMPL-H neutral model')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video file path (defaults to <video>_vis.mp4; '
                             'if not writable, shows a live preview instead)')

    args = parser.parse_args()

    if args.output is None:
        video_dir = os.path.dirname(args.video)
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        args.output = os.path.join(video_dir, f"{video_name}_vis.mp4")

    smpl_model = SMPLSkeleton(model_path=args.smpl_model_path, max_shape=10)

    vis_cam(args.motion, args.camera, args.video, smpl_model, args.output)

"""
Batch rendering pipeline for SMPL-H motion data.

Chain per motion:

    SMPL-H/X .npz --> .fbx --> camera trajectory --> Blender render --> bbox annotations

For each motion in a file list, the pipeline:
  1. Converts the motion .npz to a skinned FBX (scripts/smplh2fbx.py) if no
     .fbx exists yet
  2. Randomly selects a camera trajectory type (optionally with hand-held shake)
  3. Randomly selects assets (HDR environment, skin texture, ground)
  4. Generates a camera trajectory by driving Blender with the FBX armature
  5. Renders the video with Blender Cycles (optionally with occluders and
     motion blur)
  6. Projects the SMPL-H joints/vertices into the camera to extract per-frame
     2D bounding boxes and keypoints

Expected input layout:
    --raw_root/<filelist entry>      motion .npz (poses/betas/trans[/mocap_framerate])
    --raw_root/<same stem>.fbx       optional; generated automatically when missing

Output layout:
    --root/<relative dir>/<motion stem>/<task_id>/
        <task_id>.mp4            rendered video
        <task_id>_camera.npz     camera intrinsics/extrinsics per frame
        <task_id>_bbox.npz       per-frame bbox, 3D/2D keypoints
        <task_id>_meta.json      full render configuration (task_config)
        <stem>.fbx / <stem>.npz  copies of the input motion

Usage:
    python render_pipeline.py \
        --raw_root /path/to/motions \
        --root ./output \
        --filelist filelist.txt \
        --smpl_model_path body_models/smplh/neutral/model.npz \
        --blender blender
"""

import os
import glob
import argparse
import sys
import json
import random
import shutil
import time
import logging
import subprocess

import numpy as np
import torch
import cv2
from decord import VideoReader

from utils.smpl_skeleton import SMPLSkeleton, SMPLMesh
from utils.geometry import get_c_rootparam
from utils.bbox_utils import (
    find_longest_segment,
    project,
    get_bbox_valid,
    moving_average_smooth,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def fbx_has_uv(fbx_path):
    """Whether the FBX mesh has a UV layer (the node name is stored as plain text)."""
    with open(fbx_path, 'rb') as f:
        return b'LayerElementUV' in f.read()


def log_blender_output(e, num_lines=20):
    """Log the tail of a failed Blender call (Blender prints its errors to stdout)."""
    output = (e.stdout or '') + (e.stderr or '')
    for line in output.strip().splitlines()[-num_lines:]:
        logging.error(f"         {line}")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch render pipeline")

    # required parameters
    parser.add_argument('--raw_root', type=str, required=True,
                        help='Root directory that contains the motion .npz/.fbx files')
    parser.add_argument('--root', type=str, required=True,
                        help='Output root directory for rendered tasks')
    parser.add_argument('--filelist', type=str, required=True,
                        help='Text file with one relative motion path per line, e.g. "subdir/motion.npz"')
    parser.add_argument('--smpl_model_path', type=str, default='body_models/smplh/neutral/model.npz',
                        help='Path to the SMPL-H neutral model (.pkl/.npz) used for bbox extraction')

    # npz -> fbx conversion (step 1)
    parser.add_argument('--fbx_model_path', type=str, default=None,
                        help='Body model used for npz->fbx conversion (SMPL-H 52 joints or '
                             'SMPL-X 55 joints). Defaults to --smpl_model_path')
    parser.add_argument('--obj_template', type=str, default='body_models/smplh_uv.obj',
                        help='OBJ with SMPL-H UVs, written into the FBX mesh so skin textures map correctly')
    parser.add_argument('--fbx_dir', type=str, default=None,
                        help='Where generated .fbx files are written (default: next to the .npz)')

    # variant control: each motion can be rendered multiple times with
    # different random camera/assets; the variant index becomes the task-id suffix
    parser.add_argument('--num_variants', type=int, default=1,
                        help='Number of render variants per motion')
    parser.add_argument('--variant_start', type=int, default=0,
                        help='First variant index (useful for appending new variants)')

    # debug
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Debug mode: render into --debug_output_dir and draw bbox overlays')
    parser.add_argument('--debug_output_dir', type=str, default='./debug_output',
                        help='Debug output video directory')

    # scripts
    parser.add_argument('--camera_script', type=str,
                        default=os.path.join(SCRIPT_DIR, 'scripts', 'camera_trajectory_generator.py'),
                        help='Camera trajectory generator script')
    parser.add_argument('--render_script', type=str,
                        default=os.path.join(SCRIPT_DIR, 'scripts', 'render_in_blender.py'),
                        help='Render script')
    parser.add_argument('--camera_format', type=str, default='_camera.npz',
                        help='Suffix for camera parameter files')
    parser.add_argument('--bbox_format', type=str, default='_bbox.npz',
                        help='Suffix for bbox annotation files')

    # optional resources
    parser.add_argument('--hdr_dir', type=str, default='./assets/hdr_environments',
                        help='HDR environment file directory (optional)')
    parser.add_argument('--texture_dir', type=str, default='./assets/textures',
                        help='Skin texture directory (optional)')
    parser.add_argument('--ground_dir', type=str, default='./assets/grounds',
                        help='Ground material directory (optional)')
    parser.add_argument('--floor_image_dir', type=str, default='./assets/floors',
                        help='Tiled floor image directory (optional, used when no '
                             'GLTF ground material is available)')

    # render config
    parser.add_argument('--res_x', type=int, default=0,
                        help='Output width (0 = random selection)')
    parser.add_argument('--res_y', type=int, default=0,
                        help='Output height (0 = random selection)')
    parser.add_argument('--num_samples', type=int, default=128,
                        help='Cycles sample count')
    parser.add_argument('--denoising', action='store_true',
                        help='Enable denoising')
    parser.add_argument('--no_gpu', dest='use_gpu', action='store_false',
                        help='Render on CPU instead of GPU')

    # feature switches (each asset type is used only when its directory has files)
    parser.add_argument('--no_hdr', dest='use_hdr', action='store_false',
                        help='Do not use HDR environments')
    parser.add_argument('--no_texture', dest='use_texture', action='store_false',
                        help='Do not use skin textures')
    parser.add_argument('--no_ground', dest='use_ground', action='store_false',
                        help='Do not add a ground (GLTF material or floor image)')

    # occluder switch (occluders are additionally sampled per-task with the
    # probability below; set --occluder_prob 0 to disable entirely)
    parser.add_argument('--occluder_prob', type=float, default=0.2,
                        help='Probability of adding occluders to a task')

    # Blender executable
    parser.add_argument('--blender', type=str, default='blender',
                        help='Blender executable (3.6+ recommended)')

    # random seed
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible asset/camera selection')

    # parse arguments
    if '--' in sys.argv:
        args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    else:
        args = parser.parse_args()

    return args


def resolve_paths(raw_root, file_id, root, fbx_dir=None):
    """Resolve raw input paths and output paths for one file-list entry.

    Each entry is a path relative to --raw_root pointing at a motion .npz
    file. The matching .fbx lives next to it (or under --fbx_dir, mirroring
    the relative path) and is generated automatically when missing.
    Outputs mirror the input directory structure under --root.
    """
    npz_rel = file_id if file_id.endswith('.npz') else file_id + '.npz'
    npz_path = os.path.join(raw_root, npz_rel)
    fbx_rel = os.path.splitext(npz_rel)[0] + '.fbx'
    fbx_path = os.path.join(fbx_dir, fbx_rel) if fbx_dir else os.path.join(raw_root, fbx_rel)
    rel_dir = os.path.dirname(npz_rel)
    stem = os.path.splitext(os.path.basename(npz_rel))[0]
    task_parent = os.path.join(root, rel_dir, stem)
    return fbx_path, npz_path, rel_dir, stem, task_parent


class ResourceManager:
    """Scans and randomly selects assets and render options."""

    def __init__(self, args):
        self.args = args

    def scan_files_list(self):
        """Read the motion file list (one relative path per line)."""
        with open(self.args.filelist, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        files_list = [item.strip() for item in lines if item.strip()]
        print(f'found {len(files_list)} motions in filelist')
        logging.info(f'found {len(files_list)} motions in filelist')
        return files_list

    def scan_hdr_files(self):
        """Scan HDR environment files."""
        if not self.args.hdr_dir or not os.path.exists(self.args.hdr_dir):
            return []
        patterns = ['*.hdr', '*.exr', '**/*.hdr', '**/*.exr']
        hdr_files = []
        for pattern in patterns:
            hdr_files.extend(glob.glob(os.path.join(self.args.hdr_dir, pattern), recursive=True))
        hdr_files = list(set(hdr_files))
        logging.info(f"found {len(hdr_files)} HDR files")
        return hdr_files

    def scan_texture_dirs(self):
        """Scan skin texture directories (a dir counts if it holds images)."""
        if not self.args.texture_dir or not os.path.exists(self.args.texture_dir):
            return []
        texture_dirs = []
        for root, dirs, files in os.walk(self.args.texture_dir):
            has_texture = any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files)
            if has_texture:
                texture_dirs.append(root)
        logging.info(f"found {len(texture_dirs)} texture directories")
        return texture_dirs

    def scan_ground_materials(self):
        """Scan ground materials (a GLTF file plus a Textures/ subdirectory)."""
        if not self.args.ground_dir or not os.path.exists(self.args.ground_dir):
            return []
        ground_materials = []
        for root, dirs, files in os.walk(self.args.ground_dir):
            gltf_files = [f for f in files if f.endswith('.gltf')]
            texture_subdir = os.path.join(root, 'Textures')
            if gltf_files and os.path.exists(texture_subdir):
                texture_files = os.listdir(texture_subdir)
                has_base_color = any(f.endswith('B.png') or f.endswith('b.png') for f in texture_files)
                if has_base_color:
                    ground_materials.append({
                        'gltf_path': os.path.join(root, gltf_files[0]),
                        'texture_dir': texture_subdir,
                        'name': os.path.basename(root)
                    })
        logging.info(f"found {len(ground_materials)} ground materials")
        return ground_materials

    def scan_floor_images(self):
        """Scan tiled floor images (single image repeated on the ground plane)."""
        if not self.args.floor_image_dir or not os.path.exists(self.args.floor_image_dir):
            return []
        floor_images = []
        for root, dirs, files in os.walk(self.args.floor_image_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    floor_images.append({
                        'image_path': os.path.join(root, f),
                        'name': os.path.splitext(f)[0],
                    })
        logging.info(f"found {len(floor_images)} floor images")
        return floor_images

    def select_motion_blur(self):
        """Enable motion blur with 80% probability."""
        return random.random() < 0.8

    def select_resolution(self, res_x, res_y):
        """Randomly sample an output resolution (or use the given one)."""
        if res_x == 0:
            rand = random.random()
            if rand < 0.7:      # landscape
                return 1920, 1080
            elif rand < 0.85:   # portrait
                return 1080, 1920
            else:               # square
                return 1080, 1080
        else:
            return res_x, res_y

    def select_camera_type(self):
        """Sample a camera trajectory type from a weighted distribution."""
        camera_type_weights = {
            "static": 0.40,
            "static_track": 0.15,
            "front_track": 0.05,
            "side_track": 0.05,
            "random_track": 0.15,
            "orbit": 0.05,
            "arc": 0.05,
            "push": 0.05,
            "pull": 0.05,
        }
        movement_types = list(camera_type_weights.keys())
        weights = list(camera_type_weights.values())
        camera_type = np.random.choice(movement_types, p=weights)
        # 20% of moving cameras additionally get hand-held shake
        shake_flag = random.random() < 0.2 and camera_type != "static"
        return camera_type, shake_flag

    def select_occluder_config(self):
        """Sample an occluder configuration (disabled with some probability)."""
        occluder_config = {
            'enabled': random.random() < self.args.occluder_prob,
            'min_count': 0,
            'max_count': 3,
            'shapes': ['cube', 'sphere', 'cylinder', 'plane', 'cone', 'torus'],
            'modes': ['foreground', 'edge', 'corner', 'random'],
            'materials': ['solid', 'transparent', 'metallic', 'glass'],
            'size_range': [0.15, 0.6],
            'transparency_range': [0.3, 0.8],
            'animate': random.random() < 0.3,
            'depth_range': [0.25, 0.75],
        }
        return occluder_config

    def random_select(self, items):
        """Randomly select one asset (None if the list is empty)."""
        if not items:
            return None
        return random.choice(items)


class RenderPipeline:
    """Main render pipeline."""

    def __init__(self, args):
        self.args = args
        self.resource_mgr = ResourceManager(args)
        self.results = []
        assert args.num_variants >= 1, "--num_variants must be >= 1"

        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)

        # SMPL-H models for bbox extraction
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'loading SMPL model from {args.smpl_model_path} ...')
        self.smpl_model = SMPLSkeleton(model_path=args.smpl_model_path, max_shape=10).to(self.device)
        self.smpl_mesh = SMPLMesh(model_path=args.smpl_model_path).to(self.device)

        # logging: file + console
        filelist_stem = os.path.splitext(os.path.basename(args.filelist))[0]
        log_dir = os.path.join('./logs', filelist_stem)
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'render_v{args.variant_start}_{args.variant_start + args.num_variants}.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w', encoding='utf-8'),
                logging.StreamHandler(sys.stdout),
            ],
        )

        if args.debug:
            os.makedirs(args.debug_output_dir, exist_ok=True)

        # npz -> fbx converter, created on first use so the FBX SDK is only
        # required when some motion actually lacks an .fbx
        self._fbx_converter = None
        # set in run(): textures are used, so the FBX must carry UVs
        self.need_uv = False

    def ensure_fbx(self, npz_path, fbx_path):
        """Make sure an FBX exists for the motion, converting if needed."""
        if os.path.exists(fbx_path):
            if not self.need_uv or fbx_has_uv(fbx_path):
                return True
            logging.info(f"    - existing fbx has no UVs, regenerating: {fbx_path}")
        if self._fbx_converter is None:
            sys.path.insert(0, os.path.join(SCRIPT_DIR, 'scripts'))
            from smplh2fbx import SMPLH2FBX
            model_path = self.args.fbx_model_path or self.args.smpl_model_path
            self._fbx_converter = SMPLH2FBX(model_path, obj_template=self.args.obj_template)
        logging.info(f"    - converting npz -> fbx: {fbx_path}")
        try:
            return self._fbx_converter.convert(npz_path, fbx_path)
        except Exception as e:
            logging.error(f"      [error] npz -> fbx conversion failed: {e}")
            return False

    def generate_camera_trajectory(self, fbx_path, task_id, camera_type, shake_flag, res_x, res_y, output_task):
        """Generate a camera trajectory by calling Blender with the FBX armature.

        Returns the camera .npz path, or None on failure.
        """
        camera_output = os.path.join(output_task, f"{task_id}{self.args.camera_format}")

        cmd = [
            self.args.blender,
            '--background',
            '--python', self.args.camera_script,
            '--',
            '--fbx', fbx_path,
            '--output', camera_output,
            '--movement_type', camera_type,
            '--res_x', str(res_x),
            '--res_y', str(res_y)
        ]
        if shake_flag:
            cmd.append('--shake_flag')

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            if os.path.exists(camera_output):
                return camera_output
            logging.warning("      [warn] camera file was not generated")
            return None
        except subprocess.CalledProcessError as e:
            logging.error("      [error] camera generation failed")
            log_blender_output(e)
            return None

    def render_task(self, task_config, output_task):
        """Render a single task by calling the Blender render script."""
        task_config_file = os.path.join(output_task, f"{task_config['task_id']}_meta.json")
        with open(task_config_file, "w", encoding="utf-8") as f:
            json.dump(task_config, f, indent=2)

        try:
            cmd = [
                self.args.blender,
                '--background',
                '--python', self.args.render_script,
                '--',
                '--task_config', task_config_file
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return {
                'status': 'success',
                'task_id': task_config['task_id'],
                'output': os.path.join(output_task, f"{task_config['task_id']}.mp4")
            }
        except subprocess.CalledProcessError as e:
            logging.error("      [error] render failed")
            log_blender_output(e)
            return {
                'status': 'failed',
                'task_id': task_config['task_id'],
                'error': str(e)
            }

    def extract_bbox(self, motion, camera_data, w, h):
        """Project SMPL-H joints/vertices into the camera to get per-frame bboxes.

        A frame is considered invalid when the projected body center falls out
        of the image or fewer than 8 body joints are visible. Invalid frames
        get bbox [-1, -1, -1, -1]. The longest valid segment is smoothed with
        a sliding-window average to avoid jitter.
        """
        poses_np = motion['poses'].reshape(motion['poses'].shape[0], -1, 3)
        if poses_np.shape[1] == 55:
            # SMPL-X -> SMPL-H joint layout: drop jaw and eyes (joints 22-24)
            poses_np = np.concatenate([poses_np[:, :22], poses_np[:, 25:]], axis=1)
        poses_np = poses_np.reshape(poses_np.shape[0], -1)
        poses = torch.from_numpy(poses_np).float().to(self.device)
        betas = torch.from_numpy(np.asarray(motion['betas']).reshape(1, -1)).repeat(poses.shape[0], 1).float()[:, :10].to(self.device)

        global_orient_w = torch.from_numpy(poses_np[:, :3]).float().to(self.device)
        transl_w = torch.from_numpy(motion['trans']).float().to(self.device)

        T_w2c = torch.from_numpy(camera_data['RT']).float().to(self.device)
        offset = self.smpl_model.get_skeleton(betas[0])[0]
        global_orient_c, transl_c = get_c_rootparam(
            global_orient_w,
            transl_w,
            T_w2c,
            offset,
        )
        K = torch.from_numpy(camera_data['K']).float().to(self.device)

        # joints in world coordinates (original motion)
        params_w = {
            'shapes': betas,
            'poses': poses,
            'trans': transl_w,
        }
        joints_w = self.smpl_model(params_w)['keypoints3d']

        # joints in camera coordinates (for 2D projection)
        poses_c = poses.clone()
        poses_c[:, :3] = global_orient_c
        params_c = {
            'shapes': betas,
            'poses': poses_c,
            'trans': transl_c,
        }
        joints_c = self.smpl_model(params_c)['keypoints3d']
        vertices_c = self.smpl_mesh(params_c)['vertices']

        bbox_list = []
        joints2d_list = []
        valid_frames_log = []
        for frame_idx in range(joints_c.shape[0]):
            joints2d = project(joints_c[frame_idx], K[frame_idx])
            vertices2d = project(vertices_c[frame_idx], K[frame_idx])
            center, scale, _, bbox_xyxy = get_bbox_valid(vertices2d[:], rescale=1.05, img_width=w, img_height=h)
            is_valid = True
            if center[0] < 0 or center[1] < 0 or scale <= 0:
                is_valid = False
                bbox_xyxy = [-1, -1, -1, -1]
            else:
                _, _, num_vis_joints, _ = get_bbox_valid(joints2d[:22], rescale=1.05, img_width=w, img_height=h)
                if num_vis_joints < 8:
                    is_valid = False
                    bbox_xyxy = [-1, -1, -1, -1]
            valid_frames_log.append(is_valid)
            bbox_list.append(bbox_xyxy)
            joints2d_list.append(joints2d)
        raw_bboxes_tensor = torch.tensor(np.array(bbox_list, dtype=np.float32))

        best_start, best_end, max_len = find_longest_segment(valid_frames_log)
        if max_len == 0:
            return (raw_bboxes_tensor.numpy(), joints_c.cpu().numpy(),
                    joints_w.cpu().numpy(), np.array(joints2d_list)[..., :2], 0, 0)

        start_frame, end_frame = best_start, best_end
        valid_bbox_segment = raw_bboxes_tensor[start_frame: end_frame + 1].clone()
        # smooth the longest valid segment (two passes)
        smoothed_segment = moving_average_smooth(valid_bbox_segment, window_size=5, dim=0)
        smoothed_segment = moving_average_smooth(smoothed_segment, window_size=5, dim=0)
        raw_bboxes_tensor[start_frame: end_frame + 1] = smoothed_segment

        # clamp to image bounds
        raw_bboxes_tensor[:, 0] = torch.clamp(raw_bboxes_tensor[:, 0], min=0, max=w)
        raw_bboxes_tensor[:, 2] = torch.clamp(raw_bboxes_tensor[:, 2], min=0, max=w)
        raw_bboxes_tensor[:, 1] = torch.clamp(raw_bboxes_tensor[:, 1], min=0, max=h)
        raw_bboxes_tensor[:, 3] = torch.clamp(raw_bboxes_tensor[:, 3], min=0, max=h)

        bbox = raw_bboxes_tensor.numpy()
        joints2d = np.array(joints2d_list)
        return bbox, joints_c.cpu().numpy(), joints_w.cpu().numpy(), joints2d[..., :2], start_frame, end_frame

    def visbbox(self, bbox, output_dir, video_path):
        """Debug helper: draw per-frame bboxes onto the rendered video."""
        video = VideoReader(video_path)
        frame_sample = video[0].asnumpy()
        height, width = frame_sample.shape[:2]
        fps = video.get_avg_fps()
        output_file = os.path.join(output_dir, 'bbox.mp4')
        if len(video) != bbox.shape[0]:
            print('video length != bbox length, skipping visualization')
            return

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

        for frame_idx in range(len(video)):
            frame = video[frame_idx].asnumpy()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            bbox_frame = bbox[frame_idx]
            pt1 = (int(bbox_frame[0]), int(bbox_frame[1]))
            pt2 = (int(bbox_frame[2]), int(bbox_frame[3]))
            frame_bbox = cv2.rectangle(frame_bgr, pt1, pt2, (0, 0, 255), 2)
            ffmpeg_process.stdin.write(frame_bbox.tobytes())

        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
        print(f"saved bbox visualization: {output_file}")

    def run(self):
        """Run the complete pipeline."""
        logging.info("\n" + "=" * 60)
        logging.info("render pipeline started")
        logging.info("=" * 60)

        logging.info("\nscanning resources...")
        motion_files = self.resource_mgr.scan_files_list()
        hdr_files = self.resource_mgr.scan_hdr_files()
        texture_dirs = self.resource_mgr.scan_texture_dirs()
        ground_materials = self.resource_mgr.scan_ground_materials()
        floor_images = self.resource_mgr.scan_floor_images()

        if not motion_files:
            logging.error("error: no motions found in filelist")
            return

        logging.info("\nresource statistics:")
        logging.info(f"  motions: {len(motion_files)}")
        logging.info(f"  HDR files: {len(hdr_files)}")
        logging.info(f"  texture directories: {len(texture_dirs)}")
        logging.info(f"  ground materials: {len(ground_materials)}")
        logging.info(f"  floor images: {len(floor_images)}")
        if self.args.use_texture:
            if not texture_dirs:
                logging.warning(f"  [warn] no skin textures found in {self.args.texture_dir!r}; "
                                "the body is rendered with a plain material")
            elif not (self.args.obj_template and os.path.isfile(self.args.obj_template)):
                logging.error(f"error: skin textures need the UV template {self.args.obj_template!r} "
                              "(see README: Installation). Pass --obj_template, or --no_texture "
                              "to render without textures.")
                sys.exit(1)
            else:
                self.need_uv = True

        total_renders = len(motion_files) * self.args.num_variants
        logging.info(f"\ntotal renders: {total_renders}")

        logging.info("\n" + "=" * 60)
        logging.info("start processing (npz -> fbx -> camera -> render -> bbox)")
        logging.info("=" * 60 + "\n")

        success_count = 0
        task_counter = 0
        sum_frames = 0
        sum_times = 0
        num_file = 0
        missing_files_count = 0
        for render_idx in range(self.args.variant_start, self.args.variant_start + self.args.num_variants):
            for file_idx in motion_files:
                num_file += 1
                raw_fbx_path, raw_npz_path, rel_dir, stem, task_parent = \
                    resolve_paths(self.args.raw_root, file_idx, self.args.root, self.args.fbx_dir)

                if self.args.debug:
                    task_parent = os.path.join(self.args.debug_output_dir, stem)
                if not os.path.exists(raw_npz_path):
                    logging.warning(f'{raw_npz_path} does not exist')
                    missing_files_count += 1
                    continue

                logging.info(f"\n[{num_file}/{len(motion_files)}] processing motion: {stem}")
                start_time = time.time()
                task_counter += 1
                task_id = f"{stem}_{render_idx:04d}"
                logging.info(f"  [{task_counter}/{total_renders}] task: {task_id}")
                output_dir_path = os.path.join(task_parent, task_id)
                output_path = os.path.join(output_dir_path, f"{task_id}.mp4")
                if os.path.exists(output_path):
                    logging.info(f'{output_path} already exists, skipping')
                    continue

                # 1. npz -> fbx (skipped when the fbx already exists)
                if not self.ensure_fbx(raw_npz_path, raw_fbx_path):
                    missing_files_count += 1
                    continue

                # 2. randomly select camera type
                camera_type, shake_flag = self.resource_mgr.select_camera_type()
                logging.info(f"    - camera type: {camera_type}")
                logging.info(f"    - shake flag: {shake_flag}")

                # 3. randomly select resources
                hdr_file = None
                texture_dir = None
                ground_material = None
                floor_image = None

                if self.args.use_hdr and hdr_files:
                    hdr_file = self.resource_mgr.random_select(hdr_files)
                    logging.info(f"    - HDR: {os.path.basename(hdr_file)}")

                if self.args.use_texture and texture_dirs:
                    texture_dir = self.resource_mgr.random_select(texture_dirs)
                    logging.info(f"    - texture: {os.path.basename(texture_dir)}")

                if self.args.use_ground and ground_materials:
                    ground_material = self.resource_mgr.random_select(ground_materials)
                    logging.info(f"    - ground: {ground_material['name']}")
                elif self.args.use_ground and floor_images:
                    # fallback: tiled floor image when no GLTF material is available
                    floor_image = dict(self.resource_mgr.random_select(floor_images))
                    floor_image['tile_scale'] = random.uniform(2.0, 6.0)
                    floor_image['scale'] = 50.0
                    logging.info(f"    - floor image: {floor_image['name']} "
                                 f"(tile_scale={floor_image['tile_scale']:.1f})")

                # 4. generate camera trajectory
                logging.info("    - generating camera...")
                res_x, res_y = self.resource_mgr.select_resolution(self.args.res_x, self.args.res_y)
                os.makedirs(output_dir_path, exist_ok=True)
                # copy motion files into the task directory (keeps each task self-contained)
                shutil.copy2(raw_fbx_path, output_dir_path)
                shutil.copy2(raw_npz_path, output_dir_path)
                fbx_path = os.path.join(output_dir_path, stem + '.fbx')
                npz_path = os.path.join(output_dir_path, stem + '.npz')
                motion_npz = np.load(npz_path)
                frame_num = motion_npz['poses'].shape[0]
                sum_frames += frame_num
                camera_params_path = self.generate_camera_trajectory(
                    fbx_path,
                    task_id,
                    camera_type,
                    shake_flag,
                    res_x,
                    res_y,
                    output_dir_path
                )
                if not camera_params_path:
                    logging.error("    - camera generation failed")
                    continue
                camera_data = np.load(camera_params_path)
                focal_length = camera_data['focal_length'].item()

                # 5. render
                occluder_config = self.resource_mgr.select_occluder_config()

                task_config = {
                    'task_id': task_id,
                    'fbx_path': fbx_path,
                    'hdr_file': hdr_file,
                    'texture_dir': texture_dir,
                    'ground_material': ground_material,
                    'ground_image': floor_image,
                    'output_path': output_dir_path,
                    'camera_type': camera_type,
                    'focal_length': focal_length,
                    'camera_params_path': camera_params_path,
                    'config': {
                        'res_x': res_x,
                        'res_y': res_y,
                        'num_samples': self.args.num_samples,
                        'use_denoising': self.args.denoising,
                        'use_gpu': self.args.use_gpu,
                        'use_motion_blur': self.resource_mgr.select_motion_blur(),
                    },
                    'occluder_config': occluder_config,
                    'length': frame_num
                }

                logging.info("    - rendering...")
                result = self.render_task(task_config, output_dir_path)
                self.results.append(result)

                if result['status'] != 'success':
                    # no video -> no annotations; the task is retried on the next run
                    logging.error("      [error] rendering failed\n")
                    continue
                success_count += 1
                logging.info("      [ok] rendering completed\n")

                # 6. extract bbox annotations
                bbox, joints_c, joints_w, joints_2d, start, end = \
                    self.extract_bbox(motion_npz, camera_data, res_x, res_y)
                start_end = np.array([start, end])
                bbox_output_path = os.path.join(output_dir_path, f"{task_id}{self.args.bbox_format}")
                np.savez(bbox_output_path, bbox=bbox, joints_world=joints_w,
                         joints_camera=joints_c, kp2d=joints_2d, start_end=start_end)

                end_time = time.time()
                render_time = end_time - start_time
                sum_times += render_time
                logging.info(f'>>> frames: {frame_num}, render time: {render_time:.1f}s, fps: {frame_num / render_time:.2f}')
                logging.info(f'>>> total frames so far: {sum_frames}')
                logging.info(f'>>> total time so far: {sum_times:.1f}s')

                if self.args.debug:
                    video_path = os.path.join(output_dir_path, f"{task_id}.mp4")
                    self.visbbox(bbox, output_dir_path, video_path)

        logging.info("\n" + "=" * 60)
        logging.info("rendering completed!")
        logging.info("=" * 60)
        logging.info(f"total tasks: {len(self.results)}")
        logging.info(f"success: {success_count}")
        logging.info(f"failed: {len(self.results) - success_count}")
        logging.info(f"missing input files: {missing_files_count}")
        logging.info("=" * 60)
        logging.info(f"time taken: {sum_times:.1f} seconds")
        logging.info(f'total frames: {sum_frames}')


if __name__ == "__main__":
    args = parse_args()
    pipeline = RenderPipeline(args)
    pipeline.run()

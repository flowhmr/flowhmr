"""Camera trajectory generator (runs inside Blender).

Given an FBX with an animated armature (e.g. exported from SMPL-X/H via
scripts/smplh2fbx.py), this script samples per-frame bone positions from the
armature and synthesizes a random camera trajectory around the performing
character. The trajectory and the corresponding OpenCV-style camera
parameters (K, RT) are saved to an .npz file.

Camera model
------------
- View direction is biased toward the front of the character (front 55%,
  side 35%, back 10% by default).
- Focal length is sampled from weighted ranges (18-32mm 25%, 32-60mm 50%,
  60-100mm 15%, 100-200mm 10%).
- Camera height is sampled from 4 categories: low angle, eye level (60%),
  high angle, aerial.
- 9 trajectory types: static, static_track, front_track, side_track,
  random_track, orbit, arc, push, pull. Tracking trajectories follow a
  smoothed body center; orbit/arc rotation amplitude is limited by clip
  length.
- Optional hand-held shake: low-frequency 1D Perlin noise added to all 6
  pose channels independently.

Output .npz keys
----------------
K               (F, 3, 3)  intrinsics (OpenCV convention)
RT              (F, 3, 4)  world-to-camera extrinsics (OpenCV convention,
                          with a z-up -> y-up conversion applied)
location        (F, 3)     camera location in Blender world space
rotation        (F, 3)     camera rotation (XYZ euler)
focal_length    ()         focal length in mm
sensor_width    ()         sensor width in mm
sensor_height   ()         sensor height in mm
movement_type   ()         trajectory type string

Usage:
    blender --background --python scripts/camera_trajectory_generator.py -- \
        --fbx motion.fbx --output motion_camera.npz \
        --movement_type orbit --res_x 1920 --res_y 1080 [--shake_flag]
"""

import os
import sys
import math
import random
import argparse
from math import pi, sin, cos, radians, fabs

import numpy as np
from mathutils import Vector, Euler

import bpy


class PerlinNoise1D:
    """Simple 1D Perlin noise, used to simulate hand-held camera shake."""

    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.p = list(range(256))
        random.shuffle(self.p)
        self.p += self.p  # doubled to handle overflow

    def noise(self, x):
        # start of the unit interval
        X = int(math.floor(x)) & 255
        # relative offset
        x -= math.floor(x)
        # easing function for smoother transitions
        u = x * x * x * (x * (x * 6 - 15) + 10)
        # linear interpolation between the two gradients
        return self.lerp(u, self.grad(self.p[X], x), self.grad(self.p[X + 1], x - 1))

    def lerp(self, t, a, b):
        return a + t * (b - a)

    def grad(self, hash, x):
        # 1D gradient: map the lowest bit of the hash to +1 or -1
        return x if (hash & 1) == 0 else -x


class CameraTrajectoryGenerator:
    """Random camera trajectory generation driven by an FBX armature."""

    def __init__(self, config=None, shake_flag=False):
        self.joints_3d = None
        self.smpl_frame_count = 0

        self.pelvis_bone_idx = None
        self.left_shoulder_idx = None
        self.right_shoulder_idx = None
        self.left_hip_idx = None
        self.right_hip_idx = None
        self.shake_flag = shake_flag

        self.cfg = {
            # distance range
            "min_distance": 1.5,
            "max_distance": 30.0,

            # height range
            "ground_level": 0.0,
            "min_camera_height": 0.3,
            "max_camera_height": 6.0,

            # height categories
            "low_angle_range": [0.3, 1.2],
            "eye_level_range": [1.2, 1.8],
            "high_angle_range": [1.8, 3.5],
            "aerial_range": [3.5, 5.0],

            # height category weights
            "height_weights": {
                "low_angle": 0.15,
                "eye_level": 0.60,
                "high_angle": 0.20,
                "aerial": 0.05
            },

            # fov and margin
            "sensor_width": 36.0,
            "sensor_height": 24.0,
            "fov_margin": 1.0,

            # focal length ranges and weights
            "focal_length_ranges": {
                (18, 32): 0.25,
                (32, 60): 0.50,
                (60, 100): 0.15,
                (100, 200): 0.10,
            },

            # view angle preference (front/side/back of the character)
            "reduce_back_view": True,
            "front_view_bias": 0.55,
            "side_view_bias": 0.35,
            "back_view_bias": 0.10,
            "angle_bias_strength": 1.0,

            # padding (m) added to the joint bounds before computing distances
            "margin_xy": 0.1,
            "margin_z": 0.1,

            # fallback movement type weights (used when movement_type='random')
            "movement_types": {
                "static": 0.30,
                "static_track": 0.20,
                "front_track": 0.05,
                "side_track": 0.05,
                "random_track": 0.20,
                "orbit": 0.05,
                "arc": 0.05,
                "push": 0.05,
                "pull": 0.05,
            },

            # camera shake amplitude (enabled per task by --shake_flag)
            "shake_intensity": 0.02,           # meters
            "shake_rotation_intensity": 0.5,   # degrees

            # use global (whole-clip) bounds for distance computation
            "use_global_bounds": True,
        }

        if config:
            self.cfg.update(config)

    def load_from_blender_armature(self, armature_obj):
        """Extract per-frame world-space bone positions from an armature."""
        try:
            if armature_obj is None or armature_obj.type != 'ARMATURE':
                raise ValueError("provided object is not an armature")

            # animation frame range
            if armature_obj.animation_data and armature_obj.animation_data.action:
                action = armature_obj.animation_data.action
                frame_start = int(action.frame_range[0])
                frame_end = int(action.frame_range[1])
                total_frames = frame_end - frame_start + 1
            else:
                # no animation: only use the first frame
                frame_start = 1
                frame_end = 1
                total_frames = 1

            # keyword matching rules for key joints (by priority)
            bone_patterns = {
                'pelvis': ['pelvis', 'hips', 'spine_00', 'root'],
                'left_shoulder': ['left_shoulder', 'shoulder.l', 'shoulderleft', 'l_shoulder', 'upperarm_l'],
                'right_shoulder': ['right_shoulder', 'shoulder.r', 'shoulderright', 'r_shoulder', 'upperarm_r'],
                'left_hip': ['left_hip', 'hip.l', 'hipleft', 'l_hip', 'thigh_l', 'upleg_l'],
                'right_hip': ['right_hip', 'hip.r', 'hipright', 'r_hip', 'thigh_r', 'upleg_r'],
            }

            # traverse all bones and identify the key joints
            for i, bone in enumerate(armature_obj.pose.bones):
                bone_name_lower = bone.name.lower()
                if self.pelvis_bone_idx is None:
                    for pattern in bone_patterns['pelvis']:
                        if pattern in bone_name_lower:
                            self.pelvis_bone_idx = i
                            break
                if self.left_shoulder_idx is None:
                    for pattern in bone_patterns['left_shoulder']:
                        if pattern in bone_name_lower:
                            self.left_shoulder_idx = i
                            break
                if self.right_shoulder_idx is None:
                    for pattern in bone_patterns['right_shoulder']:
                        if pattern in bone_name_lower:
                            self.right_shoulder_idx = i
                            break
                if self.left_hip_idx is None:
                    for pattern in bone_patterns['left_hip']:
                        if pattern in bone_name_lower:
                            self.left_hip_idx = i
                            break
                if self.right_hip_idx is None:
                    for pattern in bone_patterns['right_hip']:
                        if pattern in bone_name_lower:
                            self.right_hip_idx = i
                            break

            # fallback if pelvis was not found: skip the root, use the second bone
            if self.pelvis_bone_idx is None:
                self.pelvis_bone_idx = 1

            # sample all bone positions per frame
            joints_list = []
            for frame in range(frame_start, frame_end + 1):
                bpy.context.scene.frame_set(frame)
                frame_joints = []
                for bone in armature_obj.pose.bones:
                    world_matrix = armature_obj.matrix_world @ bone.matrix
                    world_pos = world_matrix.to_translation()
                    frame_joints.append([world_pos.x, world_pos.y, world_pos.z])
                joints_list.append(frame_joints)

            self.joints_3d = np.array(joints_list)  # (T, J, 3)
            self.smpl_frame_count = total_frames
            print(f"loaded {self.smpl_frame_count} frames, {self.joints_3d.shape[1]} bones")
            return True

        except Exception:
            import traceback
            traceback.print_exc()
            return False

    def get_human_center(self, frame_idx):
        """Body center = pelvis position of the given frame."""
        if self.joints_3d is None:
            return Vector((0, 0, 0))

        frame_idx = max(0, min(frame_idx, self.smpl_frame_count - 1))
        joints_frame = self.joints_3d[frame_idx]
        pelvis_idx = getattr(self, 'pelvis_bone_idx', 0)
        pelvis = joints_frame[pelvis_idx]
        return Vector(pelvis)

    def calculate_segment_bounds(self, start_frame, end_frame):
        """Compute the 3D bounds of the body over a frame segment.

        The bounds are padded by `margin_xy` horizontally and `margin_z`
        vertically.
        """
        if self.joints_3d is None:
            return None

        start_frame = max(0, start_frame)
        end_frame = min(end_frame, self.smpl_frame_count - 1)
        if start_frame > end_frame:
            return None

        joints_subset = self.joints_3d[start_frame:end_frame + 1]
        all_joints = joints_subset.reshape(-1, 3)

        margin = np.array([self.cfg["margin_xy"], self.cfg["margin_xy"], self.cfg["margin_z"]])
        min_pos_final = all_joints.min(axis=0) - margin
        max_pos_final = all_joints.max(axis=0) + margin
        center_final = (min_pos_final + max_pos_final) / 2
        size_final = max_pos_final - min_pos_final

        frame_centers = []
        for frame_idx in range(start_frame, end_frame + 1):
            frame_center = np.mean(self.joints_3d[frame_idx], axis=0)
            frame_centers.append(Vector(frame_center))

        return {
            'min_pos': Vector(min_pos_final),
            'max_pos': Vector(max_pos_final),
            'center': Vector(center_final),
            'size': Vector(size_final),
            'motion_range_xy': np.max(size_final[:2]),
            'motion_range_z': size_final[2],
            'frame_centers': frame_centers,
            'start_frame': start_frame,
            'end_frame': end_frame
        }

    def calculate_global_bounds(self, total_frames=None):
        """Compute whole-clip bounds, with a resolution-dependent margin.

        Landscape renders (width > height) have a tighter vertical FOV, so
        the Z extent is scaled up; portrait/square renders have a tighter
        horizontal FOV, so XY is scaled up. This keeps the body inside the
        frame regardless of aspect ratio.
        """
        if self.joints_3d is None:
            return None

        if total_frames is None:
            total_frames = self.smpl_frame_count
        actual_frames = min(total_frames, self.smpl_frame_count)

        bounds = self.calculate_segment_bounds(0, actual_frames - 1)
        if bounds is None:
            return None

        frame_centers = bounds['frame_centers']
        if not frame_centers:
            trajectory_center = bounds['center']
            trajectory_range = 0.0
        else:
            trajectory_min = np.min([c[:] for c in frame_centers], axis=0)
            trajectory_max = np.max([c[:] for c in frame_centers], axis=0)
            trajectory_center = (trajectory_min + trajectory_max) / 2
            trajectory_range = np.max(trajectory_max - trajectory_min)

        scene = bpy.context.scene
        global_margin_xy, global_margin_z = 1.0, 1.0
        if scene.render.resolution_x > scene.render.resolution_y:
            # landscape: vertical FOV is the tight constraint
            global_margin_z = 1.1
        else:
            # portrait / square: horizontal FOV is the tight constraint
            global_margin_xy = 1.1

        scaling_vector = Vector((global_margin_xy, global_margin_xy, global_margin_z))
        global_size = bounds['size'] * scaling_vector

        center = bounds['center']
        half_global_size = global_size / 2.0
        global_min_pos = center - half_global_size
        global_max_pos = center + half_global_size

        global_motion_range_xy = bounds['motion_range_xy'] * global_margin_xy
        global_motion_range_z = bounds['motion_range_z'] * global_margin_z

        return {
            'global_min_pos': global_min_pos,
            'global_max_pos': global_max_pos,
            'global_center': center,
            'global_size': global_size,
            'global_motion_range_xy': global_motion_range_xy,
            'global_motion_range_z': global_motion_range_z,
            'trajectory_center': Vector(trajectory_center),
            'trajectory_range': trajectory_range,
            'frame_centers': frame_centers,
            'actual_frames': actual_frames
        }

    def calculate_distance_for_global_bounds(self, global_bounds, focal_length, angle, height):
        """Camera distance so that the whole global bounds fit in the frame.

        Accounts for the effective sensor area cropped by the render aspect
        ratio and for the vertical offset between camera and body center.
        """
        scene = bpy.context.scene
        render_width = scene.render.resolution_x * scene.render.resolution_percentage / 100
        render_height = scene.render.resolution_y * scene.render.resolution_percentage / 100

        if render_height == 0:
            return self.cfg.get("min_distance", 1.0)

        render_aspect_ratio = render_width / render_height
        physical_sensor_width = self.cfg.get("sensor_width", 36)
        physical_sensor_height = self.cfg.get("sensor_height", 24)

        if physical_sensor_height == 0:
            return self.cfg.get("min_distance", 1.0)

        sensor_aspect_ratio = physical_sensor_width / physical_sensor_height

        # effective sensor area actually used by the render aspect ratio
        if render_aspect_ratio > sensor_aspect_ratio:
            effective_sensor_width = physical_sensor_width
            effective_sensor_height = physical_sensor_width / render_aspect_ratio
        else:
            effective_sensor_height = physical_sensor_height
            effective_sensor_width = physical_sensor_height * render_aspect_ratio

        center = global_bounds['global_center']
        margin = self.cfg.get("fov_margin", 1.0)

        global_size = global_bounds['global_size']
        # horizontal extent of the body projected onto the view direction
        projected_horizontal_size = fabs(global_size.x * sin(angle)) + fabs(global_size.y * cos(angle))
        target_horizontal_size = projected_horizontal_size * margin
        dist_h = (target_horizontal_size * focal_length) / effective_sensor_width

        # vertical extent: body size plus the camera-height offset
        delta_z = fabs(height - center.z)
        effective_vertical_size = global_size.z + delta_z
        target_vertical_size = effective_vertical_size * margin
        dist_v = (target_vertical_size * focal_length) / effective_sensor_height

        required_distance = max(dist_h, dist_v)

        min_dist = self.cfg.get("min_distance", 1.0)
        max_dist = self.cfg.get("max_distance", 100.0)
        return max(min_dist, min(required_distance, max_dist))

    def detect_human_orientation(self, frame_idx):
        """Facing direction of the character (radians, XY plane)."""
        if self.joints_3d is None:
            return 0.0
        frame_idx = max(0, min(frame_idx, self.smpl_frame_count - 1))
        return self._detect_orientation_from_joints(frame_idx)

    def _detect_orientation_from_joints(self, frame_idx):
        """Facing direction from the shoulder/hip cross product normal."""
        assert frame_idx < len(self.joints_3d), f'frame_idx: {frame_idx} is out of range'
        joints = self.joints_3d[frame_idx]

        required_indices = [
            self.left_shoulder_idx, self.right_shoulder_idx,
            self.left_hip_idx, self.right_hip_idx,
            self.pelvis_bone_idx
        ]
        if any(idx is None for idx in required_indices):
            return 0.0
        if len(joints) <= max(required_indices):
            return 0.0

        left_shoulder = joints[self.left_shoulder_idx]
        right_shoulder = joints[self.right_shoulder_idx]
        left_hip = joints[self.left_hip_idx]
        right_hip = joints[self.right_hip_idx]

        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2

        hip_vec = right_hip - left_hip          # hip lateral direction
        torso_vec = shoulder_center - hip_center  # torso longitudinal direction

        # cross product normal points to the character's front (or back)
        normal = np.cross(hip_vec, torso_vec)

        if np.linalg.norm(normal[:2]) > 0.01:
            forward_2d = normal[:2]
            forward_2d = forward_2d / np.linalg.norm(forward_2d)
            return np.arctan2(forward_2d[1], forward_2d[0])
        else:
            return 0.0

    def _normalize_angle(self, angle):
        """Normalize an angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def _moving_average_smooth(self, data, window_size):
        """Moving average smoothing along axis 0."""
        if len(data) < window_size:
            return data

        smoothed = np.copy(data)
        half_window = window_size // 2
        for i in range(len(data)):
            start = max(0, i - half_window)
            end = min(len(data), i + half_window + 1)
            smoothed[i] = np.mean(data[start:end], axis=0)
        return smoothed

    def get_smoothed_human_centers(self, total_frames, window_size=5):
        """Smoothed body center positions (pelvis trajectory)."""
        centers = []
        for frame in range(total_frames):
            center = self.get_human_center(frame)
            centers.append(np.array([center.x, center.y, center.z]))

        centers = np.array(centers)
        if len(centers) >= window_size:
            smoothed = self._moving_average_smooth(centers, window_size)
        else:
            smoothed = centers

        return [Vector(c) for c in smoothed]

    def generate_biased_angle(self):
        """Sample a horizontal view angle biased toward the character's front.

        front: +/-45 deg around the facing direction, side: 45-90 deg,
        back: 90-270 deg. The bias is mixed with a uniform distribution
        controlled by `angle_bias_strength`.
        """
        if not self.cfg["reduce_back_view"]:
            return random.uniform(0, 2 * pi)

        human_orientation = self.detect_human_orientation(0)
        human_forward_angle = human_orientation + pi

        front_angles = [(human_forward_angle - pi / 4, human_forward_angle + pi / 4)]
        side_angles = [
            (human_forward_angle + pi / 4, human_forward_angle + pi / 2),
            (human_forward_angle - pi / 2, human_forward_angle - pi / 4)
        ]
        back_angles = [(human_forward_angle + pi - pi / 2, human_forward_angle + pi + pi / 2)]

        front_prob = self.cfg["front_view_bias"]
        side_prob = self.cfg["side_view_bias"]
        back_prob = self.cfg["back_view_bias"]
        bias_strength = self.cfg["angle_bias_strength"]

        # normalize, then mix with a uniform distribution
        total = front_prob + side_prob + back_prob
        front_prob /= total
        side_prob /= total
        back_prob /= total

        uniform_prob = 1.0 / 3.0
        adjusted_front = (1 - bias_strength) * uniform_prob + bias_strength * front_prob
        adjusted_side = (1 - bias_strength) * uniform_prob + bias_strength * side_prob

        rand = random.random()
        if rand < adjusted_front:
            angle_range = random.choice(front_angles)
        elif rand < adjusted_front + adjusted_side:
            angle_range = random.choice(side_angles)
        else:
            angle_range = random.choice(back_angles)

        angle = random.uniform(angle_range[0], angle_range[1])
        while angle < 0:
            angle += 2 * pi
        while angle >= 2 * pi:
            angle -= 2 * pi

        return angle

    def select_height_category(self):
        categories = list(self.cfg["height_weights"].keys())
        weights = list(self.cfg["height_weights"].values())
        return np.random.choice(categories, p=weights)

    def generate_height(self, category=None, is_track=False):
        """Sample an absolute camera height (meters above ground)."""
        if category is None:
            category = self.select_height_category()

        if is_track:
            # tracking shots follow the body, so heights are relative offsets
            height_ranges = {
                "low_angle": [-0.5, 0],
                "eye_level": [0, 1],
                "high_angle": [1, 2],
                "aerial": [2, 2.5]
            }
        else:
            height_ranges = {
                "low_angle": self.cfg["low_angle_range"],
                "eye_level": self.cfg["eye_level_range"],
                "high_angle": self.cfg["high_angle_range"],
                "aerial": self.cfg["aerial_range"]
            }

        height_range = height_ranges.get(category, self.cfg["eye_level_range"])
        height = random.uniform(*height_range)

        min_h = self.cfg["ground_level"] + self.cfg["min_camera_height"]
        max_h = self.cfg["ground_level"] + self.cfg["max_camera_height"]
        height = max(min_h, min(height, max_h))

        return height, category

    def select_focal_length(self):
        """Sample a focal length (mm) from the weighted ranges."""
        focal_lengths_ranges = list(self.cfg["focal_length_ranges"].keys())
        probabilities = list(self.cfg["focal_length_ranges"].values())
        range_idx = np.random.choice(len(focal_lengths_ranges), p=probabilities)
        selected_range = focal_lengths_ranges[range_idx]
        return np.random.uniform(selected_range[0], selected_range[1])

    def add_camera_shake(self, frames):
        """Add low-frequency Perlin-noise shake to position and rotation.

        Each of the 6 channels (x, y, z, pitch, yaw, roll) gets an
        independent noise instance so the shake pattern never repeats.
        """
        # frequency: lower values give a slower (low-frequency) sway
        freq_loc = 0.08
        freq_rot = 0.12

        # amplitude
        amp_loc = self.cfg["shake_intensity"]
        amp_rot = radians(self.cfg["shake_rotation_intensity"])

        noises = [PerlinNoise1D(seed=random.randint(0, 999)) for _ in range(6)]

        shaken_frames = []
        for i, frame_data in enumerate(frames):
            # sample the noise along the frame axis, scaled by frequency
            t_loc = i * freq_loc
            t_rot = i * freq_rot

            # positional offsets (X, Y, Z)
            off_loc = [noises[j].noise(t_loc) * amp_loc for j in range(3)]

            # rotational offsets (pitch, yaw, roll)
            off_rot = [noises[j + 3].noise(t_rot) * amp_rot for j in range(3)]

            loc = list(frame_data['location'])
            rot = list(frame_data['rotation'])

            shaken_frames.append({
                'frame': frame_data['frame'],
                'location': tuple(loc[j] + off_loc[j] for j in range(3)),
                'rotation': tuple(rot[j] + off_rot[j] for j in range(3))
            })

        return shaken_frames

    def calculate_required_distance(self, bounds, focal_length=50):
        """Camera distance so that the given bounds fit in the frame."""
        scene = bpy.context.scene
        render = scene.render
        width = render.resolution_x * render.resolution_percentage / 100
        height = render.resolution_y * render.resolution_percentage / 100

        if width == 0 or height == 0:
            return self.cfg.get("min_distance", 1.0)

        size = bounds['size']
        margin = self.cfg.get("fov_margin", 1.0)

        target_horizontal_size = max(size.x, size.y) * margin
        target_vertical_size = size.z * margin

        sensor_width = self.cfg["sensor_width"]
        sensor_height = self.cfg["sensor_height"]
        render_aspect_ratio = width / height
        sensor_aspect_ratio = sensor_width / sensor_height

        # effective sensor area actually used by the render aspect ratio
        if render_aspect_ratio > sensor_aspect_ratio:
            effective_sensor_width = sensor_width
            effective_sensor_height = sensor_width / render_aspect_ratio
        else:
            effective_sensor_height = sensor_height
            effective_sensor_width = sensor_height * render_aspect_ratio

        dist_h = (target_horizontal_size * focal_length) / effective_sensor_width
        dist_v = (target_vertical_size * focal_length) / effective_sensor_height

        required_distance = max(dist_h, dist_v)

        min_dist = self.cfg.get("min_distance", 1.0)
        max_dist = self.cfg.get("max_distance", 100.0)

        return max(min_dist, min(required_distance, max_dist))

    def generate_random_trajectory(self, movement_type=None):
        """Generate a camera trajectory of the requested type."""
        if self.joints_3d is None:
            raise ValueError("load motion data first")

        total_frames = self.smpl_frame_count

        if movement_type == 'random':
            movement_types = list(self.cfg["movement_types"].keys())
            weights = list(self.cfg["movement_types"].values())
            movement_type = np.random.choice(movement_types, p=weights)

        focal_length = self.select_focal_length()

        print(f"generate camera trajectory: {movement_type}, "
              f"focal length: {focal_length:.1f}mm, total frames: {total_frames}")

        bounds = self.calculate_segment_bounds(0, total_frames - 1)

        if movement_type == "static":
            frames = self._generate_static_trajectory(bounds, total_frames, focal_length)
        elif movement_type == "static_track":
            frames = self._generate_static_track_trajectory(bounds, total_frames, focal_length)
        elif movement_type == "random_track":
            frames = self._generate_random_track_trajectory(bounds, total_frames, focal_length)
        elif movement_type == "front_track":
            frames = self._generate_front_track_trajectory(bounds, total_frames, focal_length)
        elif movement_type == "side_track":
            frames = self._generate_side_track_trajectory(bounds, total_frames, focal_length)
        elif movement_type == "orbit":
            frames = self._generate_orbit_trajectory(bounds, total_frames, focal_length)
        elif movement_type == "arc":
            frames = self._generate_arc_trajectory(bounds, total_frames, focal_length)
        elif movement_type == "push":
            frames = self._generate_push_trajectory(bounds, total_frames, focal_length)
        elif movement_type == "pull":
            frames = self._generate_pull_trajectory(bounds, total_frames, focal_length)
        else:
            frames = self._generate_static_trajectory(bounds, total_frames, focal_length)

        if self.shake_flag:
            frames = self.add_camera_shake(frames)

        return {
            'movement_type': movement_type,
            'focal_length': focal_length,
            'total_frames': total_frames,
            'frames': frames
        }

    def _max_frame_distance(self, bounds, total_frames, focal_length, multiplier_range):
        """Max over frames of the required distance, so the body always fits.

        Tracking/orbiting trajectories keep a fixed distance from the body;
        this distance must cover the frame with the largest extent.
        """
        max_distance = 0
        for frame in range(total_frames):
            frame_bounds = self.calculate_segment_bounds(frame, frame)
            if frame_bounds is not None:
                multiplier = random.uniform(*multiplier_range)
                frame_distance = self.calculate_required_distance(frame_bounds, focal_length) * multiplier
                max_distance = max(max_distance, frame_distance)

        if max_distance == 0:
            return self.calculate_required_distance(bounds, focal_length) * 1.1
        return max_distance

    def _generate_static_trajectory(self, bounds, total_frames, focal_length):
        """Fully static camera (fixed position and rotation)."""
        angle = self.generate_biased_angle()
        height, _ = self.generate_height()

        if self.cfg.get("use_global_bounds", True):
            self.global_bounds = self.calculate_global_bounds(total_frames)
        else:
            self.global_bounds = None

        if self.global_bounds:
            center = self.global_bounds['global_center']
            distance = self.calculate_distance_for_global_bounds(self.global_bounds, focal_length, angle, height)
        else:
            center = bounds['center']
            distance = self.calculate_required_distance(bounds, focal_length)

        x = center.x + distance * cos(angle)
        y = center.y + distance * sin(angle)
        z = height

        camera_pos = Vector((x, y, z))

        # look at the clip center
        direction = center - camera_pos
        if direction.length > 0:
            rotation = direction.to_track_quat('-Z', 'Y').to_euler()
        else:
            rotation = Euler((0, 0, 0))

        frames = []
        for frame in range(total_frames):
            frames.append({
                'frame': frame + 1,
                'location': tuple(camera_pos),
                'rotation': tuple(rotation)
            })
        return frames

    def _generate_static_track_trajectory(self, bounds, total_frames, focal_length):
        """Static position, but rotation follows the (smoothed) body center."""
        distance = self._max_frame_distance(bounds, total_frames, focal_length, (1.25, 1.5))
        print(f"distance: {distance:.2f}")

        center = bounds['center']
        angle = self.generate_biased_angle()
        height, _ = self.generate_height()

        x = center.x + distance * cos(angle)
        y = center.y + distance * sin(angle)
        z = height
        camera_pos = Vector((x, y, z))

        smoothed_centers = self.get_smoothed_human_centers(total_frames, window_size=5)

        frames = []
        for frame in range(total_frames):
            human_center = smoothed_centers[frame]
            direction = human_center - camera_pos
            if direction.length > 0:
                rotation = direction.to_track_quat('-Z', 'Y').to_euler()
            else:
                rotation = Euler((0, 0, 0))
            frames.append({
                'frame': frame + 1,
                'location': tuple(camera_pos),
                'rotation': tuple(rotation)
            })
        return frames

    def _generate_random_track_trajectory(self, bounds, total_frames, focal_length):
        """Tracking camera on a random side, following the body center."""
        angle = self.generate_biased_angle()
        offset_direction = Vector((cos(angle), sin(angle), 0)).normalized()

        # fixed height offset relative to the body center
        height_offset, _ = self.generate_height()
        height_offset = random.uniform(0, height_offset)

        distance = self._max_frame_distance(bounds, total_frames, focal_length, (1.15, 1.4))

        frames = []
        for frame in range(total_frames):
            current_human_center = self.get_human_center(frame)

            camera_pos = current_human_center + offset_direction * distance
            camera_pos.z = current_human_center.z + height_offset

            # always look at the current body center
            direction = current_human_center - camera_pos
            if direction.length > 0:
                rotation = direction.to_track_quat('-Z', 'Y').to_euler()
            else:
                rotation = Euler((0, 0, 0))
            frames.append({
                'frame': frame + 1,
                'location': tuple(camera_pos),
                'rotation': tuple(rotation)
            })
        return frames

    def _generate_front_track_trajectory(self, bounds, total_frames, focal_length):
        """Tracking camera in front of the character (facing direction + 180 deg)."""
        height_offset, _ = self.generate_height(is_track=True)
        distance = self._max_frame_distance(bounds, total_frames, focal_length, (1.15, 1.4))

        smoothed_centers = self.get_smoothed_human_centers(total_frames, window_size=5)
        orientation = self.detect_human_orientation(0)

        frames = []
        for frame in range(total_frames):
            human_center = smoothed_centers[frame]

            camera_angle = orientation + pi
            camera_direction = Vector((cos(camera_angle), sin(camera_angle), 0)).normalized()

            camera_pos = human_center + camera_direction * distance
            camera_pos.z = human_center.z + height_offset

            direction = human_center - camera_pos
            if direction.length > 0:
                rotation = direction.to_track_quat('-Z', 'Y').to_euler()
            else:
                rotation = Euler((0, 0, 0))
            frames.append({
                'frame': frame + 1,
                'location': tuple(camera_pos),
                'rotation': tuple(rotation)
            })
        return frames

    def _generate_side_track_trajectory(self, bounds, total_frames, focal_length):
        """Tracking camera to the side of the character (90-120 deg off front)."""
        height_offset, _ = self.generate_height(is_track=True)
        # left or right side, kept consistent over the whole clip
        side_angle = random.uniform(pi * 2 / 3, pi / 2)
        side_offset = random.choice([-side_angle, side_angle])

        distance = self._max_frame_distance(bounds, total_frames, focal_length, (1.15, 1.4))

        smoothed_centers = self.get_smoothed_human_centers(total_frames, window_size=5)
        orientation = self.detect_human_orientation(0)

        frames = []
        for frame in range(total_frames):
            human_center = smoothed_centers[frame]

            camera_angle = orientation + side_offset
            camera_direction = Vector((cos(camera_angle), sin(camera_angle), 0)).normalized()

            camera_pos = human_center + camera_direction * distance
            camera_pos.z = human_center.z + height_offset

            direction = human_center - camera_pos
            if direction.length > 0:
                rotation = direction.to_track_quat('-Z', 'Y').to_euler()
            else:
                rotation = Euler((0, 0, 0))
            frames.append({
                'frame': frame + 1,
                'location': tuple(camera_pos),
                'rotation': tuple(rotation)
            })
        return frames

    def _generate_height_curve(self, base_height, total_frames):
        """Height-over-time curve for orbit/arc/push/pull trajectories.

        Distribution: constant 50%, ramp up/down 30%, Perlin noise 20%.

        Clip-length constraints (variation speed):
        - < 60 frames (2 s): forced constant height
        - 60-180 frames: delta limited to [0.3, 0.5] m
        - 180-360 frames: delta [0.3, 0.8] m
        - > 360 frames: delta [0.3, 1.0] m
        """
        min_h = self.cfg["ground_level"] + self.cfg["min_camera_height"]
        max_h = self.cfg["ground_level"] + self.cfg["max_camera_height"]

        # short clips: keep a constant height
        if total_frames < 60:
            return np.full(total_frames, base_height)

        # amplitude range by clip length
        if total_frames < 180:
            delta_range = (0.3, 0.5)
            perlin_amp_range = (0.15, 0.3)
        elif total_frames < 360:
            delta_range = (0.3, 0.8)
            perlin_amp_range = (0.2, 0.5)
        else:
            delta_range = (0.3, 1.0)
            perlin_amp_range = (0.3, 0.8)

        rand = random.random()

        if rand < 0.5:
            # constant height
            return np.full(total_frames, base_height)

        elif rand < 0.8:
            # linear ramp up or down
            delta = random.uniform(*delta_range)
            direction = random.choice([-1, 1])
            end_height = base_height + direction * delta
            end_height = max(min_h, min(end_height, max_h))
            return np.linspace(base_height, end_height, total_frames)

        else:
            # Perlin noise fluctuation
            amplitude = random.uniform(*perlin_amp_range)
            # lower frequency for longer clips to keep sway speed constant
            frequency = random.uniform(0.03, 0.08) * (180.0 / max(total_frames, 1))
            noise = PerlinNoise1D(seed=random.randint(0, 9999))
            heights = np.array([
                np.clip(base_height + noise.noise(f * frequency) * amplitude, min_h, max_h)
                for f in range(total_frames)
            ])
            return heights

    def _generate_orbit_trajectory(self, bounds, total_frames, focal_length):
        """Orbit around the clip center with a limited rotation amplitude."""
        radius = self._max_frame_distance(bounds, total_frames, focal_length, (1.15, 1.4))

        center = bounds['center']
        start_angle = self.generate_biased_angle()

        # rotation amplitude is limited by clip length
        if total_frames < 120:
            total_rotation = random.uniform(pi / 6, pi / 3)
        elif total_frames < 360:
            total_rotation = random.uniform(pi / 6, pi / 2)
        else:
            total_rotation = random.uniform(pi / 6, pi)
        total_rotation = min(total_rotation, total_frames * pi / 360)
        total_rotation *= random.choice([-1, 1])

        height, _ = self.generate_height()
        heights = self._generate_height_curve(height, total_frames)

        frames = []
        for frame in range(total_frames):
            progress = frame / (total_frames - 1) if total_frames > 1 else 0
            angle = start_angle + total_rotation * progress

            x = center.x + radius * cos(angle)
            y = center.y + radius * sin(angle)
            z = heights[frame]

            camera_pos = Vector((x, y, z))
            human_center = self.get_human_center(frame)

            direction = human_center - camera_pos
            if direction.length > 0:
                rotation = direction.to_track_quat('-Z', 'Y').to_euler()
            else:
                rotation = Euler((0, 0, 0))
            frames.append({
                'frame': frame + 1,
                'location': tuple(camera_pos),
                'rotation': tuple(rotation)
            })
        return frames

    def _generate_arc_trajectory(self, bounds, total_frames, focal_length):
        """Arc: orbit while increasing the distance from the center."""
        safe_distance = self._max_frame_distance(bounds, total_frames, focal_length, (1.15, 1.2))

        center = bounds['center']
        dis_factor = random.uniform(1.2, 1.6)
        start_distance = safe_distance * 1.0
        end_distance = safe_distance * dis_factor

        start_angle = self.generate_biased_angle()
        if total_frames < 120:
            total_rotation = random.uniform(pi / 6, pi / 3)
        elif total_frames < 360:
            total_rotation = random.uniform(pi / 6, pi / 2)
        else:
            total_rotation = random.uniform(pi / 6, pi)
        total_rotation = min(total_rotation, total_frames * pi / 360)
        total_rotation *= random.choice([-1, 1])

        height, _ = self.generate_height()
        heights = self._generate_height_curve(height, total_frames)

        distances = np.linspace(start_distance, end_distance, total_frames)
        angles = np.linspace(start_angle, start_angle + total_rotation, total_frames)

        frames = []
        for frame in range(total_frames):
            x = center.x + distances[frame] * cos(angles[frame])
            y = center.y + distances[frame] * sin(angles[frame])
            z = heights[frame]

            camera_pos = Vector((x, y, z))
            human_center = self.get_human_center(frame)

            direction = human_center - camera_pos
            if direction.length > 0:
                rotation = direction.to_track_quat('-Z', 'Y').to_euler()
            else:
                rotation = Euler((0, 0, 0))
            frames.append({
                'frame': frame + 1,
                'location': tuple(camera_pos),
                'rotation': tuple(rotation)
            })
        return frames

    def _generate_push_trajectory(self, bounds, total_frames, focal_length):
        """Push in: start far, move closer along a fixed direction."""
        safe_distance = self._max_frame_distance(bounds, total_frames, focal_length, (1.15, 1.2))

        center = bounds['center']
        dis_factor = random.uniform(1.2, 1.6)
        start_distance = safe_distance * dis_factor
        end_distance = safe_distance * 1.0

        angle = self.generate_biased_angle()
        direction = Vector((cos(angle), sin(angle), 0))
        height, _ = self.generate_height()
        heights = self._generate_height_curve(height, total_frames)

        distances = np.linspace(start_distance, end_distance, total_frames)

        frames = []
        for frame in range(total_frames):
            horizontal_pos = center + direction * distances[frame]
            camera_pos = Vector((horizontal_pos.x, horizontal_pos.y, heights[frame]))

            human_center = self.get_human_center(frame)
            look_direction = human_center - camera_pos

            if look_direction.length > 0:
                rotation = look_direction.to_track_quat('-Z', 'Y').to_euler()
            else:
                rotation = Euler((0, 0, 0))
            frames.append({
                'frame': frame + 1,
                'location': tuple(camera_pos),
                'rotation': tuple(rotation)
            })
        return frames

    def _generate_pull_trajectory(self, bounds, total_frames, focal_length):
        """Pull out: start close, move away along a fixed direction."""
        safe_distance = self._max_frame_distance(bounds, total_frames, focal_length, (1.15, 1.2))

        center = bounds['center']
        dis_factor = random.uniform(1.2, 1.6)
        start_distance = safe_distance * 1.0
        end_distance = safe_distance * dis_factor

        angle = self.generate_biased_angle()
        direction = Vector((cos(angle), sin(angle), 0))
        height, _ = self.generate_height()
        heights = self._generate_height_curve(height, total_frames)

        distances = np.linspace(start_distance, end_distance, total_frames)

        frames = []
        for frame in range(total_frames):
            horizontal_pos = center + direction * distances[frame]
            camera_pos = Vector((horizontal_pos.x, horizontal_pos.y, heights[frame]))

            human_center = self.get_human_center(frame)
            look_direction = human_center - camera_pos

            if look_direction.length > 0:
                rotation = look_direction.to_track_quat('-Z', 'Y').to_euler()
            else:
                rotation = Euler((0, 0, 0))
            frames.append({
                'frame': frame + 1,
                'location': tuple(camera_pos),
                'rotation': tuple(rotation)
            })
        return frames


def get_calibration_matrix_K_from_blender(camera=None):
    """Intrinsics K (OpenCV convention) from a Blender camera."""
    scene = bpy.context.scene

    scale = scene.render.resolution_percentage / 100
    width = scene.render.resolution_x * scale  # px
    height = scene.render.resolution_y * scale  # px
    if camera is None:
        camdata = scene.camera.data
    else:
        camdata = camera.data

    focal = camdata.lens  # mm
    sensor_width = camdata.sensor_width  # mm
    sensor_height = camdata.sensor_height  # mm

    if camdata.sensor_fit == 'VERTICAL':
        # sensor height fixed; sensor width scales with pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
        s_v = height / sensor_height
        s_u = s_v * pixel_aspect_ratio
    else:  # 'HORIZONTAL' and 'AUTO'
        # sensor width fixed; sensor height scales with pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = width / sensor_width
        s_v = s_u * pixel_aspect_ratio

    alpha_u = focal * s_u
    alpha_v = focal * s_v
    u_0 = width / 2
    v_0 = height / 2
    skew = 0  # rectangular pixels only
    K = np.array([
        [alpha_u, skew, u_0],
        [0, alpha_v, v_0],
        [0, 0, 1]
    ], dtype=np.float32)

    return K


def get_3x4_RT_matrix_from_blender(obj):
    """World-to-camera extrinsics (OpenCV convention) from a Blender object."""
    is_camera = (obj.type == 'CAMERA')
    R_blender_to_opencv = np.diag([1 if is_camera else -1, -1, -1])

    location, rotation = obj.matrix_world.decompose()[:2]
    R_blender_view = rotation.to_matrix().transposed()

    T_blender_view = -1.0 * np.asarray(R_blender_view) @ location

    R_opencv = R_blender_to_opencv @ R_blender_view
    T_opencv = R_blender_to_opencv @ T_blender_view

    return np.column_stack((R_opencv, T_opencv))


def parse_args():
    """Parse command line arguments (after the `--` separator in Blender)."""
    parser = argparse.ArgumentParser(description='Generate camera from FBX')
    parser.add_argument('--fbx', type=str, required=True, help='FBX file path')
    parser.add_argument('--output', type=str, required=True, help='Output camera NPZ file')
    parser.add_argument('--movement_type', type=str, default='orbit', help='Camera movement type')
    parser.add_argument('--shake_flag', action='store_true', default=False,
                        help='Add hand-held camera shake')
    parser.add_argument('--res_x', type=int, default=1920, help='Resolution X')
    parser.add_argument('--res_y', type=int, default=1080, help='Resolution Y')

    if '--' in sys.argv:
        args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    else:
        args = parser.parse_args()

    return args


def generate_camera(fbx_path, camera_output, movement_type, shake_flag,
                    resolution_x=1920, resolution_y=1080):
    """Generate a camera trajectory for the given FBX and save it as NPZ."""
    # clear scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    bpy.context.scene.render.resolution_x = resolution_x
    bpy.context.scene.render.resolution_y = resolution_y

    scale = bpy.context.scene.render.resolution_percentage / 100
    width = resolution_x * scale  # px
    height = resolution_y * scale  # px

    print(f"resolution: {bpy.context.scene.render.resolution_x}x{bpy.context.scene.render.resolution_y}")

    # load FBX
    print(f"loading FBX: {fbx_path}")
    bpy.ops.import_scene.fbx(filepath=fbx_path)

    # find armature
    armature = None
    for obj in bpy.context.scene.objects:
        if obj.type == 'ARMATURE':
            armature = obj
            break

    if not armature:
        print("error: no armature found in FBX")
        return False

    print(f"found armature: {armature.name}")

    generator = CameraTrajectoryGenerator(config=None, shake_flag=shake_flag)
    success = generator.load_from_blender_armature(armature)
    if not success:
        print("error: failed to load data from armature")
        return False

    camera_params = generator.generate_random_trajectory(movement_type=movement_type)
    if not camera_params:
        print("error: failed to generate camera trajectory")
        return False

    # create the camera
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    bpy.context.scene.camera = camera

    camera.data.sensor_width = 36
    camera.data.sensor_height = 24
    focal_length = camera_params['focal_length']
    sensor_width = camera.data.sensor_width
    sensor_height = camera.data.sensor_height

    horizontal_fov_rad = 2 * math.atan(sensor_width / (2 * focal_length))
    vertical_fov_rad = 2 * math.atan(sensor_height / (2 * focal_length))

    # match the sensor fit to the render aspect ratio so the lens covers
    # exactly the intended field of view
    render_aspect_ratio = width / height
    sensor_aspect_ratio = camera.data.sensor_width / camera.data.sensor_height
    camera.data.lens_unit = 'FOV'
    if render_aspect_ratio > sensor_aspect_ratio:
        camera.data.sensor_fit = 'HORIZONTAL'
        camera.data.angle = horizontal_fov_rad
    else:
        camera.data.sensor_fit = 'VERTICAL'
        camera.data.angle = vertical_fov_rad
    camera.data.lens = focal_length

    num_frames = len(camera_params['frames'])

    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = max(num_frames - 1, 0)

    # apply the trajectory as per-frame keyframes
    print(f"applying camera trajectory for {num_frames} frames...")
    for frame_idx in range(num_frames):
        bpy.context.scene.frame_set(frame_idx)
        camera.location = camera_params['frames'][frame_idx]['location']
        camera.rotation_euler = camera_params['frames'][frame_idx]['rotation']
        camera.keyframe_insert(data_path="location", frame=frame_idx)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame_idx)

    # export camera parameters (K, RT, location, rotation, lens)
    print("calculating camera parameters...")
    K_list, w2c_list = [], []
    location_list, rotation_list = [], []
    for frame in range(num_frames):
        bpy.context.scene.frame_set(frame)

        K = get_calibration_matrix_K_from_blender(camera)
        w2c = get_3x4_RT_matrix_from_blender(camera)

        location = np.array(camera.location)
        rotation = np.array(camera.rotation_euler)

        # z-up (Blender) -> y-up world convention used by the annotations
        translation = np.array([[1, 0, 0, 0],
                                [0, 0, -1, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 1]])
        w2c = w2c @ translation

        K_list.append(K)
        w2c_list.append(w2c)
        location_list.append(location)
        rotation_list.append(rotation)

    os.makedirs(os.path.dirname(camera_output), exist_ok=True)
    np.savez(
        camera_output,
        K=np.array(K_list),
        RT=np.array(w2c_list),
        location=np.array(location_list),
        rotation=np.array(rotation_list),
        focal_length=focal_length,
        sensor_width=sensor_width,
        sensor_height=sensor_height,
        movement_type=camera_params['movement_type']
    )

    print(f"camera parameters saved: {camera_output}")
    return True


if __name__ == "__main__":
    args = parse_args()
    generate_camera(args.fbx, args.output, args.movement_type, args.shake_flag,
                    args.res_x, args.res_y)

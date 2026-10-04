"""Blender render executor (runs inside Blender).

Reads a task_config JSON (written by render_pipeline.py as <task_id>_meta.json) and renders
an SMPL-X/H FBX motion to an H.264 MP4 with Blender Cycles.

Pipeline inside this script:
  1. Clear the scene and import the FBX (armature + skinned mesh)
  2. Restore the camera from the trajectory .npz (per-frame keyframes), or
     fall back to a fixed default camera
  3. Optionally add occluders (procedural shapes with random materials,
     placement modes and optional animation)
  4. Set up lighting: HDR environment (strength 2.0) or a default sun
  5. Apply a skin texture (single or multi-texture mode with in-video
     outfit switching), or a plain default material
  6. Add a ground: PBR material set (GLTF + basecolor/normal/ORM/MR
     textures) or a single tiled image
  7. Configure Cycles (samples, adaptive sampling, GPU, optional motion
     blur with random shutter) and render to a temp dir, then move the
     finished MP4 to the final output path

Usage:
    blender --background --python scripts/render_in_blender.py -- \
        --task_config output/.../<task_id>_meta.json
"""

import os
import sys
import json
import math
import random
import shutil
import tempfile
import argparse
import warnings
import logging

import numpy as np
from mathutils import Vector, Euler

import bpy

os.environ.setdefault('PYTHONWARNINGS', 'ignore')
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
logging.disable(logging.WARNING)


class OccluderGenerator:
    """Procedural occluder generator.

    Places random shapes between the camera and the target to create partial
    occlusions in the rendered video.
    """

    # shape types
    SHAPE_CUBE = 'cube'
    SHAPE_SPHERE = 'sphere'
    SHAPE_CYLINDER = 'cylinder'
    SHAPE_CONE = 'cone'
    SHAPE_TORUS = 'torus'
    SHAPE_PLANE = 'plane'
    SHAPE_MONKEY = 'monkey'       # Blender's Suzanne head
    SHAPE_ICO_SPHERE = 'ico_sphere'
    SHAPE_CAPSULE = 'capsule'

    # placement modes
    MODE_FOREGROUND = 'foreground'  # between camera and target
    MODE_EDGE = 'edge'              # at the frame border
    MODE_CORNER = 'corner'          # at a frame corner
    MODE_RANDOM = 'random'          # random position
    MODE_PASSING = 'passing'        # crosses the frame (animated)

    # material types
    MATERIAL_SOLID = 'solid'
    MATERIAL_TRANSPARENT = 'transparent'
    MATERIAL_EMISSION = 'emission'
    MATERIAL_GLASS = 'glass'
    MATERIAL_METALLIC = 'metallic'
    MATERIAL_GRADIENT = 'gradient'
    MATERIAL_CHECKER = 'checker'
    MATERIAL_NOISE = 'noise'

    def __init__(self, scene, camera, config=None):
        """
        Args:
            scene: Blender scene
            camera: camera object
            config: occluder configuration dict
        """
        self.scene = scene
        self.camera = camera
        self.config = config or {}
        self.occluders = []

        self.default_config = {
            'enabled': True,
            'min_count': 0,
            'max_count': 5,
            'shapes': [self.SHAPE_CUBE, self.SHAPE_SPHERE, self.SHAPE_CYLINDER],
            'modes': [self.MODE_FOREGROUND, self.MODE_EDGE, self.MODE_RANDOM],
            'materials': [self.MATERIAL_SOLID, self.MATERIAL_TRANSPARENT],
            'size_range': (0.1, 0.8),
            'transparency_range': (0.3, 0.9),
            'animate': False,
            'depth_range': (0.2, 0.8),     # fraction of camera-target distance
        }

        # merge user config with defaults
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value

    def generate_occluders(self, target_object=None):
        """
        Args:
            target_object: target object (used to compute placement)

        Returns:
            list of generated occluder objects
        """
        if not self.config.get('enabled', True):
            return []

        min_count = self.config.get('min_count', 0)
        max_count = self.config.get('max_count', 5)
        num_occluders = random.randint(min_count, max_count)

        if num_occluders == 0:
            return []

        target_location = Vector((0, 0, 0))
        if target_object:
            target_location = self._get_object_center(target_object)

        for i in range(num_occluders):
            occluder = self._create_single_occluder(i, target_location)
            if occluder:
                self.occluders.append(occluder)

        return self.occluders

    def _create_single_occluder(self, index, target_location):
        """Create a single occluder (random shape, mode and material)."""
        shapes = self.config.get('shapes', [self.SHAPE_CUBE])
        shape = random.choice(shapes)

        modes = self.config.get('modes', [self.MODE_RANDOM])
        mode = random.choice(modes)

        materials = self.config.get('materials', [self.MATERIAL_SOLID])
        material_type = random.choice(materials)

        size_range = self.config.get('size_range', (0.1, 0.8))
        size = random.uniform(size_range[0], size_range[1])

        occluder = self._create_shape(shape, size, index)
        if not occluder:
            return None

        self._position_occluder(occluder, mode, target_location)
        self._apply_material(occluder, material_type)

        if self.config.get('animate', False):
            self._animate_occluder(occluder, mode)

        return occluder

    def _create_shape(self, shape_type, size, index):
        """Create the geometry for the given shape type."""
        name = f"Occluder_{index}_{shape_type}"

        if shape_type == self.SHAPE_CUBE:
            bpy.ops.mesh.primitive_cube_add(size=size)

        elif shape_type == self.SHAPE_SPHERE:
            bpy.ops.mesh.primitive_uv_sphere_add(radius=size / 2, segments=16, ring_count=8)

        elif shape_type == self.SHAPE_CYLINDER:
            bpy.ops.mesh.primitive_cylinder_add(
                radius=size / 3,
                depth=size * random.uniform(0.5, 2.0),
                vertices=16
            )

        elif shape_type == self.SHAPE_CONE:
            bpy.ops.mesh.primitive_cone_add(
                radius1=size / 2,
                depth=size * random.uniform(0.8, 1.5),
                vertices=16
            )

        elif shape_type == self.SHAPE_TORUS:
            bpy.ops.mesh.primitive_torus_add(
                major_radius=size / 2,
                minor_radius=size / 6,
                major_segments=24,
                minor_segments=12
            )

        elif shape_type == self.SHAPE_PLANE:
            bpy.ops.mesh.primitive_plane_add(size=size * 2)

        elif shape_type == self.SHAPE_MONKEY:
            bpy.ops.mesh.primitive_monkey_add(size=size)

        elif shape_type == self.SHAPE_ICO_SPHERE:
            bpy.ops.mesh.primitive_ico_sphere_add(radius=size / 2, subdivisions=2)

        elif shape_type == self.SHAPE_CAPSULE:
            # capsule = cylinder + two half spheres
            bpy.ops.mesh.primitive_cylinder_add(
                radius=size / 4,
                depth=size,
                vertices=16
            )
            capsule = bpy.context.active_object

            bpy.ops.mesh.primitive_uv_sphere_add(radius=size / 4, segments=16, ring_count=8)
            sphere1 = bpy.context.active_object
            sphere1.location.z = size / 2

            bpy.ops.mesh.primitive_uv_sphere_add(radius=size / 4, segments=16, ring_count=8)
            sphere2 = bpy.context.active_object
            sphere2.location.z = -size / 2

            bpy.ops.object.select_all(action='DESELECT')
            capsule.select_set(True)
            sphere1.select_set(True)
            sphere2.select_set(True)
            bpy.context.view_layer.objects.active = capsule
            bpy.ops.object.join()

        else:
            # fallback: cube
            bpy.ops.mesh.primitive_cube_add(size=size)

        occluder = bpy.context.active_object
        occluder.name = name

        # random orientation
        occluder.rotation_euler = Euler((
            random.uniform(0, math.pi * 2),
            random.uniform(0, math.pi * 2),
            random.uniform(0, math.pi * 2)
        ))

        # random per-axis scale variation
        scale_variation = random.uniform(0.7, 1.3)
        occluder.scale = Vector((
            random.uniform(0.5, 1.5) * scale_variation,
            random.uniform(0.5, 1.5) * scale_variation,
            random.uniform(0.5, 1.5) * scale_variation
        ))

        return occluder

    def _position_occluder(self, occluder, mode, target_location):
        """Place an occluder according to its placement mode."""
        cam_loc = self.camera.location.copy()

        if mode == self.MODE_FOREGROUND:
            # between the camera and the target
            depth_range = self.config.get('depth_range', (0.2, 0.8))
            t = random.uniform(depth_range[0], depth_range[1])

            direction = target_location - cam_loc
            base_pos = cam_loc + direction * t

            right = direction.cross(Vector((0, 0, 1))).normalized()
            up = direction.cross(right).normalized()

            lateral_offset = random.uniform(-0.5, 0.5)
            vertical_offset = random.uniform(-0.3, 0.3)

            occluder.location = base_pos + right * lateral_offset + up * vertical_offset

        elif mode == self.MODE_EDGE:
            # at one of the four frame borders
            depth_range = self.config.get('depth_range', (0.2, 0.8))
            t = random.uniform(depth_range[0], depth_range[1])

            direction = target_location - cam_loc
            base_pos = cam_loc + direction * t

            right = direction.cross(Vector((0, 0, 1))).normalized()
            up = direction.cross(right).normalized()

            edge = random.choice(['left', 'right', 'top', 'bottom'])
            edge_distance = random.uniform(0.8, 1.2)

            if edge == 'left':
                occluder.location = base_pos - right * edge_distance
            elif edge == 'right':
                occluder.location = base_pos + right * edge_distance
            elif edge == 'top':
                occluder.location = base_pos + up * edge_distance
            elif edge == 'bottom':
                occluder.location = base_pos - up * edge_distance

        elif mode == self.MODE_CORNER:
            # at one of the four frame corners
            depth_range = self.config.get('depth_range', (0.2, 0.8))
            t = random.uniform(depth_range[0], depth_range[1])

            direction = target_location - cam_loc
            base_pos = cam_loc + direction * t

            right = direction.cross(Vector((0, 0, 1))).normalized()
            up = direction.cross(right).normalized()

            corner = random.choice(['top_left', 'top_right', 'bottom_left', 'bottom_right'])
            corner_distance = random.uniform(0.6, 1.0)

            if corner == 'top_left':
                occluder.location = base_pos - right * corner_distance + up * corner_distance
            elif corner == 'top_right':
                occluder.location = base_pos + right * corner_distance + up * corner_distance
            elif corner == 'bottom_left':
                occluder.location = base_pos - right * corner_distance - up * corner_distance
            elif corner == 'bottom_right':
                occluder.location = base_pos + right * corner_distance - up * corner_distance

        elif mode == self.MODE_PASSING:
            # starts off-frame on one side (used with animation)
            depth_range = self.config.get('depth_range', (0.2, 0.8))
            t = random.uniform(depth_range[0], depth_range[1])

            direction = target_location - cam_loc
            base_pos = cam_loc + direction * t

            right = direction.cross(Vector((0, 0, 1))).normalized()

            occluder.location = base_pos - right * 2.0

        else:  # MODE_RANDOM
            t = random.uniform(0.2, 0.9)
            direction = target_location - cam_loc
            base_pos = cam_loc + direction * t

            offset = Vector((
                random.uniform(-1.0, 1.0),
                random.uniform(-1.0, 1.0),
                random.uniform(-0.5, 0.5)
            ))

            occluder.location = base_pos + offset

    def _apply_material(self, occluder, material_type):
        """Apply a procedural node material to an occluder."""
        mat_name = f"{occluder.name}_Material"
        material = bpy.data.materials.new(name=mat_name)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links

        # clear default nodes and create a fresh output
        nodes.clear()
        output_node = nodes.new(type='ShaderNodeOutputMaterial')
        output_node.location = (400, 0)

        base_color = (
            random.random(),
            random.random(),
            random.random(),
            1.0
        )

        def new_principled_bsdf():
            bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf.location = (0, 0)
            links.new(bsdf.outputs['BSDF'], output_node.inputs['Surface'])
            return bsdf

        if material_type == self.MATERIAL_SOLID:
            bsdf = new_principled_bsdf()
            bsdf.inputs['Base Color'].default_value = base_color
            bsdf.inputs['Roughness'].default_value = random.uniform(0.3, 0.9)

        elif material_type == self.MATERIAL_TRANSPARENT:
            transparency_range = self.config.get('transparency_range', (0.3, 0.9))
            alpha = random.uniform(transparency_range[0], transparency_range[1])

            bsdf = new_principled_bsdf()
            bsdf.inputs['Base Color'].default_value = base_color
            bsdf.inputs['Alpha'].default_value = alpha
            bsdf.inputs['Roughness'].default_value = random.uniform(0.1, 0.5)

            material.blend_method = 'BLEND'
            material.shadow_method = 'HASHED'

        elif material_type == self.MATERIAL_EMISSION:
            emission = nodes.new(type='ShaderNodeEmission')
            emission.location = (0, 0)
            emission.inputs['Color'].default_value = base_color
            emission.inputs['Strength'].default_value = random.uniform(1.0, 5.0)
            links.new(emission.outputs['Emission'], output_node.inputs['Surface'])

        elif material_type == self.MATERIAL_GLASS:
            bsdf = nodes.new(type='ShaderNodeBsdfGlass')
            bsdf.location = (0, 0)
            bsdf.inputs['Color'].default_value = base_color
            bsdf.inputs['Roughness'].default_value = random.uniform(0.0, 0.3)
            bsdf.inputs['IOR'].default_value = random.uniform(1.3, 1.6)

            material.blend_method = 'BLEND'
            material.shadow_method = 'HASHED'

            links.new(bsdf.outputs['BSDF'], output_node.inputs['Surface'])

        elif material_type == self.MATERIAL_METALLIC:
            bsdf = new_principled_bsdf()
            bsdf.inputs['Base Color'].default_value = base_color
            bsdf.inputs['Metallic'].default_value = 1.0
            bsdf.inputs['Roughness'].default_value = random.uniform(0.1, 0.5)

        elif material_type == self.MATERIAL_GRADIENT:
            gradient = nodes.new(type='ShaderNodeTexGradient')
            gradient.location = (-400, 0)
            gradient.gradient_type = random.choice(['LINEAR', 'RADIAL', 'EASING'])

            color_ramp = nodes.new(type='ShaderNodeValToRGB')
            color_ramp.location = (-200, 0)
            color_ramp.color_ramp.elements[0].color = base_color
            color_ramp.color_ramp.elements[1].color = (
                random.random(),
                random.random(),
                random.random(),
                1.0
            )

            bsdf = new_principled_bsdf()

            links.new(gradient.outputs['Fac'], color_ramp.inputs['Fac'])
            links.new(color_ramp.outputs['Color'], bsdf.inputs['Base Color'])

        elif material_type == self.MATERIAL_CHECKER:
            checker = nodes.new(type='ShaderNodeTexChecker')
            checker.location = (-200, 0)
            checker.inputs['Scale'].default_value = random.uniform(2.0, 10.0)
            checker.inputs['Color1'].default_value = base_color
            checker.inputs['Color2'].default_value = (
                random.random(),
                random.random(),
                random.random(),
                1.0
            )

            bsdf = new_principled_bsdf()

            links.new(checker.outputs['Color'], bsdf.inputs['Base Color'])

        elif material_type == self.MATERIAL_NOISE:
            noise = nodes.new(type='ShaderNodeTexNoise')
            noise.location = (-400, 0)
            noise.inputs['Scale'].default_value = random.uniform(2.0, 20.0)
            noise.inputs['Detail'].default_value = random.uniform(2.0, 8.0)

            color_ramp = nodes.new(type='ShaderNodeValToRGB')
            color_ramp.location = (-200, 0)
            color_ramp.color_ramp.elements[0].color = base_color
            color_ramp.color_ramp.elements[1].color = (
                random.random(),
                random.random(),
                random.random(),
                1.0
            )

            bsdf = new_principled_bsdf()

            links.new(noise.outputs['Fac'], color_ramp.inputs['Fac'])
            links.new(color_ramp.outputs['Color'], bsdf.inputs['Base Color'])

        else:
            # fallback: plain solid color
            bsdf = new_principled_bsdf()
            bsdf.inputs['Base Color'].default_value = base_color

        if occluder.data.materials:
            occluder.data.materials[0] = material
        else:
            occluder.data.materials.append(material)

    def _animate_occluder(self, occluder, mode):
        """Add animation to an occluder (passing / floating / rotating / wobbling)."""
        frame_start = self.scene.frame_start
        frame_end = self.scene.frame_end
        num_frames = frame_end - frame_start + 1

        if mode == self.MODE_PASSING:
            # translate across the frame
            start_loc = occluder.location.copy()

            direction = self.camera.matrix_world.to_3x3() @ Vector((1, 0, 0))
            end_loc = start_loc + direction * 4.0

            for frame in range(frame_start, frame_end + 1):
                t = (frame - frame_start) / num_frames
                occluder.location = start_loc.lerp(end_loc, t)
                occluder.keyframe_insert(data_path="location", frame=frame)

        else:
            # subtle idle animation
            animation_type = random.choice(['float', 'rotate', 'wobble', 'none'])

            if animation_type == 'float':
                # bob up and down
                base_loc = occluder.location.copy()
                amplitude = random.uniform(0.05, 0.2)
                frequency = random.uniform(0.5, 2.0)

                for frame in range(frame_start, frame_end + 1):
                    t = (frame - frame_start) / 30.0  # assumes ~30 fps
                    offset_z = amplitude * math.sin(frequency * t * math.pi * 2)
                    occluder.location = base_loc + Vector((0, 0, offset_z))
                    occluder.keyframe_insert(data_path="location", frame=frame)

            elif animation_type == 'rotate':
                # spin around one axis
                base_rot = occluder.rotation_euler.copy()
                rotation_speed = random.uniform(0.5, 2.0)
                axis = random.choice(['x', 'y', 'z'])

                for frame in range(frame_start, frame_end + 1):
                    t = (frame - frame_start) / 24.0
                    angle = rotation_speed * t * math.pi * 2

                    if axis == 'x':
                        occluder.rotation_euler = Euler((base_rot.x + angle, base_rot.y, base_rot.z))
                    elif axis == 'y':
                        occluder.rotation_euler = Euler((base_rot.x, base_rot.y + angle, base_rot.z))
                    else:
                        occluder.rotation_euler = Euler((base_rot.x, base_rot.y, base_rot.z + angle))

                    occluder.keyframe_insert(data_path="rotation_euler", frame=frame)

            elif animation_type == 'wobble':
                # combined positional and rotational sway
                base_loc = occluder.location.copy()
                base_rot = occluder.rotation_euler.copy()

                amplitude_loc = random.uniform(0.02, 0.1)
                amplitude_rot = random.uniform(0.05, 0.2)
                frequency = random.uniform(1.0, 3.0)

                for frame in range(frame_start, frame_end + 1):
                    t = (frame - frame_start) / 24.0

                    offset = Vector((
                        amplitude_loc * math.sin(frequency * t * math.pi * 2),
                        amplitude_loc * math.cos(frequency * t * math.pi * 2 * 0.7),
                        amplitude_loc * math.sin(frequency * t * math.pi * 2 * 1.3) * 0.5
                    ))
                    occluder.location = base_loc + offset
                    occluder.keyframe_insert(data_path="location", frame=frame)

                    rot_offset = Euler((
                        amplitude_rot * math.sin(frequency * t * math.pi * 2),
                        amplitude_rot * math.cos(frequency * t * math.pi * 2 * 0.8),
                        amplitude_rot * math.sin(frequency * t * math.pi * 2 * 1.2)
                    ))
                    occluder.rotation_euler = Euler((
                        base_rot.x + rot_offset.x,
                        base_rot.y + rot_offset.y,
                        base_rot.z + rot_offset.z
                    ))
                    occluder.keyframe_insert(data_path="rotation_euler", frame=frame)

    def _get_object_center(self, obj):
        """World-space center of an object (bounding box center for meshes)."""
        if obj.type == 'MESH':
            world_bound_box = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
            center = sum(world_bound_box, Vector()) / 8
            return center
        return obj.location.copy()


def _first_image(texture_dir):
    """First image file (sorted by name) in a texture directory, or None."""
    for filename in sorted(os.listdir(texture_dir)):
        file_path = os.path.join(texture_dir, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return file_path
    return None


class BlenderRenderExecutor:
    """Executes a single render task described by a task_config dict."""

    def __init__(self, task_config):
        """
        Args:
            task_config: task config dictionary (see RenderPipeline.run in render_pipeline.py)
        """
        self.task_config = task_config
        self.render_config = task_config.get('config', {})
        self.occluder_config = task_config.get('occluder_config', {})

    def render(self):
        """Execute the render task."""
        print(f"start render task: {self.task_config['task_id']}")

        # clear scene
        self._clear_scene()

        # import FBX
        fbx_path = self.task_config['fbx_path']
        print(f"   import FBX: {os.path.basename(fbx_path)}")
        bpy.ops.import_scene.fbx(filepath=fbx_path)

        obj_names = [o.name for o in bpy.context.selected_objects]
        armature, mesh_object, mesh_object_list = self._find_armature_and_mesh(obj_names)

        if not mesh_object:
            print("   error: no mesh object found")
            return {'status': 'failed', 'error': 'No mesh object found'}

        # set the animation frame range from the armature action
        if armature and armature.animation_data and armature.animation_data.action:
            action = armature.animation_data.action
            frame_start = int(action.frame_range[0])
            frame_end = int(action.frame_range[1])
            bpy.context.scene.frame_start = frame_start
            bpy.context.scene.frame_end = frame_end
            print(f"   animation frames: {frame_start} - {frame_end}")

        # load and apply the camera trajectory
        camera_params_path = self.task_config.get('camera_params_path')
        if camera_params_path and os.path.exists(camera_params_path):
            print(f"   load camera parameters: {os.path.basename(camera_params_path)}")
            self._load_camera_from_npz(camera_params_path)
        else:
            print("   use default camera")
            self._setup_default_camera(mesh_object)

        # add occluders
        self._add_occluders(mesh_object)

        # HDR environment or default lighting
        hdr_file = self.task_config.get('hdr_file')
        if hdr_file:
            print(f"   set HDR: {os.path.basename(hdr_file)}")
            self._setup_hdr_environment(hdr_file)
        else:
            print("   use default lighting")
            self._setup_default_lighting()

        # skin texture (single- or multi-texture mode)
        texture_dirs = self.task_config.get('texture_dirs')  # multi-texture list
        texture_dir = self.task_config.get('texture_dir')    # single texture

        if texture_dirs and len(texture_dirs) > 1 and mesh_object_list:
            # multi-texture mode: switch outfits during the video
            print(f"   set multi-textures: {len(texture_dirs)} textures")
            for td in texture_dirs:
                print(f"      - {os.path.basename(td)}")
            self._set_multi_textures(mesh_object_list, texture_dirs)
        elif texture_dirs and len(texture_dirs) == 1 and mesh_object_list:
            print(f"   set texture: {os.path.basename(texture_dirs[0])}")
            for mesh_obj in mesh_object_list:
                self._set_texture(mesh_obj, texture_dirs[0])
        elif texture_dir and mesh_object_list:
            print(f"   set texture: {os.path.basename(texture_dir)}")
            for mesh_obj in mesh_object_list:
                self._set_texture(mesh_obj, texture_dir)
        else:
            print("   use default material")
            for mesh_obj in mesh_object_list:
                self._set_default_material(mesh_obj)

        # ground
        ground_material = self.task_config.get('ground_material')
        ground_image = self.task_config.get('ground_image')
        if ground_image:
            # tiled single-image ground
            image_path = ground_image.get('image_path') if isinstance(ground_image, dict) else ground_image
            tile_scale = ground_image.get('tile_scale', 1.0) if isinstance(ground_image, dict) else 1.0
            scale = ground_image.get('scale', 50.0) if isinstance(ground_image, dict) else 50.0
            print(f"   set ground image: {os.path.basename(image_path)}")
            try:
                ground_obj = self._setup_ground_with_image(
                    image_path=image_path,
                    scale=scale,
                    tile_scale=tile_scale,
                    use_shadow_catcher=False
                )
                if ground_obj and mesh_object:
                    self._auto_position_ground(ground_obj, mesh_object)
            except Exception as e:
                print(f"   warning: ground image setup failed: {e}")
        elif ground_material:
            # GLTF + PBR texture set
            print(f"   set ground: {ground_material['name']}")
            try:
                ground_obj = self._setup_ground_with_gltf_material(
                    gltf_path=ground_material['gltf_path'],
                    texture_dir=ground_material['texture_dir'],
                    scale=50.0,
                    use_shadow_catcher=False
                )
                if ground_obj and mesh_object:
                    self._auto_position_ground(ground_obj, mesh_object)
            except Exception as e:
                print(f"   warning: ground setup failed: {e}")

        # renderer and output
        self._setup_renderer()

        task_id = self.task_config['task_id']
        output_path = os.path.join(self.task_config['output_path'], f"{task_id}.mp4")

        # render into a temp dir first, move on success (avoids partial files)
        temp_dir = tempfile.mkdtemp(prefix=f"blender_render_{task_id}_")
        temp_output_path = os.path.join(temp_dir, f"{task_id}.mp4")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self._set_output_properties(temp_output_path)

        print("   start render...")
        try:
            # silence the extremely verbose per-tile render progress
            self._render_with_silent_output()

            if os.path.exists(temp_output_path):
                shutil.move(temp_output_path, output_path)
                print(f"  [ok] render completed: {output_path}")
            else:
                raise FileNotFoundError(f"Rendered file not found: {temp_output_path}")

            return {
                'status': 'success',
                'task_id': task_id,
                'output': output_path
            }

        except Exception as e:
            print(f"  [error] render failed: {e}")
            return {
                'status': 'failed',
                'task_id': task_id,
                'error': str(e)
            }
        finally:
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"   warning: failed to clean up temp dir {temp_dir}: {e}")

    @staticmethod
    def _render_with_silent_output():
        """Render the animation while redirecting stdout/stderr to /dev/null."""
        sys.stdout.flush()
        sys.stderr.flush()

        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()

        stdout_dup = os.dup(stdout_fd)
        stderr_dup = os.dup(stderr_fd)

        devnull_fd = os.open(os.devnull, os.O_WRONLY)

        try:
            os.dup2(devnull_fd, stdout_fd)
            os.dup2(devnull_fd, stderr_fd)
            bpy.ops.render.render(animation=True)
        finally:
            os.dup2(stdout_dup, stdout_fd)
            os.dup2(stderr_dup, stderr_fd)
            os.close(stdout_dup)
            os.close(stderr_dup)
            os.close(devnull_fd)
            sys.stdout.flush()
            sys.stderr.flush()

    def _setup_ground_with_image(self, image_path, location=(0, 0, 0), scale=50.0,
                                  rotation=(0, 0, 0), use_shadow_catcher=False, tile_scale=1.0):
        """Set up a ground plane textured with a single tiled image.

        Args:
            image_path: image file path (jpg, png, ...)
            location: ground position
            scale: ground plane size
            rotation: ground rotation (rx, ry, rz)
            use_shadow_catcher: set as shadow catcher
            tile_scale: texture tiling factor (higher = more repeats,
                        tiles instead of stretching)

        Returns:
            ground object
        """
        if not os.path.exists(image_path):
            print(f"   warning: image not found: {image_path}")
            return None

        bpy.ops.mesh.primitive_plane_add(size=scale, location=location)
        ground_obj = bpy.context.active_object
        ground_obj.name = "Ground"

        if rotation != (0, 0, 0):
            ground_obj.rotation_euler = rotation

        # shadow catcher (version-dependent attribute)
        try:
            ground_obj.is_shadow_catcher = use_shadow_catcher  # Blender 3.x+
        except Exception:
            ground_obj.cycles.is_shadow_catcher = use_shadow_catcher  # Blender 2.x

        material = bpy.data.materials.new(name="GroundImageMaterial")
        material.use_nodes = True
        material.blend_method = "BLEND"
        try:
            material.shadow_method = "HASHED"  # Blender 3.x+
        except AttributeError:
            pass

        ground_obj.data.materials.append(material)
        ground_obj.active_material = material

        tree = material.node_tree
        nodes = tree.nodes
        links = tree.links

        # remove default nodes (keep Material Output)
        for node in list(nodes):
            if node.name != "Material Output":
                nodes.remove(node)

        # texture coordinates + mapping for tiling
        texcoord_node = nodes.new("ShaderNodeTexCoord")
        texcoord_node.location = (-1000, 0)

        mapping_node = nodes.new("ShaderNodeMapping")
        mapping_node.location = (-800, 0)
        mapping_node.inputs["Scale"].default_value = (tile_scale, tile_scale, tile_scale)
        # Generated coordinates make the texture repeat (tile) instead of stretch
        links.new(texcoord_node.outputs["Generated"], mapping_node.inputs["Vector"])

        image_node = nodes.new("ShaderNodeTexImage")
        image_node.location = (-600, 0)
        image_node.image = bpy.data.images.load(image_path)
        image_node.image.alpha_mode = "STRAIGHT"
        # REPEAT extension tiles the texture outside UV space
        image_node.extension = "REPEAT"
        image_node.interpolation = "Linear"
        links.new(mapping_node.outputs["Vector"], image_node.inputs["Vector"])

        if use_shadow_catcher:
            transparent_node = nodes.new("ShaderNodeBsdfTransparent")
            transparent_node.location = (0, 0)
            transparent_node.inputs["Color"].default_value = (1, 1, 1, 1)
            links.new(transparent_node.outputs["BSDF"], nodes["Material Output"].inputs["Surface"])
        else:
            bsdf_node = nodes.new("ShaderNodeBsdfPrincipled")
            bsdf_node.location = (0, 0)
            bsdf_node.inputs["Roughness"].default_value = 1.0
            links.new(image_node.outputs["Color"], bsdf_node.inputs["Base Color"])
            links.new(bsdf_node.outputs["BSDF"], nodes["Material Output"].inputs["Surface"])

        return ground_obj

    def _add_occluders(self, target_object=None):
        """Add procedural occluders according to the task configuration."""
        default_occluder_config = {
            'enabled': True,
            'min_count': 0,
            'max_count': 3,
            'shapes': [
                OccluderGenerator.SHAPE_CUBE,
                OccluderGenerator.SHAPE_SPHERE,
                OccluderGenerator.SHAPE_CYLINDER,
                OccluderGenerator.SHAPE_PLANE,
            ],
            'modes': [
                OccluderGenerator.MODE_FOREGROUND,
                OccluderGenerator.MODE_EDGE,
                OccluderGenerator.MODE_CORNER,
                OccluderGenerator.MODE_RANDOM,
            ],
            'materials': [
                OccluderGenerator.MATERIAL_SOLID,
                OccluderGenerator.MATERIAL_TRANSPARENT,
                OccluderGenerator.MATERIAL_METALLIC,
            ],
            'size_range': (0.15, 0.6),
            'transparency_range': (0.3, 0.8),
            'animate': False,  # the pipeline samples this (30%) and passes it in
            'depth_range': (0.25, 0.75),
        }

        occluder_config = {**default_occluder_config, **self.occluder_config}

        if not occluder_config.get('enabled', True):
            print("   occluders disabled")
            return

        scene = bpy.context.scene
        camera = scene.camera

        if not camera:
            print("   warning: no camera found, skipping occluders")
            return

        generator = OccluderGenerator(scene, camera, occluder_config)
        occluders = generator.generate_occluders(target_object)

        if occluders:
            print(f"   added {len(occluders)} occluders")
        else:
            print("   no occluders added")

    def _load_camera_from_npz(self, camera_npz_path):
        """Restore the camera from a trajectory .npz (per-frame keyframes)."""
        camera_data = np.load(camera_npz_path)
        location_list = camera_data['location']
        rotation_list = camera_data['rotation']
        focal_length = float(camera_data['focal_length'])
        sensor_width = float(camera_data['sensor_width'])
        sensor_height = float(camera_data['sensor_height'])

        # create or get the camera
        if "Camera" not in bpy.data.objects:
            camera_data_obj = bpy.data.cameras.new(name="Camera")
            camera_obj = bpy.data.objects.new("Camera", camera_data_obj)
            bpy.context.collection.objects.link(camera_obj)
        else:
            camera_obj = bpy.data.objects["Camera"]

        bpy.context.scene.camera = camera_obj

        if camera_obj.animation_data:
            camera_obj.animation_data_clear()

        camera_obj.data.lens = focal_length
        camera_obj.data.sensor_width = sensor_width
        camera_obj.data.sensor_height = sensor_height

        resolution_x = self.render_config.get('res_x', 1024)
        resolution_y = self.render_config.get('res_y', 1024)
        if resolution_x > resolution_y:
            camera_obj.data.sensor_fit = 'HORIZONTAL'
        else:
            camera_obj.data.sensor_fit = 'VERTICAL'

        num_frames = len(location_list)
        for frame_idx in range(num_frames):
            frame = bpy.context.scene.frame_start + frame_idx
            bpy.context.scene.frame_set(frame)

            camera_obj.location = location_list[frame_idx]
            camera_obj.rotation_euler = rotation_list[frame_idx]

            camera_obj.keyframe_insert(data_path="location", frame=frame)
            camera_obj.keyframe_insert(data_path="rotation_euler", frame=frame)

    def _clear_scene(self):
        """Clear the Blender scene and orphaned data blocks."""
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

        for block in bpy.data.meshes:
            if block.users == 0:
                bpy.data.meshes.remove(block)
        for block in bpy.data.materials:
            if block.users == 0:
                bpy.data.materials.remove(block)
        for block in bpy.data.textures:
            if block.users == 0:
                bpy.data.textures.remove(block)
        for block in bpy.data.images:
            if block.users == 0:
                bpy.data.images.remove(block)

    def _find_armature_and_mesh(self, obj_names):
        """Find the armature and mesh(es) among the imported objects."""
        armature = None
        mesh_object = None
        mesh_object_list = []

        for obj_name in obj_names:
            obj = bpy.data.objects[obj_name]
            if obj.type == 'ARMATURE' or (obj.animation_data and obj.animation_data.action):
                armature = obj
            if obj.type == 'MESH':
                if mesh_object is None:
                    mesh_object = obj
                mesh_object_list.append(obj)

        return armature, mesh_object, mesh_object_list

    def _setup_default_camera(self, mesh_object):
        """Fallback: fixed camera looking at the mesh center."""
        if mesh_object:
            world_bound_box = [mesh_object.matrix_world @ Vector(corner)
                               for corner in mesh_object.bound_box]
            min_x = min(corner.x for corner in world_bound_box)
            max_x = max(corner.x for corner in world_bound_box)
            min_y = min(corner.y for corner in world_bound_box)
            max_y = max(corner.y for corner in world_bound_box)
            min_z = min(corner.z for corner in world_bound_box)
            max_z = max(corner.z for corner in world_bound_box)

            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            center_z = (min_z + max_z) / 2

            camera = bpy.data.objects.get("Camera")
            if not camera:
                camera_data = bpy.data.cameras.new(name="Camera")
                camera = bpy.data.objects.new("Camera", camera_data)
                bpy.context.collection.objects.link(camera)

            bpy.context.scene.camera = camera

            distance = 2.5
            camera.location = (center_x, center_y - distance, center_z + 1.0)

            direction = Vector((center_x, center_y, center_z)) - camera.location
            rot_quat = direction.to_track_quat('-Z', 'Y')
            camera.rotation_euler = rot_quat.to_euler()

    def _setup_hdr_environment(self, hdr_path):
        """Set up an HDR environment world with strength 2.0."""
        scene = bpy.context.scene
        world = scene.world

        world.use_nodes = True
        node_tree = world.node_tree

        node_tree.nodes.clear()

        node_background = node_tree.nodes.new(type="ShaderNodeBackground")
        node_background.inputs["Strength"].default_value = 2.0

        node_environment = node_tree.nodes.new("ShaderNodeTexEnvironment")
        node_environment.image = bpy.data.images.load(hdr_path)
        node_environment.location = (-300, 0)

        node_output = node_tree.nodes.new(type="ShaderNodeOutputWorld")
        node_output.location = (200, 0)

        node_tree.links.new(node_environment.outputs["Color"],
                            node_background.inputs["Color"])
        node_tree.links.new(node_background.outputs["Background"],
                            node_output.inputs["Surface"])

    def _setup_default_lighting(self):
        """Fallback: sun light + white world background."""
        bpy.ops.object.light_add(
            type='SUN',
            location=(10., 0., 5.),
            rotation=(0., -np.pi / 4, 3.14)
        )

        sun_object = bpy.context.object
        sun_object.data.use_nodes = True
        sun_object.data.node_tree.nodes["Emission"].inputs["Strength"].default_value = 1.0

        scene = bpy.context.scene
        scene.world.use_nodes = True
        world_nodes = scene.world.node_tree.nodes
        world_links = scene.world.node_tree.links

        for node in world_nodes:
            world_nodes.remove(node)

        background_node = world_nodes.new(type='ShaderNodeBackground')
        output_node = world_nodes.new(type='ShaderNodeOutputWorld')

        background_node.inputs['Color'].default_value = (1, 1, 1, 1)
        background_node.inputs['Strength'].default_value = 1.0

        world_links.new(background_node.outputs['Background'],
                        output_node.inputs['Surface'])

    def _set_texture(self, mesh_obj, texture_dir):
        """Apply a single texture (first image found in the directory)."""
        material = self._create_texture_material(texture_dir, "TextureMaterial")
        if material:
            if mesh_obj.data.materials:
                mesh_obj.data.materials[0] = material
            else:
                mesh_obj.data.materials.append(material)

    def _create_texture_material(self, texture_dir, material_name):
        """Create a material with the first image found in texture_dir."""
        material = bpy.data.materials.new(name=material_name)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links

        bsdf = nodes["Principled BSDF"]

        diffuse_path = _first_image(texture_dir)
        if diffuse_path:
            diffuse_image = bpy.data.images.load(diffuse_path)
            diffuse_node = nodes.new("ShaderNodeTexImage")
            diffuse_node.image = diffuse_image
            diffuse_node.location = (-400, 300)
            links.new(diffuse_node.outputs["Color"], bsdf.inputs["Base Color"])

        return material

    def _set_multi_textures(self, mesh_obj_list, texture_dirs):
        """Apply multiple textures that switch during the video.

        Args:
            mesh_obj_list: mesh objects
            texture_dirs: list of texture directories
        """
        if not texture_dirs or len(texture_dirs) == 0:
            return

        num_textures = len(texture_dirs)
        frame_start = bpy.context.scene.frame_start
        frame_end = bpy.context.scene.frame_end
        total_frames = frame_end - frame_start + 1

        frames_per_texture = max(total_frames // num_textures, 1)

        print(f"   multi-texture mode: {num_textures} textures, {frames_per_texture} frames each")

        for mesh_obj in mesh_obj_list:
            self._setup_animated_materials(mesh_obj, texture_dirs, frame_start, frames_per_texture)

    def _setup_animated_materials(self, mesh_obj, texture_dirs, frame_start, frames_per_texture):
        """Build a node-based material that switches between textures.

        A keyframed Value node (constant interpolation) drives a chain of
        Mix Shader nodes, hard-switching to the next texture every
        frames_per_texture frames.
        """
        main_material = bpy.data.materials.new(name="AnimatedTextureMaterial")
        main_material.use_nodes = True
        nodes = main_material.node_tree.nodes
        links = main_material.node_tree.links

        nodes.clear()

        output_node = nodes.new(type='ShaderNodeOutputMaterial')
        output_node.location = (800, 0)

        # load all textures
        texture_nodes = []
        bsdf_nodes = []

        for idx, texture_dir in enumerate(texture_dirs):
            diffuse_path = _first_image(texture_dir)
            if diffuse_path:
                tex_node = nodes.new(type='ShaderNodeTexImage')
                tex_node.image = bpy.data.images.load(diffuse_path)
                tex_node.location = (-600, 300 - idx * 300)
                tex_node.label = f"Texture_{idx}"
                texture_nodes.append(tex_node)

                bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
                bsdf.location = (-200, 300 - idx * 300)
                bsdf.label = f"BSDF_{idx}"
                links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
                bsdf_nodes.append(bsdf)

        if len(bsdf_nodes) == 0:
            # no valid textures: plain default material
            bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf.location = (0, 0)
            bsdf.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1.0)
            links.new(bsdf.outputs['BSDF'], output_node.inputs['Surface'])

        elif len(bsdf_nodes) == 1:
            links.new(bsdf_nodes[0].outputs['BSDF'], output_node.inputs['Surface'])

        else:
            # multiple textures: chain of Mix Shaders driven by a
            # keyframed value node holding the current texture index
            frame_driver = nodes.new(type='ShaderNodeValue')
            frame_driver.location = (-800, 0)
            frame_driver.label = "TextureIndex"

            current_output = bsdf_nodes[0].outputs['BSDF']

            for idx in range(1, len(bsdf_nodes)):
                mix_shader = nodes.new(type='ShaderNodeMixShader')
                mix_shader.location = (200 + idx * 200, 0)
                mix_shader.label = f"Mix_{idx}"

                # factor = 1 if texture_index >= idx else 0
                compare_node = nodes.new(type='ShaderNodeMath')
                compare_node.operation = 'GREATER_THAN'
                compare_node.location = (0, -100 - idx * 100)
                compare_node.inputs[1].default_value = idx - 0.5  # threshold

                links.new(frame_driver.outputs['Value'], compare_node.inputs[0])
                links.new(compare_node.outputs['Value'], mix_shader.inputs['Fac'])
                links.new(current_output, mix_shader.inputs[1])
                links.new(bsdf_nodes[idx].outputs['BSDF'], mix_shader.inputs[2])

                current_output = mix_shader.outputs['Shader']

            links.new(current_output, output_node.inputs['Surface'])

            # keyframe the texture index over time
            frame_end = bpy.context.scene.frame_end

            for tex_idx in range(len(texture_dirs)):
                start_frame = frame_start + tex_idx * frames_per_texture
                end_frame = start_frame + frames_per_texture - 1

                if end_frame > frame_end:
                    end_frame = frame_end

                frame_driver.outputs['Value'].default_value = float(tex_idx)
                frame_driver.outputs['Value'].keyframe_insert('default_value', frame=start_frame)

                if tex_idx < len(texture_dirs) - 1:
                    next_start = frame_start + (tex_idx + 1) * frames_per_texture - 1
                    frame_driver.outputs['Value'].default_value = float(tex_idx)
                    frame_driver.outputs['Value'].keyframe_insert('default_value', frame=next_start)

            # constant interpolation = hard switches
            if main_material.node_tree.animation_data and main_material.node_tree.animation_data.action:
                for fcurve in main_material.node_tree.animation_data.action.fcurves:
                    for keyframe in fcurve.keyframe_points:
                        keyframe.interpolation = 'CONSTANT'

        # apply the material to the mesh
        if mesh_obj.data.materials:
            mesh_obj.data.materials[0] = main_material
        else:
            mesh_obj.data.materials.append(main_material)

        print(f"      applied animated material to {mesh_obj.name}")

    def _import_ground_plane(self, gltf_path=None, location=(0, 0, 0), scale=50.0):
        """Import a GLTF ground plane or create a simple plane."""
        if gltf_path and os.path.exists(gltf_path):
            bpy.ops.import_scene.gltf(filepath=gltf_path)

            ground_object = bpy.context.selected_objects[0] if bpy.context.selected_objects else None

            if ground_object:
                ground_object.location = location
                ground_object.scale = (scale, scale, scale)
                return ground_object

        bpy.ops.mesh.primitive_plane_add(size=scale, location=location)
        ground_object = bpy.context.active_object
        ground_object.name = "Ground"

        return ground_object

    def _setup_pbr_material_from_textures(self, mesh_obj, texture_dir, material_name="GroundMaterial"):
        """Build a PBR material from a texture directory.

        Recognizes common PBR naming conventions: basecolor/diffuse/albedo,
        normal, ORM (Occlusion-Roughness-Metallic, Unreal), MR
        (Metallic-Roughness, glTF), and separate roughness/metallic/AO maps.
        """
        if not os.path.exists(texture_dir):
            print(f"   warning: texture directory not found: {texture_dir}")
            return None

        texture_files = {}

        for filename in os.listdir(texture_dir):
            filepath = os.path.join(texture_dir, filename)
            if not os.path.isfile(filepath):
                continue

            filename_lower = filename.lower()

            # base color
            if '_b.' in filename_lower or 'basecolor' in filename_lower or 'diffuse' in filename_lower or 'albedo' in filename_lower:
                texture_files['base_color'] = filepath

            # normal
            elif '_n.' in filename_lower or 'normal' in filename_lower:
                texture_files['normal'] = filepath

            # ORM merged texture (Occlusion-Roughness-Metallic)
            elif '_orm.' in filename_lower or 'orm' in filename_lower:
                texture_files['orm'] = filepath

            # MR merged texture (Metallic-Roughness, glTF standard)
            elif '_mr.' in filename_lower or ('metallic' in filename_lower and 'roughness' in filename_lower):
                texture_files['mr'] = filepath

            # roughness
            elif '_r.' in filename_lower or 'roughness' in filename_lower:
                texture_files['roughness'] = filepath

            # metallic
            elif '_m.' in filename_lower or 'metallic' in filename_lower or 'metalness' in filename_lower:
                texture_files['metallic'] = filepath

            # occlusion
            elif '_ao.' in filename_lower or 'occlusion' in filename_lower or 'ambientocclusion' in filename_lower:
                texture_files['ao'] = filepath

        if not texture_files:
            print(f"   warning: no texture files found in {texture_dir}")
            return None

        material = bpy.data.materials.new(name=material_name)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links

        nodes.clear()

        output_node = nodes.new(type='ShaderNodeOutputMaterial')
        output_node.location = (400, 0)

        bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf_node.location = (0, 0)

        links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])

        # node layout cursor
        current_y = 300
        y_offset = 300

        def mix_into_base_color(second_input_socket):
            """Multiply the existing base color input by the given socket."""
            mix_node = nodes.new(type='ShaderNodeMixRGB')
            mix_node.blend_type = 'MULTIPLY'
            mix_node.inputs['Fac'].default_value = 1.0  # fully mixed
            mix_node.location = (-300, 300)

            base_color_source = None
            link_to_remove = None
            for link in list(links):
                if link.to_socket == bsdf_node.inputs['Base Color']:
                    base_color_source = link.from_socket
                    link_to_remove = link
                    break

            if link_to_remove:
                links.remove(link_to_remove)

            if base_color_source:
                links.new(base_color_source, mix_node.inputs['Color1'])
            links.new(second_input_socket, mix_node.inputs['Color2'])
            links.new(mix_node.outputs['Color'], bsdf_node.inputs['Base Color'])

        # 1. base color
        if 'base_color' in texture_files:
            base_color_node = nodes.new(type='ShaderNodeTexImage')
            base_color_node.image = bpy.data.images.load(texture_files['base_color'])
            base_color_node.location = (-600, current_y)
            links.new(base_color_node.outputs['Color'], bsdf_node.inputs['Base Color'])
            current_y -= y_offset

        # 2. normal map
        if 'normal' in texture_files:
            normal_tex_node = nodes.new(type='ShaderNodeTexImage')
            normal_tex_node.image = bpy.data.images.load(texture_files['normal'])
            normal_tex_node.image.colorspace_settings.name = 'Non-Color'
            normal_tex_node.location = (-600, current_y)

            normal_map_node = nodes.new(type='ShaderNodeNormalMap')
            normal_map_node.location = (-300, current_y)

            links.new(normal_tex_node.outputs['Color'], normal_map_node.inputs['Color'])
            links.new(normal_map_node.outputs['Normal'], bsdf_node.inputs['Normal'])
            current_y -= y_offset

        # 3. MR merged texture (glTF standard)
        if 'mr' in texture_files:
            mr_node = nodes.new(type='ShaderNodeTexImage')
            mr_node.image = bpy.data.images.load(texture_files['mr'])
            mr_node.image.colorspace_settings.name = 'Non-Color'
            mr_node.location = (-600, current_y)

            separate_node = nodes.new(type='ShaderNodeSeparateRGB')
            separate_node.location = (-300, current_y)
            links.new(mr_node.outputs['Color'], separate_node.inputs['Image'])

            links.new(separate_node.outputs['G'], bsdf_node.inputs['Roughness'])
            links.new(separate_node.outputs['B'], bsdf_node.inputs['Metallic'])

            current_y -= y_offset

        # 4. ORM merged texture (Unreal Engine standard)
        elif 'orm' in texture_files:
            orm_node = nodes.new(type='ShaderNodeTexImage')
            orm_node.image = bpy.data.images.load(texture_files['orm'])
            orm_node.image.colorspace_settings.name = 'Non-Color'
            orm_node.location = (-600, current_y)

            separate_node = nodes.new(type='ShaderNodeSeparateRGB')
            separate_node.location = (-300, current_y)
            links.new(orm_node.outputs['Color'], separate_node.inputs['Image'])

            links.new(separate_node.outputs['G'], bsdf_node.inputs['Roughness'])
            links.new(separate_node.outputs['B'], bsdf_node.inputs['Metallic'])

            # AO (R channel) is multiplied into the base color
            if 'base_color' in texture_files:
                mix_into_base_color(separate_node.outputs['R'])

            current_y -= y_offset

        else:
            # 5. separate roughness map
            if 'roughness' in texture_files:
                roughness_node = nodes.new(type='ShaderNodeTexImage')
                roughness_node.image = bpy.data.images.load(texture_files['roughness'])
                roughness_node.image.colorspace_settings.name = 'Non-Color'
                roughness_node.location = (-600, current_y)
                links.new(roughness_node.outputs['Color'], bsdf_node.inputs['Roughness'])
                current_y -= y_offset

            # 6. separate metallic map
            if 'metallic' in texture_files:
                metallic_node = nodes.new(type='ShaderNodeTexImage')
                metallic_node.image = bpy.data.images.load(texture_files['metallic'])
                metallic_node.image.colorspace_settings.name = 'Non-Color'
                metallic_node.location = (-600, current_y)
                links.new(metallic_node.outputs['Color'], bsdf_node.inputs['Metallic'])
                current_y -= y_offset

            # 7. separate AO map (multiplied into the base color)
            if 'ao' in texture_files and 'base_color' in texture_files:
                ao_node = nodes.new(type='ShaderNodeTexImage')
                ao_node.image = bpy.data.images.load(texture_files['ao'])
                ao_node.image.colorspace_settings.name = 'Non-Color'
                ao_node.location = (-600, current_y)

                mix_into_base_color(ao_node.outputs['Color'])

        if mesh_obj.data.materials:
            mesh_obj.data.materials[0] = material
        else:
            mesh_obj.data.materials.append(material)

        return material

    def _setup_ground_with_gltf_material(self, gltf_path=None, texture_dir=None,
                                         location=(0, 0, 0), scale=10.0,
                                         rotation=(0, 0, 0), use_shadow_catcher=False):
        """Set up a ground plane with a GLTF mesh and PBR textures.

        Args:
            gltf_path: GLTF file path (optional)
            texture_dir: PBR texture directory
            location: ground position
            scale: ground scale
            rotation: ground rotation (rx, ry, rz)
            use_shadow_catcher: set as shadow catcher

        Returns:
            ground object
        """
        ground_obj = self._import_ground_plane(gltf_path, location, scale)

        if not ground_obj:
            print("   error: failed to create ground object")
            return None

        if rotation != (0, 0, 0):
            ground_obj.rotation_euler = rotation

        if texture_dir and os.path.exists(texture_dir):
            self._setup_pbr_material_from_textures(ground_obj, texture_dir)

        if use_shadow_catcher:
            ground_obj.is_shadow_catcher = True

        return ground_obj

    def _auto_position_ground(self, ground_obj, character_obj, offset_z=0.0):
        """Reset the ground to the world origin (under the character)."""
        if not character_obj:
            return
        ground_obj.location.z = offset_z
        ground_obj.location.y = 0.0
        ground_obj.location.x = 0.0

    def _set_default_material(self, mesh_obj):
        """Plain white default material."""
        material = bpy.data.materials.new(name="DefaultMaterial")
        material.use_nodes = True
        nodes = material.node_tree.nodes

        bsdf = nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)

        if mesh_obj.data.materials:
            mesh_obj.data.materials[0] = material
        else:
            mesh_obj.data.materials.append(material)

    def _setup_renderer(self):
        """Configure the Cycles renderer."""
        bpy.app.debug_value = 0
        bpy.app.debug_wm = False
        bpy.app.debug_python = False

        scene = bpy.context.scene
        scene.render.engine = 'CYCLES'

        scene.render.film_transparent = False
        scene.view_layers[0].cycles.use_denoising = self.render_config.get('use_denoising', True)

        scene.cycles.samples = self.render_config.get('num_samples', 128)

        # persistent data reduces GPU-CPU transfers across frames
        scene.render.use_persistent_data = True

        # adaptive sampling speeds up convergence
        try:
            scene.cycles.use_adaptive_sampling = True
            scene.cycles.adaptive_threshold = 0.01
        except AttributeError:
            pass

        scene.cycles.max_bounces = 8
        scene.cycles.diffuse_bounces = 4
        scene.cycles.glossy_bounces = 2
        scene.cycles.transmission_bounces = 2
        scene.cycles.volume_bounces = 0
        scene.cycles.transparent_max_bounces = 2

        scene.cycles.caustics_reflective = False
        scene.cycles.caustics_refractive = False
        scene.cycles.blur_glossy = 0.0

        # optional motion blur with a random shutter
        if self.render_config.get('use_motion_blur', False):
            blur_shutter = random.uniform(0.2, 0.6)
            scene.render.motion_blur_shutter = blur_shutter
            scene.cycles.motion_blur_type = 'OBJECT'
            scene.cycles.motion_blur_position = 'CENTER'
            scene.cycles.use_motion_blur = True
            scene.cycles.motion_blur_samples = 16

        # GPU rendering
        if self.render_config.get('use_gpu', True):
            prefs = bpy.context.preferences.addons["cycles"].preferences

            try:
                prefs.compute_device_type = "CUDA"
            except Exception:
                pass

            prefs.get_devices()

            for device in prefs.devices:
                if device.type in ['CUDA', 'OPTIX', 'OPENCL', 'HIP']:
                    device.use = True
                elif device.type == 'CPU':
                    device.use = False

            scene.cycles.device = "GPU"

    def _set_output_properties(self, output_path):
        """Configure resolution and H.264 MP4 output."""
        scene = bpy.context.scene

        scene.render.resolution_percentage = 100
        scene.render.resolution_x = self.render_config.get('res_x', 1024)
        scene.render.resolution_y = self.render_config.get('res_y', 1024)

        scene.render.filepath = output_path
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"


def parse_args():
    """Parse command line arguments (after the `--` separator in Blender)."""
    parser = argparse.ArgumentParser(description="Blender render executor with procedural occluders")
    parser.add_argument('--task_config', type=str, required=True,
                        help="task config JSON file path (a <task_id>_meta.json written by render_pipeline.py)")

    if '--' in sys.argv:
        args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    else:
        args = parser.parse_args()

    return args


def main():
    """Load the task config and execute the render."""
    # disable Blender debug output
    bpy.app.debug_value = 0
    bpy.app.debug_wm = False
    bpy.app.debug_python = False

    args = parse_args()

    with open(args.task_config, 'r') as f:
        task_config = json.load(f)

    print(f"load task config: {args.task_config}")

    executor = BlenderRenderExecutor(task_config)
    result = executor.render()

    if result['status'] == 'success':
        print("render success")
    else:
        print(f"render failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()

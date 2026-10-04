# Render Pipeline Hyperparameters & Variables — Complete Reference

## 1. Pipeline Overview

```
render_pipeline.py
  ├── read the motion file list (filelist)
  ├── scan assets (HDR / textures / ground materials / floor images)
  └── for each motion and variant index:
       ├── 1. convert motion .npz -> .fbx (scripts/smplh2fbx.py, skipped if it exists)
       ├── 2. random camera type (+ optional shake)
       ├── 3. random HDR / texture / ground
       ├── 4. random resolution, copy motion into the task dir,
       │      call Blender to generate the camera trajectory -> _camera.npz
       ├── 5. sample occluders, write <task_id>_meta.json,
       │      call Blender to render -> .mp4
       └── 6. extract bbox annotations -> _bbox.npz (only if the render succeeded)
```

---

## 2. Hyperparameters by Layer

### 2.1 Pipeline layer — CLI arguments (`render_pipeline.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--raw_root` | (required) | root directory with the motion .npz/.fbx files |
| `--root` | (required) | output root directory for rendered tasks |
| `--filelist` | (required) | text file with one motion path per line |
| `--smpl_model_path` | `body_models/smplh/neutral/model.npz` | SMPL-H neutral model for bbox extraction |
| `--fbx_model_path` | = `--smpl_model_path` | body model for npz -> fbx (SMPL-H 52 / SMPL-X 55 joints) |
| `--obj_template` | `body_models/smplh_uv.obj` | OBJ with SMPL-H UVs, written into the FBX (required for skin textures) |
| `--fbx_dir` | None (next to the .npz) | where generated .fbx files are written |
| `--num_variants` | 1 | render variants per motion (task-id suffix range) |
| `--variant_start` | 0 | first variant index (for appending variants) |
| `--res_x` / `--res_y` | 0 / 0 | output resolution; 0 = random selection |
| `--num_samples` | 128 | Cycles sample count |
| `--denoising` | off | enable denoising |
| `--no_gpu` | off | render on CPU instead of GPU |
| `--hdr_dir` | `./assets/hdr_environments` | HDR environment directory |
| `--texture_dir` | `./assets/textures` | skin texture directory |
| `--ground_dir` | `./assets/grounds` | GLTF ground material directory |
| `--floor_image_dir` | `./assets/floors` | tiled floor image directory (fallback when no GLTF ground is available) |
| `--no_hdr` / `--no_texture` / `--no_ground` | off | disable an asset type even if its directory has files |
| `--occluder_prob` | 0.2 | probability that a task gets occluders |
| `--camera_format` | `_camera.npz` | camera file suffix |
| `--bbox_format` | `_bbox.npz` | bbox file suffix |
| `--blender` | `blender` | Blender executable (3.6+ recommended) |
| `--seed` | None | seed for pipeline-level randomness (reproducibility) |
| `--debug` | False | debug mode (renders into `--debug_output_dir`, draws bbox overlays) |

### 2.2 Random resolution selection

| Resolution | Probability | Aspect |
|------------|-------------|--------|
| 1920x1080 | 70% | 16:9 landscape |
| 1080x1920 | 15% | 9:16 portrait |
| 1080x1080 | 15% | 1:1 square |

### 2.3 Camera types and probability distribution

| Camera type | Probability | Description |
|-------------|-------------|-------------|
| `static` | 40% | fully static |
| `static_track` | 15% | fixed position, rotation follows the body |
| `random_track` | 15% | follows the body from a random direction |
| `front_track` | 5% | front tracking |
| `side_track` | 5% | side tracking |
| `orbit` | 5% | orbit around the body |
| `arc` | 5% | arc (orbit while increasing distance) |
| `push` | 5% | push in (far -> near) |
| `pull` | 5% | pull out (near -> far) |

> Note: the camera script (`camera_trajectory_generator.py`) keeps its own
> `movement_types` weight table, but it is only used when the pipeline passes
> `--movement_type random`. In normal operation the pipeline-level
> distribution above is authoritative.

### 2.4 Camera shake

| Parameter | Value | Description |
|-----------|-------|-------------|
| shake probability | 20% | only for non-`static` types |
| position frequency `freq_loc` | 0.08 | Perlin noise frequency |
| rotation frequency `freq_rot` | 0.12 | Perlin noise frequency |
| position amplitude `amp_loc` | 0.02 m (`shake_intensity`) | meters |
| rotation amplitude `amp_rot` | 0.5 deg (`shake_rotation_intensity`) | degrees |

### 2.5 Motion blur

| Parameter | Value | Description |
|-----------|-------|-------------|
| enable probability | 80% | `select_motion_blur()` |
| shutter range | [0.2, 0.6] | `random.uniform(0.2, 0.6)` |
| type | OBJECT | object motion blur |
| position | CENTER | center sampling |
| samples | 16 | motion blur samples |

### 2.6 Camera trajectory generator internals (`CameraTrajectoryGenerator.cfg`)

#### Distance control
| Parameter | Value | Description |
|-----------|-------|-------------|
| `min_distance` | 1.5 m | minimum camera distance |
| `max_distance` | 30.0 m | maximum camera distance |

#### Height control
| Category | Range (absolute, m) | Probability |
|----------|---------------------|-------------|
| `low_angle` | [0.3, 1.2] | 15% |
| `eye_level` | [1.2, 1.8] | 60% |
| `high_angle` | [1.8, 3.5] | 20% |
| `aerial` | [3.5, 5.0] | 5% |
| `ground_level` | 0.0 | — |
| `min_camera_height` | 0.3 m | hard constraint |
| `max_camera_height` | 6.0 m | hard constraint |

Tracking trajectories use different (relative offset) height ranges:
| Category | Range |
|----------|-------|
| `low_angle` | [-0.5, 0] |
| `eye_level` | [0, 1] |
| `high_angle` | [1, 2] |
| `aerial` | [2, 2.5] |

#### Focal length selection
| Range (mm) | Probability | Equivalent field of view |
|------------|-------------|--------------------------|
| 18-32 | 25% | wide angle |
| 32-60 | 50% | standard |
| 60-100 | 15% | short telephoto |
| 100-200 | 10% | telephoto |

#### View angle preference (angle bias)
| Parameter | Value | Description |
|-----------|-------|-------------|
| `reduce_back_view` | True | reduce back views |
| `front_view_bias` | 0.55 | front probability |
| `side_view_bias` | 0.35 | side probability |
| `back_view_bias` | 0.10 | back probability |
| `angle_bias_strength` | 1.0 | bias strength (1.0 = full bias) |

#### Sensor parameters
| Parameter | Value |
|-----------|-------|
| `sensor_width` | 36 mm |
| `sensor_height` | 24 mm |
| `fov_margin` | 1.0 |

#### Motion bounds margin
| Parameter | Value | Description |
|-----------|-------|-------------|
| `margin_xy` | 0.1 m | horizontal padding added to the joint bounds |
| `margin_z` | 0.1 m | vertical padding added to the joint bounds |
| global bounds scale | 1.1 | applied to Z for landscape, XY for portrait/square |

#### Trajectory-specific parameters

- **Orbit / Arc**: `total_rotation` depends on clip length
  - < 120 frames: [pi/6, pi/3]
  - < 360 frames: [pi/6, pi/2]
  - >= 360 frames: [pi/6, pi]
  - hard cap: `total_frames * pi / 360`
  - random direction (CW / CCW)

- **Push / Pull**: distance factor `dis_factor` in [1.2, 1.6];
  push goes from `safe_distance * dis_factor` to `safe_distance * 1.0`,
  pull from `safe_distance * 1.0` to `safe_distance * dis_factor`

- **Static track / random track**: distance multiplied by a random factor
  in [1.25, 1.5] or [1.15, 1.4]

- **Side track**: side angle offset in [pi*2/3, pi/2], random left/right

- **Height curves** (orbit/arc/push/pull): constant 50%, ramp up/down 30%,
  Perlin fluctuation 20%; short clips (< 60 frames) are forced constant

### 2.7 Occluder configuration (`OccluderGenerator`)

#### Configuration sampled by the pipeline
| Parameter | Value | Description |
|-----------|-------|-------------|
| `enabled` | True with probability `--occluder_prob` (20%) | master switch |
| `min_count` | 0 | minimum occluder count |
| `max_count` | 3 | maximum occluder count |
| `shapes` | cube, sphere, cylinder, plane, cone, torus | available shapes (monkey, ico_sphere, capsule also supported) |
| `modes` | foreground, edge, corner, random | placement modes (passing also supported) |
| `materials` | solid, transparent, metallic, glass | material types (emission, gradient, checker, noise also supported) |
| `size_range` | [0.15, 0.6] | size range |
| `transparency_range` | [0.3, 0.8] | alpha range |
| `animate` | 30% probability | enable animation |
| `depth_range` | [0.25, 0.75] | fraction of camera-target distance |

#### Geometry deformation
| Parameter | Value |
|-----------|-------|
| rotation | random [0, 2pi] per axis |
| scale variation | `scale_variation` in [0.7, 1.3], per axis in [0.5, 1.5] |

#### Animation types (when `animate` is on)
- `float`: bob up/down, amplitude [0.05, 0.2], frequency [0.5, 2.0]
- `rotate`: spin, speed [0.5, 2.0], random axis
- `wobble`: sway, location amplitude [0.02, 0.1], rotation amplitude [0.05, 0.2]
- `none`: no animation
- `passing` (MODE_PASSING): crosses the frame

### 2.8 Cycles renderer configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| engine | CYCLES | — |
| `num_samples` | 128 | sample count |
| `use_denoising` | False (default) | denoising |
| `use_persistent_data` | True | reduces GPU-CPU transfers |
| `use_adaptive_sampling` | True | adaptive sampling |
| `adaptive_threshold` | 0.01 | adaptive threshold |
| `max_bounces` | 8 | max light bounces |
| `diffuse_bounces` | 4 | diffuse bounces |
| `glossy_bounces` | 2 | glossy bounces |
| `transmission_bounces` | 2 | transmission bounces |
| `volume_bounces` | 0 | volume bounces |
| `transparent_max_bounces` | 2 | transparent bounces |
| `caustics_reflective` | False | reflective caustics |
| `caustics_refractive` | False | refractive caustics |
| `blur_glossy` | 0.0 | glossy blur |
| `film_transparent` | False | transparent film |
| output format | MPEG4 / H264 | — |

### 2.9 HDR environment

| Parameter | Value |
|-----------|-------|
| background strength | 2.0 |
| supported formats | .hdr, .exr |

### 2.10 Ground settings

Two ground sources are supported. Selection priority per task: a GLTF PBR
ground material (`--ground_dir`) if available, otherwise a tiled floor
image (`--floor_image_dir`).

GLTF ground material:

| Parameter | Value |
|-----------|-------|
| ground size `scale` | 50.0 |
| `use_shadow_catcher` | False |
| positioning | auto, at world origin (0, 0, 0) |

Tiled floor image:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `scale` | 50.0 | ground plane size |
| `tile_scale` | uniform [2.0, 6.0] | texture repeat factor (sampled per task) |
| texture extension | REPEAT | tiles instead of stretching |
| supported formats | .png / .jpg / .jpeg | single image per file |

### 2.11 Bbox extraction parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| SMPL model | `--smpl_model_path` (SMPL-H neutral) | configurable |
| `max_shape` | 10 | number of beta dimensions |
| `rescale` | 1.05 | bbox scale factor |
| `min_vis_joints` | 8 | minimum visible joints (of the first 22 body joints) |
| smoothing window | 5, two passes | moving average window |
| bbox clamp | [0, w] / [0, h] | clamped to image bounds |

---

## 3. Key Data Structures

### 3.1 task_config JSON (`<task_id>_meta.json`)

Written into every task directory by `render_pipeline.py` and passed to
`scripts/render_in_blender.py`. Top-level keys: `task_id`, `fbx_path`,
`hdr_file`, `texture_dir`, `ground_material`, `ground_image`,
`output_path`, `camera_type`, `focal_length`, `camera_params_path`,
`config` (`res_x`, `res_y`, `num_samples`, `use_denoising`, `use_gpu`,
`use_motion_blur`), `occluder_config` (see 2.7) and `length` (frames).

### 3.2 Camera NPZ contents

| Key | Shape | Description |
|-----|-------|-------------|
| `K` | (F, 3, 3) | intrinsics (complete mode, sensor-fit aware) |
| `RT` | (F, 3, 4) | world-to-camera extrinsics (z-up -> y-up conversion applied) |
| `location` | (F, 3) | camera location in Blender world space |
| `rotation` | (F, 3) | camera euler angles |
| `focal_length` | scalar | focal length (mm) |
| `sensor_width` | 36 | sensor width (mm) |
| `sensor_height` | 24 | sensor height (mm) |
| `movement_type` | string | trajectory type name |

### 3.3 Bbox NPZ contents

| Key | Shape | Description |
|-----|-------|-------------|
| `bbox` | (F, 4) | xyxy bounding box; invalid frames are [-1, -1, -1, -1] |
| `joints_world` | (F, J, 3) | joint positions in world coordinates |
| `joints_camera` | (F, J, 3) | joint positions in camera coordinates |
| `kp2d` | (F, J, 2) | projected 2D joint coordinates |
| `start_end` | (2,) | longest consecutive valid segment [start, end] |

---

## 4. Known Limitations

1. **Per-frame bounds computation**: tracking/orbit trajectories compute
   `calculate_segment_bounds(frame, frame)` once per frame to find the
   maximum required distance. Long clips pay an O(F) cost here; could be
   sampled at keyframes only.
2. **`--seed` only seeds the pipeline process**: the Blender subprocesses
   (occluders, shake, motion-blur shutter) use their own unseeded RNGs, so
   end-to-end reproducibility is not guaranteed. Record the
   `<task_id>_meta.json` of tasks you need to reproduce.
3. **Ground positioning** always resets the ground to the world origin and
   assumes the character performs around the origin.

---

## 5. Practical Notes

### 5.1 Body models and UVs

- `scripts/setup_body_models.sh` links the SMPL-H model to
  `body_models/smplh/neutral/model.npz` (default source:
  `../../assets/body_models/smplh/neutral/model.npz`; a SMPL-H `.pkl` can be passed
  instead and is converted) and exports the mesh UVs of the MANO SMPL-H FBX to
  `body_models/smplh_uv.obj` (`scripts/export_uv_obj.py`, run inside Blender).
- An OBJ exported elsewhere can be linked with `--uv_obj`. Only its UVs are read;
  faces are matched to the SMPL-H model by vertex index, so the OBJ must keep the
  SMPL-H vertex order and have 13776 triangles. Male and female SMPL-H FBX files share
  the same UVs.
- UVs are written into the `.fbx` when it is generated. If textures are used and an
  existing `.fbx` has no UVs, it is regenerated. If textures are found but the UV
  template is missing, the pipeline stops; pass `--no_texture` to render with a plain
  material instead.

### 5.2 GPU rendering

Cycles renders on the GPU by default. Blender 3.6 ships no prebuilt CUDA kernel for
some GPUs (e.g. A100, sm_80) and falls back to a PTX built with CUDA 12.1, which needs
an NVIDIA driver >= 530. On older drivers rendering fails with
`Unsupported PTX version`; use `--no_gpu`. When a Blender step fails, the last lines of
its output are written to the log.

### 5.3 Resuming and debugging

- Tasks whose `.mp4` already exists are skipped, so interrupted runs can be resumed.
- To parallelize, split the filelist and run one process per part.
- `--debug` renders into `--debug_output_dir` and writes a bbox overlay video;
  `scripts/vis_cam.py` overlays projected joints on a rendered video. Both need
  `ffmpeg` with `libx264` on `PATH`.

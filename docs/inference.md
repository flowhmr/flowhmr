# Command-Line Inference

Complete the [installation](install.md), then run from the repository root:

```bash
conda activate flowhmr
python run_v2m_demo_generation.py assets/example/example.mp4
```

Multiple files or directories are supported:

```bash
python run_v2m_demo_generation.py a.mp4 b.mov my_videos/
```

## Options

| Option | Default | Description |
|---|---|---|
| `--ckpt` | `checkpoints/flowhmr_latest/flowhmr_latest.ckpt` | Model checkpoint |
| `--model_cfg` | auto | Model YAML; uses `config.yml` beside the checkpoint |
| `--device` | `cuda` | Inference device, e.g. `cuda:0` |
| `--seed` | `0` | Random seed |
| `--cfg_scale` | `1.0` | Guidance scale |
| `--steps` | model config (`20`) | Sampling steps |
| `--detector` | `yolo26m` | Person detector: `yolo26m` or `yolox`, followed by ByteTrack |
| `--camera` | `pi4` | Camera backend: `vggt`, `pi4`, `simplevo`, or `static` |
| `--static_camera` | off | Alias for `--camera static` |
| `--camera_interval` / `--vggt_interval` | `1` | Sample every N frames with VGGT or Pi4 and interpolate the tracked frames |
| `--sam_chunk_size` | `16` | Frames per SAM body batch; hand batch is automatically twice this size |
| `--postprocess` | off | Enable RGBMocap contact planning and anatomical IK ([installation](install.md#optional-post-processing)) |
| `--overwrite` | off | Recompute cached preprocessing and features |

```bash
# Fixed camera
python run_v2m_demo_generation.py video.mp4 --static_camera

# Original detection and camera backends, sampled every 10 frames
python run_v2m_demo_generation.py video.mp4 --detector yolox --camera vggt --vggt_interval 10 --steps 20 --cfg_scale 1.0
```

YOLO26m and YOLOX-L detect the COCO person class. Pi4 estimates camera poses and metric translation
scale. It fits focal lengths and principal points to its two predicted ray channels on each
sampled frame. SAM uses these intrinsics in original-image pixels, with linear interpolation
between sampled frames. This path skips the depth output, depth refinement, point clouds and tracking.
SimpleVO follows GVHMR's half-resolution SIFT matching and
calibrated two-view estimation every eight frames, including the final frame. It
interpolates rotations, sets translation to zero and uses a 24 mm full-frame focal prior.
FlowHMR conditions on camera rotations relative to the first tracked frame.

```bash
python run_v2m_demo_generation.py video.mp4 --detector yolo26m --camera pi4 --camera_interval 8
# If camera estimation runs out of GPU memory, estimate camera rotation on the CPU.
python run_v2m_demo_generation.py video.mp4 --camera simplevo
```

The Web Settings expose the same **Person detector** and **Camera** choices. Each CLI
run updates the standard checkpoint-named motion file and keeps a backend-named copy
for comparisons, e.g. `flowhmr_latest_yolox_vggt.npz`.

## Acceleration

Defaults: SAM chunk **16** (body16/hand32), query batching **off**, compile **off**.
To use larger SAM batches:

```bash
python run_v2m_demo_generation.py video.mp4 --sam_chunk_size 32
```

Add `--overwrite` to recompute SAM features if the video already has cached results.

Set these environment variables before launching the CLI or [Web demo](web_demo.md):

| Variable | Default | Setting |
|---|---|---|
| `FLOWHMR_SAM_COMPILE` | `0` | `1` to compile the SAM backbone |
| `FLOWHMR_ATTENTION_BACKEND` | `original` | `query_batched` for long-sequence generation |
| `FLOWHMR_ATTENTION_QUERY_CHUNK_SIZE` | `256` | Queries per batch when `query_batched` is enabled |

```bash
FLOWHMR_SAM_COMPILE=1 FLOWHMR_ATTENTION_BACKEND=query_batched \
python run_v2m_demo_generation.py video.mp4
```

Compilation adds first-use startup time. Restart the Web service after changing
these variables. Model YAML attention settings take precedence over environment variables.

## Output

Results: `output/<video stem>/<checkpoint stem>.npz`, plus `.fbx` when the FBX SDK
is installed. Nonzero seeds add `_seed<N>` to the filename.

See [parameters](params.md) for configuration details and
[performance](performance.md) for timing and VRAM comparisons.

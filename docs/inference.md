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
| `--static_camera` | off | Skip VGGT for a fixed camera |
| `--vggt_interval` | `1` | Estimate camera every N frames; ignored with `--static_camera` |
| `--sam_chunk_size` | `16` | Frames per SAM body batch; hand batch is automatically twice this size |
| `--postprocess` | off | Enable RGBMocap contact planning and anatomical IK ([installation](install.md#optional-post-processing)) |
| `--overwrite` | off | Recompute cached preprocessing and features |

```bash
# Fixed camera
python run_v2m_demo_generation.py video.mp4 --static_camera

# Moving camera, sampled every 10 frames
python run_v2m_demo_generation.py video.mp4 --vggt_interval 10 --steps 20 --cfg_scale 1.0
```

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

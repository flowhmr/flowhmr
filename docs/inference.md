# Command-Line Inference

Copy this into your coding agent:

```text
Run FlowHMR on my video using the command-line pipeline. Read docs/install.md,
docs/inference.md, and docs/params.md. Use my video path if provided; otherwise
use assets/example/example.mp4. Prepare the V2M environment and required models,
run inference with flowhmr_latest, and verify that the output motion file exists.
Report the output path, processed videos, and any failures. Adjust the camera
frame interval if GPU memory is limited and explain the setting used.
```

## Run inference

Complete the V2M setup in [install.md](install.md). Run commands from the
**repository root**:

```bash
conda activate flowhmr
python run_v2m_demo_generation.py assets/example/example.mp4
```

The pipeline transcodes the video to 30 fps, detects and tracks the person,
estimates camera motion, extracts body features, and generates SMPL-H motion.
It accepts multiple video paths or directories of videos:

```bash
python run_v2m_demo_generation.py a.mp4 b.mov my_videos/
```

## Verify the output

For the bundled example and default checkpoint, the log should end with
`=== Done | ok=1 failed=0 ===`, and the result is:

```text
output/example/flowhmr_latest.npz
```

The motion contains `poses (T, 52, 3)` and `trans (T, 3)`. FBX is also exported
when the optional SDK is installed; see [install.md](install.md).
Extracted features are cached next to the result for reuse.

## Adjust settings

See [params.md](params.md#command-line-inference) for every CLI option,
including checkpoint selection, camera interval, caching, and optional
post-processing. Startup checks list missing model files; their expected
locations are in [resources.md](resources.md).

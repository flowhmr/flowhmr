# BlenderMotion

Render SMPL-H / SMPL-X motions into randomized videos with per-frame camera
parameters, 2D/3D keypoints and bounding boxes, using Blender.

```
motion.npz -> .fbx -> camera trajectory -> Blender Cycles render -> bbox / keypoints
```

Each video randomizes the camera trajectory, focal length, HDR environment, skin
texture, ground, resolution, motion blur and occluders. Arguments, distributions and
practical notes are in [docs/hyperparameters.md](docs/hyperparameters.md).

## Installation

1. [Blender](https://www.blender.org/download/releases/3-6/) 3.6+ (tested with 3.6.18).
2. Python 3.10 (the FlowHMR environment works):

   ```bash
   pip install -r requirements.txt
   pip install fbxsdkpy==2020.1.post2 --extra-index-url https://gitlab.inria.fr/api/v4/projects/18692/packages/pypi/simple
   ```

3. Body models (SMPL / MANO licenses, not included):
   - the SMPL-H neutral model from the FlowHMR repo
     ([docs/install.md](../../docs/install.md#41-body-models-manual));
   - `f_avg_noFlatHand.fbx` from the [MANO](https://mano.is.tue.mpg.de/) downloads, used
     for the skin texture UVs.

   ```bash
   cd data/rendering_data
   bash scripts/setup_body_models.sh --uv_fbx /path/to/f_avg_noFlatHand.fbx --blender /path/to/blender
   ```

## Quick start

From `data/rendering_data/`, download the example HDR environment
([Poly Haven](https://polyhaven.com/a/shanghai_bund), CC0, 21 MB), then render
(add `--blender /path/to/blender` if Blender is not on `PATH`):

```bash
wget -P examples/assets/hdr_environments \
    https://dl.polyhaven.org/file/ph-assets/HDRIs/exr/4k/shanghai_bund_4k.exr

python render_pipeline.py \
    --raw_root examples/motions \
    --filelist examples/filelist.txt \
    --root examples_output \
    --hdr_dir examples/assets/hdr_environments \
    --texture_dir examples/assets/textures \
    --floor_image_dir examples/assets/floors \
    --seed 0
```

Output:

```
examples_output/flowhmr_demo/flowhmr_demo_0000/
├── flowhmr_demo_0000.mp4
├── flowhmr_demo_0000_camera.npz   # K (F,3,3), RT (F,3,4), focal length, ...
├── flowhmr_demo_0000_bbox.npz     # bbox (F,4), joints_world / joints_camera, kp2d, start_end
├── flowhmr_demo_0000_meta.json    # render config
├── flowhmr_demo.fbx
└── flowhmr_demo.npz
```

Invalid frames have bbox `[-1, -1, -1, -1]`.

## Your own data

- **Motions**: `--filelist` lists `.npz` paths relative to `--raw_root`, one per line.
  Each file holds `poses` ((F, 156) SMPL-H or (F, 165) SMPL-X), `betas`, `trans` and
  optionally `mocap_framerate`.
- **Skin textures**: 1024x1024 maps in the SMPL UV layout, one directory per texture
  (e.g. `assets/textures/00000_00000/00000_00000.png`). We use
  [ATLAS](https://huggingface.co/datasets/ggxxii/ATLAS) (CC BY 4.0).
- **Environments and grounds**: `.hdr` / `.exr` files in `--hdr_dir`, floor images in
  `--floor_image_dir`.

Use `--num_variants N` to render each motion N times with different random draws.

## Training data

Rendered tasks plus SAM 3D Body features are the training samples. From the FlowHMR
repo root (FlowHMR environment, `ckpts/sam-3d-body-dinov3`, one GPU):

```bash
python data/rendering_data/scripts/make_webdataset.py \
    --render_root data/rendering_data/examples_output \
    --output /path/to/webdataset/synthetic/synthetic-%05d.tar \
    --save_features
```

This extracts features on the tracked frames and writes WebDataset shards for base
training ([format](../../docs/resources.md#webdataset-shards-base-training)).
`--save_features` also keeps `<task>_sam3d_feat.pt` in each task directory, which makes
the directory a GRPO sample. See [Prepare data](../../docs/training.md#0-prepare-data)
for the configs. Training expects SMPL-H motions at 30 fps.

## License

Copyright 2025-2026 FlowHMR authors. This directory is licensed under the
[Apache License 2.0](LICENSE), separately from the rest of the repository. `utils/lbs.py` (from [SMPL-X](https://github.com/vchoutas/smplx),
SMPL-X license) and `utils/geometry.py` (from
[PyTorch3D](https://github.com/facebookresearch/pytorch3d), BSD-3-Clause) keep their original
licenses.

Example assets in `examples/`, for testing the pipeline only:

| File | Source |
|------|--------|
| `motions/flowhmr_demo.npz` | captured from a video with FlowHMR |
| `assets/hdr_environments/shanghai_bund_4k.exr` | [Poly Haven](https://polyhaven.com/a/shanghai_bund), CC0; not bundled, downloaded in [Quick start](#quick-start) |
| `assets/floors/floor_gemini.png` | AI-generated |
| `assets/textures/atlas_00000_00000/00000_00000.png` | sample `00000_00000` of [ATLAS](https://huggingface.co/datasets/ggxxii/ATLAS), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), unmodified |

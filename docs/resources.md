# Resources and Data Layout

Copy this into your coding agent:

```text
Check the resources required for my FlowHMR task using docs/resources.md and
docs/install.md. Inspect the local files and separate resources needed for
V2M inference from those needed for PHC+ simulation or training.
Prepare the downloadable resources and tell me which files require my account
access, license acceptance, or local paths. Verify the final file locations.
```

All paths below are relative to the **repository root**. Download and
environment commands live in [install.md](install.md).

## Checkpoints and outputs

| Resource | Expected location |
|---|---|
| FlowHMR base / latest | `checkpoints/<name>/<name>.ckpt` and sibling `config.yml` |
| YOLOX detector | `ckpts/yolox/yolox_l.pth` |
| VGGT-Omega camera model | `ckpts/vggt-omega/vggt_omega_1b_512.pt` |
| SAM-3D-Body | `ckpts/sam-3d-body-dinov3/` with `model.ckpt`, `model_config.yaml`, and `assets/mhr_model.pt` |
| PHC composite controller | `output/HumanoidIm/phc_comp_3/Humanoid.pth` |
| PHC MCP controller | `output/HumanoidIm/phc_3/Humanoid.pth` |
| CLI results and cached features | `output/<video stem>/` |
| Web uploads and results | `output/web_jobs/<job_id>/` by default |

The PHC controller paths follow the simulator's existing `output/HumanoidIm/`
convention; these are downloaded inputs, unlike generated inference results.

## Body models

SMPL-family model downloads require the applicable licenses. The V2M and
simulation environments use different file layouts.

**V2M inference and training:**

```text
assets/body_models/
├── smplh/neutral/model.npz
└── smpl_neutral_J_regressor.pt
```

The SMPL-H neutral model comes from the AMASS `smplh_amass_neutral` release
([SMPL-H / MANO](https://mano.is.tue.mpg.de/)). It is used by the SMPL skeleton
and mesh implementations. The joint regressor is a public
[GVHMR asset](https://raw.githubusercontent.com/zju3dv/GVHMR/main/hmr4d/utils/body_model/smpl_neutral_J_regressor.pt).

**PHC simulation:**

```text
data/smpl/
└── SMPL_NEUTRAL.pkl / SMPL_MALE.pkl / SMPL_FEMALE.pkl
```

These SMPL v1.1.0 models are consumed by `smpl_sim` for the `smpl_humanoid` robot.
Follow [install.md](install.md#41-body-models-manual) for the required models for
each setup.

## Training inputs

Training data is not bundled. Both stages use the same per-sample content
(30 FPS, meters, world-to-camera extrinsics; `T` = number of frames):

| Item | Keys |
|---|---|
| motion `.npz` | `poses` (T, 156) or (T, 52, 3) SMPL-H axis-angle; `trans` (T, 3); `betas` (1, 16); optional `mocap_framerate` (must be 30) |
| camera `.npz` | `RT` (T, 4, 4) or (T, 3, 4) world-to-camera; `K` (T, 3, 3) intrinsics in pixels |
| bbox `.npz` | `bbox` (T, 4) `x1, y1, x2, y2` in pixels (extra columns such as a detection score are ignored); `start_end` (2,) = first and last tracked frame, inclusive |
| feature `.pt` | float tensor (`start_end[1] - start_end[0] + 1`, 3072): SAM 3D Body tokens of the tracked frames, extracted with [make_webdataset.py](../data/rendering_data/scripts/make_webdataset.py) (uses [runtime_sam_features.py](../flowhmr/utils/runtime_sam_features.py)) |

All per-frame arrays must have the same `T` (at least 10 frames). Samples that
fail these checks are skipped with a warning.

`data/rendering_data/` renders SMPL-H motions into synthetic samples in this format
and packs them into the inputs below; see [Prepare data](training.md#0-prepare-data).

### WebDataset shards (base training)

Set `tar_urls` in [configs/base/data.yml](../configs/base/data.yml). Each entry is
a shard pattern with brace expansion; add one entry per dataset:

```yaml
tar_urls:
  - "/path/to/webdataset/mydata/mydata-{00000..00025}.tar"   # 26 shards
  - "/path/to/webdataset/<dataset>/<dataset>-{00000..000NN}.tar"
```

Each shard is a plain tar file in which one sample is a group of files sharing a
key (the key must not contain `.`):

```text
mydata-00000.tar
├── 00000000_seq0001.motion.npz
├── 00000000_seq0001.camera.npz
├── 00000000_seq0001.bbox.npz
├── 00000000_seq0001.feature.pt
├── 00000000_seq0001.metadata.json   # optional, e.g. {"sequence_name": "seq0001"}
├── 00000001_seq0002.motion.npz
└── ...
```

Use about 1000 samples per shard; the epoch length is estimated as 1024 × the
number of shards. Shards can be written with `webdataset.ShardWriter`:

```python
import io
import numpy as np
import torch
import webdataset as wds

def npz_bytes(**arrays):
    buf = io.BytesIO(); np.savez(buf, **arrays); return buf.getvalue()

def pt_bytes(tensor):
    buf = io.BytesIO(); torch.save(tensor, buf); return buf.getvalue()

with wds.ShardWriter("mydata/mydata-%05d.tar", maxcount=1000) as sink:
    for i, s in enumerate(samples):  # your samples, keys as in the table above
        sink.write({
            "__key__": f"{i:08d}_{s['name']}",
            "motion.npz": npz_bytes(poses=s["poses"], trans=s["trans"], betas=s["betas"], mocap_framerate=30),
            "camera.npz": npz_bytes(RT=s["RT"], K=s["K"]),
            "bbox.npz": npz_bytes(bbox=s["bbox"], start_end=s["start_end"]),
            "feature.pt": pt_bytes(s["feature"]),
        })
```

Shards are read by
[train_dataset_raw.py](../flowhmr/datasets/v2m_generation/train_dataset_raw.py)
(`WDSV2MTrainDatasetRaw`).

### Sample list (GRPO)

Set `root` in [configs/grpo/data.yml](../configs/grpo/data.yml) to a `.txt` file
listing one sample directory per line (default `data/grpo_gt_pool/list.txt`).
For a directory named `<base>_<suffix>`:

```text
<base>_<suffix>/
├── <base>.npz                          # motion
├── <base>_<suffix>_camera.npz          # camera
├── <base>_<suffix>_bbox.npz            # bbox      (bbox_format)
└── <base>_<suffix>_sam3d_feat.pt       # feature   (feature_format)
```

The bbox and feature filenames follow `bbox_format` / `feature_format` in the
config. Render task directories from `data/rendering_data` use this layout.
Validation defaults to a subsample of the same list (`sample_step`); use a separate
list for held-out validation.

See [training.md](training.md) for the agent prompts and launch workflow.

## Bundled examples and datasets

`assets/example/` includes `example.mp4` for an inference smoke test. The PHC AMASS
shape statistics `amass_isaac_gender_betas{,_unique}.pkl` also go here; they are not
bundled and are downloaded with gdown (see [install.md](install.md#42-phc-checkpoints-full-pipeline-only)).

`data/wild4k/` contains the clip list of the Wild-4K evaluation set (4182 clips from
[Koala-36M](https://huggingface.co/datasets/Koala-36M/Koala-36M-v1)); videos are not
distributed. See [data/wild4k/README.md](../data/wild4k/README.md).

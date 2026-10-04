# Training

Copy this into your coding agent:

```text
Prepare a FlowHMR base pretraining run in my existing checkout.
Read docs/install.md, docs/training.md, docs/resources.md, and docs/params.md.
Inspect the available GPUs and ask for my training data paths if not provided.
Configure a separate experiment using configs/base as the starting point.
Check that the data and required model files are readable, then run a short
training smoke test before starting the full run. Report the command, config
changes, and output directory.
```

Set up the environments first ([install.md](install.md)): base training needs the
V2M env; GRPO additionally needs the PHC+ simulation env. Run all commands from
the **repository root**. The examples below use eight GPUs; adjust the launch
configuration to your machine. See [params.md](params.md#training) for options
and [resources.md](resources.md#training-inputs) for data locations.

## 0. Prepare data

Training data is not bundled. Each sample is a video with SMPL-H motion, per-frame
cameras, bounding boxes and SAM 3D Body features
([format](resources.md#training-inputs)). Bring your own data in that format, or
render synthetic data with [data/rendering_data](../data/rendering_data/README.md):

```bash
# render motions into videos with cameras and bounding boxes
cd data/rendering_data
python render_pipeline.py --raw_root /path/to/motions --filelist filelist.txt --root /path/to/renders ...
cd ../..

# extract SAM 3D Body features and write WebDataset shards (base training);
# --save_features also keeps <task>_sam3d_feat.pt next to each render (GRPO)
python data/rendering_data/scripts/make_webdataset.py \
    --render_root /path/to/renders \
    --output /path/to/webdataset/synthetic/synthetic-%05d.tar \
    --save_features
```

- **Base training** reads the shards: add the pattern to `tar_urls` in
  [configs/base/data.yml](../configs/base/data.yml).
- **GRPO** reads sample directories: each render task directory is one sample. List
  them in a `.txt` file and set `root` in [configs/grpo/data.yml](../configs/grpo/data.yml):

  ```bash
  find /path/to/renders -mindepth 2 -maxdepth 2 -type d > data/grpo_gt_pool/list.txt
  ```

## 1. Base model training

```bash
accelerate launch --config_file configs/accelerate/config_ddp.yaml \
    --num_processes 8 --main_process_port 29501 train_v2m.py \
    --model configs/base/model_basic05B.yml \
    --data  configs/base/data.yml \
    --train configs/base/train.yml \
    --name  output/base_model
```

Edit `configs/base/data.yml` to point to your WebDataset shards ([0. Prepare data](#0-prepare-data)).

## 2. GRPO post-training

Copy this into your coding agent:

```text
Prepare FlowHMR GRPO post-training following docs/training.md and the full
pipeline setup in docs/install.md. Start from configs/grpo and the base checkpoint.
Ask for my training pool path and simulation-environment Python if unavailable.
Configure the reward client to use that environment and verify the PHC+ smoke
test before a short GRPO training run. Report the configs, logs, and output path.
```

GRPO needs the PHC+ score server. It is launched **lazily and automatically** by the
reward client on the first rollout batch (one server per rank, on the rank's GPU) —
no manual step is required, but the *training process itself must be started in an
environment where Isaac Gym is importable*, or configured to reach a simulator
environment via `grpo.sim.phc_python` (see `configs/grpo/model.yml`).

```bash
accelerate launch --config_file configs/accelerate/config_ddp.yaml \
    --num_processes 8 --main_process_port 29501 train_v2m.py \
    --model configs/grpo/model.yml \
    --data  configs/grpo/data.yml \
    --train configs/grpo/train.yml \
    --name  output/grpo \
    --resume auto
```

Key hyperparameters and environment settings are listed in
[params.md](params.md#grpo-settings).

> **DDP note**: GRPO's second forward pass goes through `unwrap_model(...)` (because
> `grpo_one_step` is not `forward`), so torch DDP's gradient hooks do not fire —
> gradients are all-reduced manually in `V2MGRPOTrainer._sync_grads_manually`.

## 3. Simulator-as-reward architecture

```
V2M training proc (env A)                     PHC+ score server (env B, Isaac Gym)
┌──────────────────────────┐   bundle npz    ┌─────────────────────────────────┐
│ V2MGRPOTrainer           │ ──────────────► │ phc_score_server.py             │
│  ├─ rollout (SDE, 20 st) │  /dev/shm       │  ├─ convert_flowhmr_to_phc      │
│  ├─ PHCTrackingReward ───┼─ cmd_<id>.json► │  ├─ motion_lib reload + rollout │
│  └─ reward = MPJPE(GT)   │ ◄────────────── │  └─ {key: tracking_score,...}   │
│     + z(tracking_score)  │  scores.json    └─────────────────────────────────┘
└──────────────────────────┘  (cmd file deleted = done)
```

The server keeps Isaac Gym loaded (~30 s init) and scores each batch in seconds,
making simulator-in-the-loop RL practical. Isaac Gym's stdout pollution is avoided by
communicating through files only.

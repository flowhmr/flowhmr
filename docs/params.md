# Parameters and Configuration

Copy this into your coding agent:

```text
Help me configure FlowHMR for my task and available hardware. Read
docs/params.md and the relevant Web demo, inference, or training guide.
Check the current argument parser and YAML configs before changing settings.
Explain the relevant tradeoffs and give me the exact command or config changes.
If my task or hardware constraints are missing, ask for those details first.
```

Run commands from the **repository root**. The reference below follows the
current entry points and checked-in configurations. Use the source links when
checking a different revision.

## Web demo

Entry point: `python app.py`. Arguments are defined in
[tools/web/server.py](../tools/web/server.py).
Workflow: [web_demo.md](web_demo.md).

| Option | Default | Description |
|---|---|---|
| `--ckpt` | `checkpoints/flowhmr_latest/flowhmr_latest.ckpt` | FlowHMR checkpoint |
| `--model_cfg` | auto | Uses `config.yml` beside the checkpoint; otherwise `configs/base/model_basic05B.yml` |
| `--host` | `0.0.0.0` | Server bind address |
| `--port` | `8080` | Server port |
| `--device` | `cuda` | Device for the models, e.g. `cuda:0` |
| `--work_dir` | `output/web_jobs` | Uploads and job results |

Everything else is set per run in the page's **Settings** card: sampling seed
(`0`), CFG scale (`1.0`), flow-matching ODE steps (`20`), max frames at 30 fps
(`900` = 30 seconds), camera frame stride (`1`), manual person selection (on)
and post-processing (off).

```bash
python app.py --device cuda:0 --port 8080
```

## Command-line inference

Entry point and source: [run_v2m_demo_generation.py](../run_v2m_demo_generation.py).
Workflow: [inference.md](inference.md).

| Option | Default | Description |
|---|---|---|
| `videos` | required | One or more video paths or directories of videos |
| `--ckpt` | `checkpoints/flowhmr_latest/flowhmr_latest.ckpt` | FlowHMR checkpoint |
| `--model_cfg` | auto | Uses sibling `config.yml`; otherwise `configs/base/model_basic05B.yml` |
| `--device` | `cuda` | Device for the models |
| `--seed` | `0` | Sampling seed; nonzero seeds add `_seed<N>` to the result name |
| `--cfg_scale` | `1.0` | Classifier-free guidance scale (the Web option is named `--cfg`) |
| `--steps` | from config (`20` in the bundled config) | Override flow-matching ODE steps |
| `--postprocess` | off | Enable temporal smoothing and foot/ground-contact IK cleanup |
| `--vggt_interval` | `1` | Run camera estimation on every N-th frame |
| `--overwrite` | off | Recompute cached video preprocessing, detection, camera, and SAM features |

Results are saved under `output/<video stem>/<checkpoint stem>.npz` by default.
Use absolute paths for custom checkpoints and configuration files to avoid
ambiguity when the launcher changes to the repository root.

```bash
python run_v2m_demo_generation.py assets/example/example.mp4 \
    --ckpt checkpoints/flowhmr_base/flowhmr_base.ckpt --vggt_interval 10
```

## Camera settings

Both launchers accept `--vggt_interval`. Increasing it to `10` or `30` makes
VGGT-Omega process fewer frames and interpolates the camera between them. This
can help with long videos or limited GPU memory, but camera motion is sampled
less frequently; check the reconstruction quality for your input.

Other camera reconstruction approaches include
[ViPE](https://github.com/nv-tlabs/vipe),
[DA3-Streaming](https://github.com/ByteDance-Seed/Depth-Anything-3/tree/main/da3_streaming),
and [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM). These require a
separate integration; the current launchers use VGGT-Omega.

## Training

Entry point: [train_v2m.py](../train_v2m.py).
Agent prompts and launch commands: [training.md](training.md).

| Option | Default | Description |
|---|---|---|
| `--model` | required | Model/pipeline YAML |
| `--data` | required | Dataset YAML |
| `--train` | required | Training YAML |
| `--name` | derived from the three config filenames | Experiment/output directory; default under `output/v2m_train/` |
| `--postfix` | none | Suffix appended to the experiment name |
| `--resume` | none | State directory, checkpoint file, or `auto` to find a saved state/checkpoint in the experiment directory |

Configs are merged in order **model → data → train** using top-level dictionary
updates; a later top-level key replaces the earlier value. Keep custom
experiment configs separate from the supplied examples.

### Configuration groups

- [configs/base/](../configs/base/): `model_basic05B.yml` defines the 0.46B MMDiT
  and 491-dimensional motion pipeline; `data.yml` selects WebDataset shards;
  `train.yml` controls optimization.
- [configs/grpo/](../configs/grpo/): `model.yml` adds the GRPO trainer, rewards,
  and simulation settings; `data.yml` selects training/validation pools;
  `train.yml` controls optimization.
- [configs/accelerate/config_ddp.yaml](../configs/accelerate/config_ddp.yaml):
  supplied single-node, eight-GPU DDP configuration. Match `--num_processes`
  to the GPUs used for the run.

Defaults below come from the respective `train.yml` files:

| Setting | Base | GRPO |
|---|---|---|
| `train.max_epoch` | `25` | `8` |
| `train.train_iterations` | `10000` | `2500` |
| `train.batch_size` | `8` | `4` |
| `train.batch_size_val` | `1` | `1` |
| `train.lr` | `1e-4` | `2.5e-6` |
| `train.num_workers` | `4` | `2` |
| `train.scheduler` | `cosine_annealing` | `cosine_annealing` |

### GRPO settings

These settings live in [configs/grpo/model.yml](../configs/grpo/model.yml).

| Setting | Default | Purpose |
|---|---|---|
| `load_from_checkpoint` | `checkpoints/flowhmr_base/flowhmr_base.ckpt` | Base model initialization |
| `grpo.num_generations` | `8` | Samples per group |
| `grpo.sampling_steps` | `20` | Rollout steps |
| `grpo.inner_updates` | `4` | Optimization updates |
| `grpo.clip_range` | `1e-3` | GRPO clipping range |
| `grpo.mpjpe_weight` / `grpo.tracking_weight` | `1.0` / `1.0` | Reconstruction and physics reward weights |
| `grpo.sim.max_frames` | `180` | Motion length sent to the simulator |
| `grpo.sim.num_envs` | `32` | Parallel simulation environments |
| `grpo.sim.phc_root` | `null` | Configured as the repo root by default, containing `tools/phc_score_server.py` |
| `grpo.sim.phc_python` | `null` | Current interpreter; set an absolute path to the simulation env's Python when using separate envs |

`FLOWHMR_PHC_ROOT` overrides the **PHC submodule location** used by the score
server; this differs from `grpo.sim.phc_root`, which identifies the FlowHMR
directory containing the server script. Simulator GPU and environment notes
are in [install.md](install.md#3-simulation-environment-full-pipeline-only).

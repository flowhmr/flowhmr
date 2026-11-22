# Installation

Run all commands from the repository root, not from `docs/`. See [resources.md](resources.md)
for file locations and [params.md](params.md) for configuration options.

## Setups

| Setup | Supports | Steps |
|---|---|---|
| **V2M** | Inference, web demo, base model training | 1, 2, 4.1 (SMPL-H only), 4.3, 5.1–5.2 |
| **Full pipeline** | V2M + PHC+ physics simulator (GRPO post-training, physics tracking) | All |

The two Python environments never share imports; the training process and the PHC+
simulator communicate through files on `/dev/shm`.

Steps marked **[manual]** require license acceptance or account access and must be done by the user.

## 1. Clone

```bash
git clone https://github.com/flowhmr/flowhmr.git FlowHMR
cd FlowHMR
git submodule update --init --recursive -- third_party/sam-3d-body
```

## 2. V2M environment

```bash
conda create -n flowhmr python=3.10 -y
conda activate flowhmr
pip install -r requirements.txt -r tools/web/requirements.txt
```

Validated with Python 3.10, PyTorch 2.5.1, and CUDA driver ≥ 470. `requirements.txt` pins
the cu118 build of torch and torchvision.

Check: `python -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available())"`
prints `2.5.1+cu118 True`.

**Default detector and camera.** Install YOLO26m and Pi4 for video inference. The Pi4 setup
script installs its camera code. The download commands below fetch the model weights.
For the shared body feature extractor, request access to
[SAM 3D Body](https://huggingface.co/facebook/sam-3d-body-dinov3), then run
`huggingface-cli login`.

```bash
huggingface-cli download Ultralytics/YOLO26 yolo26m.pt --local-dir ckpts/yolo26m
python tools/setup_pi4_camera.py
huggingface-cli download yyfz233/Pi4 model.safetensors --local-dir ckpts/pi4
huggingface-cli download facebook/sam-3d-body-dinov3 --local-dir ckpts/sam-3d-body-dinov3
```

```
ckpts/
├── yolo26m/yolo26m.pt
├── pi4/model.safetensors
└── sam-3d-body-dinov3/{model.ckpt, model_config.yaml, assets/mhr_model.pt}
```

### Optional detection and camera backends

The default combination is YOLO26m with Pi4. Both the detector and camera can be
selected independently. Existing YOLOX and VGGT installations can use the original
pipeline with `--detector yolox --camera vggt`.

| Backend | Preparation |
|---|---|
| YOLOX-L | Run `git submodule update --init --recursive -- third_party/YOLOX`, then `pip install --no-build-isolation --no-deps -e third_party/YOLOX`. Place [yolox_l.pth](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth) in `ckpts/yolox/`. |
| VGGT-Omega | Run `git submodule update --init --recursive -- third_party/vggt-omega`. Request access to [VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega) and download `vggt_omega_1b_512.pt` to `ckpts/vggt-omega/`. |
| SimpleVO | If GPU memory is limited, use `--camera simplevo` to estimate camera rotation on the CPU. Install its optional dependency with `pip install pycolmap==3.12.6`. [GVHMR SimpleVO](https://github.com/zju3dv/GVHMR/tree/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/utils/preproc/relpose) |

**Optional: FBX export** (verified on Python 3.10 only). Without it, `.npz` results are
still saved and the `.fbx` export is skipped.

```bash
pip install fbxsdkpy==2020.1.post2 --extra-index-url https://gitlab.inria.fr/api/v4/projects/18692/packages/pypi/simple
```

### Optional post-processing

Install the bundled [RGBMocap correction](../third_party/RGBMocapCorrection) and its
[Kimodo native solver](../third_party/KimodoMotionCorrection). The native solver
requires a C++17 compiler and CMake ≥ 3.15. Enable correction with CLI `--postprocess`
or the Web **Post-processing** toggle (both off by default).

```bash
pip install pybind11==2.11.1
pip install --no-build-isolation --no-deps ./third_party/KimodoMotionCorrection
pip install --no-build-isolation --no-deps ./third_party/RGBMocapCorrection
```

CMake uses an installed Eigen ≥ 3.3 or downloads Eigen 3.4.0. For an offline build,
prefix the second command with `CMAKE_ARGS="-DEigen3_DIR=/path/to/eigen/share/eigen3/cmake"`.

Uses RGBMocap's **normal**, **flat ground**, **automatic floor**, **foot lock on**,
**hand lock off** defaults, including contact planning, pelvis correction and anatomical IK.

## 3. Simulation environment (full pipeline only)

**Apply the PHC+ patches** once:

```bash
git submodule update --init --recursive -- third_party/PerpetualHumanoidControl
cd third_party/PerpetualHumanoidControl
git apply ../patches/*.patch
cd ../..
```

| Patch (`third_party/patches/`) | Change |
|---|---|
| `0001-smpl_humanoid-axes` | SMPL humanoid MJCF axis convention |
| `0002-base_task-headless-render` | headless rendering + automatic video recording |
| `0003-draw_utils-matplotlib-compat` | matplotlib >= 3.7 compatibility |
| `0004-amass-stats-in-example-dir` | read AMASS shape stats from `assets/example/` |

Check: `(cd third_party/PerpetualHumanoidControl && git apply --check --reverse ../patches/*.patch)`
succeeds.

**Install Isaac Gym [manual].** NVIDIA Isaac Gym Preview (Linux only) is required. Install
it into its own Python environment following NVIDIA's official instructions. We validated
Isaac Gym Preview 4 with Python 3.8 and PyTorch 1.13 (cu117). Then, in that environment
(`phc` below):

```bash
pip install -r requirements-phc.txt
pip install --no-build-isolation git+https://github.com/mattloper/chumpy
```

This installs `smpl_sim` and `smplx` from their public repos, plus the PHC dependencies.
`requirements-phc.txt` constrains `torch<2.0` so the Isaac Gym PyTorch is kept. `chumpy`
is needed to load the SMPL `.pkl` files and must be installed without build isolation.

Check: `python -c "from isaacgym import gymapi; import torch; from isaacgym import gymtorch; import rl_games, smpl_sim"`.
`gymtorch` compiles an extension on first import. If it fails with `Ninja is required`,
run `pip install ninja`.

**GPU notes**

- Launch the score server and the evaluator with `--device-id <physical GPU>`. Do not
  remap GPUs with `CUDA_VISIBLE_DEVICES`, because PhysX may deadlock silently.
- The reward client adds the simulator Python's `bin/` directory to `PATH` automatically.
- For GRPO, set `grpo.sim.phc_python` in `configs/grpo/model.yml` to the `phc`
  environment's Python (see [training.md](training.md)).

## 4. Data

### 4.1 Body models [manual]

Download the SMPL-family models after registering:

- **V2M**: SMPL-H neutral from [MANO](https://mano.is.tue.mpg.de/) (AMASS `smplh_amass_neutral`).
- **Full pipeline**: also [SMPL](https://smpl.is.tue.mpg.de/) v1.1.0 under `data/smpl/`, read by
  `smpl_sim`. Rename `basicmodel_{neutral,m,f}_lbs_10_207_0_v1.1.0.pkl` to
  `SMPL_{NEUTRAL,MALE,FEMALE}.pkl`. SMPL-X is not needed (the simulator uses the `smpl_humanoid` robot).

The joint regressor is public:

```bash
wget -O assets/body_models/smpl_neutral_J_regressor.pt \
    https://raw.githubusercontent.com/zju3dv/GVHMR/main/hmr4d/utils/body_model/smpl_neutral_J_regressor.pt
```

```
assets/body_models/                 # V2M, required
├── smplh/neutral/model.npz
└── smpl_neutral_J_regressor.pt

data/smpl/                          # full pipeline only
└── SMPL_{NEUTRAL,MALE,FEMALE}.pkl
```

### 4.2 PHC+ checkpoints (full pipeline only)

```bash
pip install gdown
mkdir -p output/HumanoidIm/phc_comp_3 output/HumanoidIm/phc_3
gdown "https://drive.google.com/uc?id=1JbK9Vzo1bEY8Pig6D92yAUv8l-1rKWo3" -O output/HumanoidIm/phc_comp_3/Humanoid.pth  # ~400 MB
gdown "https://drive.google.com/uc?id=1pS1bRUbKFDp6o6ZJ9XSFaBlXv6_PrhNc" -O output/HumanoidIm/phc_3/Humanoid.pth       # ~419 MB
# PHC AMASS shape statistics, loaded by the simulator
gdown "https://drive.google.com/uc?id=1bLp4SNIZROMB7Sxgt0Mh4-4BLOPGV9_U" -O assets/example/   # amass_isaac_gender_betas.pkl, ~3 MB
gdown "https://drive.google.com/uc?id=1arpCsue3Knqttj75Nt9Mwo32TKC4TYDx" -O assets/example/   # amass_isaac_gender_betas_unique.pkl, ~70 KB
```

- `phc_comp_3`: composite controller, loaded by the tracking evaluator and the GRPO reward server.
- `phc_3`: MCP controller, referenced by `env.models`.
- `amass_isaac_gender_betas{,_unique}.pkl`: AMASS shape statistics from the PHC release
  (same files as PHC's `download_data.sh`), subject to the AMASS license.

If Google Drive rate-limits the download, fetch the files manually from the PHC release.

### 4.3 FlowHMR checkpoints

```bash
huggingface-cli download fafsaf1/flowhmr-0.46B --include "flowhmr_latest/*" "config.json" --local-dir checkpoints
```

For training or inference with the base checkpoint, also download it with
`huggingface-cli download fafsaf1/flowhmr-0.46B --include "flowhmr_base/*" --local-dir checkpoints`.

```
checkpoints/
├── flowhmr_base/{flowhmr_base.ckpt, config.yml}
└── flowhmr_latest/{flowhmr_latest.ckpt, config.yml}
```

Keep each `.ckpt` next to its `config.yml`. Pass the `.ckpt` path via `--ckpt`, and
`config.yml` is loaded automatically. These `config.yml` files are meant for
inference only; for training, use the configs in `configs/` (see
[training.md](training.md)).

## 5. Smoke tests

### 5.1 Inference

Runs detection, camera estimation, SAM 3D Body features, and FlowHMR on the bundled example:

```bash
conda activate flowhmr
python run_v2m_demo_generation.py assets/example/example.mp4
```

Expected: the log ends with `=== Done | ok=1 failed=0 ===` and the script writes
`output/example/flowhmr_latest.npz` (`poses (T, 52, 3)`, `trans (T, 3)`). If any
checkpoint or body model is missing, the script lists it and exits. On smaller GPUs,
use `--camera_interval 10` to sample fewer frames (see [params.md](params.md)).

### 5.2 Web demo

```bash
python app.py --port 8080 --device cuda:0
```

Expected: after about 1 minute, the log prints `FlowHMR web demo ready`. Open
`http://<host>:8080` and upload `assets/example/example.mp4`. See [web_demo.md](web_demo.md)
and [params.md](params.md#web-demo) for launch options.

### 5.3 Simulator (full pipeline only)

Scores the 5.1 output with PHC+, the same way the GRPO reward does. Pick a free GPU for
`--device-id`:

```bash
conda activate phc
python tools/phc_score_server.py --ctrl-dir /tmp/flowhmr_phc_smoke --num-envs 4 \
    --device-id 0 --idle-timeout 120 > /tmp/flowhmr_phc_smoke.log 2>&1 &
# wait until /tmp/flowhmr_phc_smoke/ready exists (~30 s), then:
python tools/phc_smoke_test.py --ctrl-dir /tmp/flowhmr_phc_smoke \
    --motion output/example/flowhmr_latest.npz
```

Expected: a line like `{"example": {"tracking_score": 0.77..., "terminated": false,
"survival": 1.0, ...}}` followed by `PHC_SMOKE_OK`. `PHC_SMOKE_FAIL` or a crash in
`/tmp/flowhmr_phc_smoke.log` means a previous step is incomplete. The server exits
automatically after `--idle-timeout` seconds.

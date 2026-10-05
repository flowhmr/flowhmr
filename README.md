<div align="center">
<a href="https://flowhmr.github.io/"><img src="assets/FlowHMR-logo-tagline.svg" alt="FlowHMR: Physically Plausible Motion Capture from Video" width="880"></a>

<a href="https://arxiv.org/abs/2610.03691"><img src="https://img.shields.io/badge/arXiv-2610.03691-b31b1b" alt="arXiv"></a>
<a href="https://flowhmr.github.io/"><img src="https://img.shields.io/badge/Project_Page-FlowHMR-green" alt="Project Page"></a>
<a href="https://huggingface.co/fafsaf1/flowhmr-0.46B"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue" alt="Hugging Face Model"></a>
</div>

## Release Plan

We are releasing the full workflow, from data synthesis to training and evaluation, step by step.

- [x] Inference code and checkpoints ([guide](docs/inference.md), [models](#model-zoo))
- [x] Pretraining and GRPO post-training code ([guide](docs/training.md))
- [x] Web demo ([guide](docs/web_demo.md))
- [x] Synthetic data generation code ([guide](data/rendering_data/README.md))
- [x] Wild-4K evaluation clip list ([list](data/wild4k/README.md))
- [ ] Hugging Face demo

## Model Zoo

| Model | Description | Params | Tracking Success ↑ | Download |
|---|---|:---:|:---:|:---:|
| `flowhmr_base` | Base model | 0.46B | 78.79% | [🤗 HF](https://huggingface.co/fafsaf1/flowhmr-0.46B/tree/main/flowhmr_base) |
| `flowhmr_latest` | `flowhmr_base` + GRPO with PHC+ physics reward **(recommended)** | 0.46B | **82.47%** | [🤗 HF](https://huggingface.co/fafsaf1/flowhmr-0.46B/tree/main/flowhmr_latest) |

*Tracking Success*: the rate at which the PHC+ controller successfully tracks the predicted motion in simulation, measured on Wild-4K.

## Getting Started

Follow the guide for each step, or paste the prompt into your coding agent.

### Installation

[Installation guide](docs/install.md)

FlowHMR supports two setups:

- **V2M**: inference, web demo, and base model training in a single environment
  (tested with Python 3.10 + PyTorch 2.5.1, cu118).
- **Full pipeline**: V2M plus the PHC+ physics simulator for GRPO post-training and
  physics tracking. The simulator runs in a separate Isaac Gym environment (tested with
  Isaac Gym Preview 4, Python 3.8 + PyTorch 1.13, cu117). The two environments share no
  imports and communicate through files on `/dev/shm`.

Some assets require license acceptance or gated access: the SMPL-family body models, VGGT-Omega, and SAM 3D Body.

V2M:

```text
Install FlowHMR from https://github.com/flowhmr/flowhmr following docs/install.md with the V2M setup.
```

Full pipeline:

```text
Install FlowHMR from https://github.com/flowhmr/flowhmr following docs/install.md with the full pipeline setup.
```

### Web Demo

[Web demo guide](docs/web_demo.md)

Upload a video in the browser and reconstruct its 3D motion; when several people are
detected, pick the one to reconstruct. Requires the V2M setup.

```text
Launch the FlowHMR web demo following docs/web_demo.md.
```

### Inference

[Inference guide](docs/inference.md)

Reconstruct motion from a video on the command line. The pipeline runs person detection
(YOLOX), camera estimation (VGGT-Omega), body feature extraction (SAM 3D Body), and
FlowHMR, and saves the motion as `.npz` (plus `.fbx` if FBX export is installed).

```text
Run FlowHMR on my video following docs/inference.md.
```

### Data

[Data guide](data/README.md)

- **Training data** is not bundled. Bring your own in the
  [documented format](docs/resources.md#training-inputs), or render synthetic videos
  from SMPL-H motions with the Blender toolkit in [data/rendering_data](data/rendering_data/README.md).
- **Wild-4K** is the 4182-clip in-the-wild evaluation set behind Tracking Success.
  We release the [clip list](data/wild4k/README.md) (clip IDs from Koala-36M-v1),
  not the videos.

### Training

[Training guide](docs/training.md)

Training has two stages: pretraining the base model, then GRPO post-training with a physics reward.
Prepare training data first (see [Data](#data) and [Prepare data](docs/training.md#0-prepare-data)).

**Pretraining** (V2M setup) trains the flow-matching base model, `flowhmr_base`.

```text
Train the FlowHMR base model following docs/training.md.
```

**GRPO post-training** (full pipeline setup) fine-tunes `flowhmr_base` with the PHC+
simulator as the reward: each generated motion is scored by how well the PHC+ controller
tracks it. This produces `flowhmr_latest`.

```text
Run FlowHMR GRPO post-training following docs/training.md.
```

## Acknowledgements

We thank the authors of these projects for releasing their code and models:

- [HY-Motion 1.0](https://github.com/Tencent-Hunyuan/HY-Motion-1.0): MMDiT flow-matching backbone
- [MixGRPO](https://github.com/Tencent-Hunyuan/MixGRPO): GRPO post-training for flow matching
- [PHC](https://github.com/ZhengyiLuo/PerpetualHumanoidControl): PHC+ controller for physics reward and tracking evaluation
- [GVHMR](https://github.com/zju3dv/GVHMR): world-grounded human motion recovery
- [VGGT-Omega](https://github.com/facebookresearch/vggt-omega): camera estimation
- [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body): body features
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX): person detection

## Citation

If you find FlowHMR useful in your research, please cite:

```bibtex
@misc{wang2026flowhmr,
      title={FlowHMR: Physically Plausible Motion Capture from Video},
      author={Zhanke Wang and Chengfeng Zhao and Qing Shuai and Jingzhong Lin and Heng Li and Zeyu Ling and Yuxin Wen and Jing Li and Di Kang and Chunchao Guo and Linchao Bao},
      year={2026},
      eprint={2610.03691},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.03691},
}
```

"""FlowHMR inference from raw videos: video -> SMPL-H motion (npz, + fbx if fbxsdkpy is installed).

    python run_v2m_demo_generation.py assets/example/example.mp4
    python run_v2m_demo_generation.py a.mp4 b.mov my_videos/ --ckpt checkpoints/flowhmr_base/flowhmr_base.ckpt

Every video goes through the full pipeline: 30 fps transcode -> YOLOX person detection &
tracking -> VGGT-Omega camera -> SAM-3D-Body tokens -> FlowHMR generation. Results land in
output/<video name>/<ckpt name>.npz; the extracted features are kept next to it and reused
when the same video is run again (e.g. with another checkpoint). --overwrite recomputes them.
"""
import argparse
import glob
import os
import shutil
import sys
import time

import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

DEFAULT_CKPT = os.path.join(REPO_ROOT, "checkpoints", "flowhmr_latest", "flowhmr_latest.ckpt")
DEFAULT_MODEL_CFG = os.path.join(REPO_ROOT, "configs", "base", "model_basic05B.yml")
OUTPUT_ROOT = os.path.join(REPO_ROOT, "output")
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
REQUIRED_FILES = [
    "ckpts/yolox/yolox_l.pth",
    "ckpts/vggt-omega/vggt_omega_1b_512.pt",
    "ckpts/sam-3d-body-dinov3/model.ckpt",
    "ckpts/sam-3d-body-dinov3/assets/mhr_model.pt",
    "assets/body_models/smplh/neutral/model.npz",
    "assets/body_models/smpl_neutral_J_regressor.pt",
]


def collect_videos(inputs):
    videos = []
    for path in inputs:
        if os.path.isdir(path):
            videos += sorted(p for p in glob.glob(os.path.join(path, "*"))
                             if p.lower().endswith(VIDEO_EXTS))
        elif os.path.isfile(path):
            videos.append(path)
        else:
            sys.exit(f"not found: {path}")
    if not videos:
        sys.exit("no input videos found")
    return [os.path.abspath(v) for v in videos]


def check_files(ckpt):
    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(REPO_ROOT, f))]
    if not os.path.exists(ckpt):
        missing.insert(0, ckpt)
    if missing:
        sys.exit("missing files (see docs/install.md):\n  " + "\n  ".join(missing))


def run_video(engine, video, args, tag):
    stem = os.path.splitext(os.path.basename(video))[0]
    out_dir = os.path.join(OUTPUT_ROOT, stem)
    if args.overwrite:
        for name in (f"{stem}_30fps.mp4", f"{stem}_bbox.npz", f"{stem}_vggt_camera.npz",
                     f"{stem}_sam3d_feat.pt"):
            path = os.path.join(out_dir, name)
            if os.path.exists(path):
                os.remove(path)

    engine.process_video(video, OUTPUT_ROOT, seed=args.seed, cfg_scale=args.cfg_scale,
                         vggt_frame_interval=args.vggt_interval)

    # process_video writes <stem>_seed<k>_origin.* (raw) and, with post-processing,
    # <stem>_seed<k>.*; keep the final one as <tag>.* and drop the other.
    gen = os.path.join(out_dir, f"{stem}_seed{args.seed}")
    final = gen if engine.postprocess is not None else gen + "_origin"
    for ext in (".npz", ".fbx"):
        if os.path.exists(final + ext):
            shutil.move(final + ext, os.path.join(out_dir, tag + ext))
        for leftover in (gen + ext, gen + "_origin" + ext):
            if os.path.exists(leftover):
                os.remove(leftover)
    result = os.path.join(out_dir, tag + ".npz")
    if not os.path.exists(result):
        raise FileNotFoundError(f"generation output missing: {result}")
    return result


def parse_args():
    p = argparse.ArgumentParser(description="FlowHMR: raw video -> SMPL-H motion")
    p.add_argument("videos", nargs="+", help="video files and/or directories of videos")
    p.add_argument("--ckpt", default=DEFAULT_CKPT,
                   help="FlowHMR checkpoint (config.yml next to it)")
    p.add_argument("--model_cfg", default=None,
                   help="model config yaml (default: config.yml next to the ckpt)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=None,
                   help="flow-matching ODE steps (default: from config, 20)")
    p.add_argument("--postprocess", action="store_true",
                   help="apply temporal smoothing + foot-contact / ground-contact IK cleanup "
                        "(off: raw model output)")
    p.add_argument("--vggt_interval", type=int, default=1,
                   help="run VGGT-Omega on every N-th frame (camera interpolated in between); "
                        "use 10-30 for long videos or limited GPU memory")
    p.add_argument("--overwrite", action="store_true",
                   help="recompute cached detection / camera / SAM features")
    return p.parse_args()


def main():
    args = parse_args()
    args.ckpt = os.path.abspath(args.ckpt)
    videos = collect_videos(args.videos)
    check_files(args.ckpt)

    model_cfg = args.model_cfg
    if model_cfg is None:
        sibling = os.path.join(os.path.dirname(args.ckpt), "config.yml")
        model_cfg = sibling if os.path.exists(sibling) else DEFAULT_MODEL_CFG
    tag = os.path.splitext(os.path.basename(args.ckpt))[0]
    if args.seed != 0:
        tag += f"_seed{args.seed}"

    if args.device.startswith("cuda:"):
        torch.cuda.set_device(args.device)  # YOLOX builds its grids on the current device
    os.chdir(REPO_ROOT)  # body-model / asset paths are repo-relative

    from flowhmr.pipeline.v2m_inference import V2MInferenceEngine

    engine = V2MInferenceEngine.build(ckpt=args.ckpt, model_cfg_path=model_cfg,
                                      device=args.device, validation_steps=args.steps,
                                      postprocess=args.postprocess)

    results = []
    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {video}")
        t0 = time.time()
        try:
            out = run_video(engine, video, args, tag)
            results.append(("ok", video, out))
            print(f"  [ok] {time.time() - t0:.1f}s -> {out}")
        except Exception as e:  # keep going with the remaining videos
            import traceback
            traceback.print_exc()
            results.append(("failed", video, str(e)))
            print(f"  [failed] {time.time() - t0:.1f}s: {e}")

    n_ok = sum(r[0] == "ok" for r in results)
    print(f"\n=== Done | ok={n_ok} failed={len(results) - n_ok} ===")
    for status, video, info in results:
        print(f"  {status:6s} {video} -> {info}")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()

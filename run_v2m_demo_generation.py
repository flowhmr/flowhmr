"""FlowHMR inference from raw videos: video -> SMPL-H motion (npz, + fbx if fbxsdkpy is installed).

    python run_v2m_demo_generation.py assets/example/example.mp4
    python run_v2m_demo_generation.py a.mp4 b.mov my_videos/ --ckpt checkpoints/flowhmr_base/flowhmr_base.ckpt

The default pipeline is: 30 fps transcode -> YOLO26m person detection & tracking ->
Pi4 camera -> SAM-3D-Body tokens -> FlowHMR generation. --static_camera skips
camera estimation for a fixed camera using approximate intrinsics. Results land in
output/<video name>/<ckpt name>.npz; the extracted features are kept next to it and reused
when the same video is run again (e.g. with another checkpoint). --overwrite recomputes them.
"""
import argparse
import glob
import os
import shutil
import sys
import tempfile
import time

from flowhmr.utils.perception_options import (
    CAMERA_WEIGHTS, DEFAULT_CAMERA, DEFAULT_DETECTOR, DETECTOR_WEIGHTS, camera_cache_suffixes,
)

import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

DEFAULT_CKPT = os.path.join(REPO_ROOT, "checkpoints", "flowhmr_latest", "flowhmr_latest.ckpt")
DEFAULT_MODEL_CFG = os.path.join(REPO_ROOT, "configs", "base", "model_basic05B.yml")
OUTPUT_ROOT = os.path.join(REPO_ROOT, "output")
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
REQUIRED_FILES = [
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


def check_files(ckpt, *, static_camera=False, camera=DEFAULT_CAMERA, detector=DEFAULT_DETECTOR):
    required = [*REQUIRED_FILES, DETECTOR_WEIGHTS[detector]]
    camera_weight = CAMERA_WEIGHTS["static" if static_camera else camera]
    if camera_weight:
        required.append(camera_weight)
    missing = [f for f in required if not os.path.exists(os.path.join(REPO_ROOT, f))]
    if not os.path.exists(ckpt):
        missing.insert(0, ckpt)
    if missing:
        sys.exit("missing files (see docs/install.md):\n  " + "\n  ".join(missing))


def run_video(engine, video, args, tag, *, variant_tag=None):
    stem = os.path.splitext(os.path.basename(video))[0]
    out_dir = os.path.join(OUTPUT_ROOT, stem)
    if args.overwrite:
        suffixes = ["30fps.mp4", "bbox.npz"]
        for mode in CAMERA_WEIGHTS:
            suffixes.extend(camera_cache_suffixes(mode))
        for suffix in suffixes:
            path = os.path.join(out_dir, f"{stem}_{suffix}")
            if os.path.exists(path):
                os.remove(path)

    engine.process_video(video, OUTPUT_ROOT, seed=args.seed, cfg_scale=args.cfg_scale,
                         vggt_frame_interval=args.vggt_interval, static_camera=args.static_camera,
                         camera_backend=getattr(args, "camera", DEFAULT_CAMERA), sam_chunk_size=args.sam_chunk_size)

    # process_video writes <stem>_seed<k>_origin.* (raw) and, with post-processing,
    # <stem>_seed<k>.*; keep the final one as <tag>.* and drop the other.
    gen = os.path.join(out_dir, f"{stem}_seed{args.seed}")
    final = gen if engine.postprocess is not None else gen + "_origin"
    if not os.path.isfile(final + ".npz"):
        raise FileNotFoundError(f"generation output missing: {final}.npz")
    for ext in (".npz", ".fbx"):
        if os.path.exists(final + ext):
            if variant_tag:
                # Preserve the named comparison result when a later run updates the canonical file.
                with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
                    snapshot = os.path.join(tmp, "motion" + ext)
                    try:
                        os.link(final + ext, snapshot)
                    except OSError:
                        shutil.copy2(final + ext, snapshot)
                    os.replace(snapshot, os.path.join(out_dir, variant_tag + ext))
            os.replace(final + ext, os.path.join(out_dir, tag + ext))
        for leftover in (gen + ext, gen + "_origin" + ext):
            if os.path.exists(leftover):
                os.remove(leftover)
    return os.path.join(out_dir, tag + ".npz")


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
                   help="apply RGBMocap contact planning and anatomical IK (see docs/install.md) "
                        "(off: raw model output)")
    p.add_argument("--detector", choices=DETECTOR_WEIGHTS, default=DEFAULT_DETECTOR,
                   help="person detector (default: %(default)s)")
    camera_group = p.add_mutually_exclusive_group()
    camera_group.add_argument("--camera", choices=CAMERA_WEIGHTS, default=DEFAULT_CAMERA,
                              help="camera backend (default: %(default)s, estimates pose and intrinsics)")
    p.add_argument("--camera_interval", "--vggt_interval", dest="vggt_interval", type=int, default=1,
                   help="run VGGT-Omega or Pi4 on every N-th frame (camera interpolated in between); "
                        "use 10-30 for long videos or limited GPU memory; ignored with --static_camera")
    camera_group.add_argument("--static_camera", action="store_true",
                   help="skip camera estimation for a fixed camera; use a constant pose and default intrinsics")
    p.add_argument("--sam_chunk_size", type=int, default=16,
                   help="frames per SAM body batch; hand batch is twice this size (default: 16)")
    p.add_argument("--overwrite", action="store_true",
                   help="recompute cached detection / camera / SAM features")
    args = p.parse_args()
    if args.vggt_interval < 1:
        p.error("--camera_interval must be a positive integer")
    if args.sam_chunk_size < 1:
        p.error("--sam_chunk_size must be a positive integer")
    return args


def main():
    args = parse_args()
    args.ckpt = os.path.abspath(args.ckpt)
    videos = collect_videos(args.videos)
    check_files(args.ckpt, static_camera=args.static_camera, camera=args.camera, detector=args.detector)

    model_cfg = args.model_cfg
    if model_cfg is None:
        sibling = os.path.join(os.path.dirname(args.ckpt), "config.yml")
        model_cfg = sibling if os.path.exists(sibling) else DEFAULT_MODEL_CFG
    checkpoint_name = os.path.splitext(os.path.basename(args.ckpt))[0]
    seed_suffix = f"_seed{args.seed}" if args.seed else ""
    tag = checkpoint_name + seed_suffix
    mode = "static" if args.static_camera else args.camera
    variant_tag = f"{checkpoint_name}_{args.detector}_{mode}{seed_suffix}"

    if args.device.startswith("cuda:"):
        torch.cuda.set_device(args.device)  # YOLOX builds its grids on the current device
    os.chdir(REPO_ROOT)  # body-model / asset paths are repo-relative

    from flowhmr.pipeline.v2m_inference import V2MInferenceEngine

    engine = V2MInferenceEngine.build(ckpt=args.ckpt, model_cfg_path=model_cfg,
                                      device=args.device, validation_steps=args.steps,
                                      postprocess=args.postprocess, detector_backend=args.detector)

    results = []
    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {video}")
        t0 = time.time()
        try:
            out = run_video(engine, video, args, tag, variant_tag=variant_tag)
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

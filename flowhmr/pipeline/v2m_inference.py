import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional

import cv2
import numpy as np
import torch
import yaml
from torch import Tensor


# Enums & Dataclasses

class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_USER = "waiting_user"
    COMPLETED = "completed"
    FAILED = "failed"


class StageName(str, Enum):
    PREPROCESS = "preprocess"
    DETECTION = "detection"
    CAMERA = "camera"
    SAM_EXTRACT = "sam_extract"
    FEATURE_EXTRACT = "feature_extract"
    GENERATION = "generation"


@dataclass
class StageProgress:
    name: StageName
    status: StageStatus = StageStatus.PENDING
    progress: int = 0
    description: str = ""
    result: dict | None = None


@dataclass
class VideoTask:
    video_name: str
    stages: list[StageProgress] = field(default_factory=list)
    current_stage_index: int = 0

    def __post_init__(self):
        if not self.stages:
            self.stages = [
                StageProgress(name=StageName.PREPROCESS),
                StageProgress(name=StageName.DETECTION),
                StageProgress(name=StageName.CAMERA),
                StageProgress(name=StageName.FEATURE_EXTRACT),
                StageProgress(name=StageName.GENERATION),
            ]


# Utility functions (extracted from test_batch_v2m.py)

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _set_buffer(module, dotted_key: str, tensor: Tensor):
    parts = dotted_key.split(".")
    obj = module
    for p in parts[:-1]:
        obj = getattr(obj, p)
    delattr(obj, parts[-1])
    obj.register_buffer(parts[-1], tensor)


def build_pipeline(model_cfg: dict, device: str = "cpu", ckpt: str = None,
                   validation_steps: int = None,
                   smpl_model_path: str = None,
                   j_regressor_path: str = None):
    from flowhmr.core.framework.loaders import load_object

    pipeline_args = dict(model_cfg["train_pipeline_args"])
    network_module = model_cfg["network_module"]
    network_module_args = dict(model_cfg["network_module_args"])

    if validation_steps is not None:
        pipeline_args.setdefault("infer_noise_scheduler_cfg", {})
        pipeline_args["infer_noise_scheduler_cfg"]["validation_steps"] = validation_steps

    if smpl_model_path is not None:
        pipeline_args["smpl_model_path"] = smpl_model_path
    if j_regressor_path is not None:
        pipeline_args["j_regressor_path"] = j_regressor_path

    train_pipeline = model_cfg.get(
        "train_pipeline",
        "flowhmr/pipeline/pipeline_v2m_simple.SimpleV2MPipeline",
    )
    print(f"  pipeline: {train_pipeline}")

    pipeline = load_object(
        train_pipeline,
        pipeline_args,
        network_module=network_module,
        network_module_args=network_module_args,
    )

    if ckpt:
        print(f"  loading checkpoint: {ckpt}")
        checkpoint = torch.load(ckpt, map_location="cpu")
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            epoch = checkpoint.get("epoch", "?")
            step = checkpoint.get("global_step", "?")
            print(f"  checkpoint epoch={epoch}, global_step={step}")
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        model_sd = pipeline.state_dict()
        filtered_sd = {}
        skipped = []
        resized = []
        for k, v in state_dict.items():
            if k not in model_sd:
                skipped.append(k)
                continue
            if v.shape != model_sd[k].shape:
                _set_buffer(pipeline, k, torch.zeros_like(v))
                resized.append(f"{k}: {list(model_sd[k].shape)} -> {list(v.shape)}")
            filtered_sd[k] = v

        missing, unexpected = pipeline.load_state_dict(filtered_sd, strict=False)
        print(f"  loaded {len(filtered_sd)} params/buffers")
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if skipped:
            print(f"  Skipped keys ({len(skipped)}): {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
        if resized:
            print(f"  Resized buffers ({len(resized)}): {resized}")

    pipeline.eval()
    pipeline = pipeline.to(device)
    return pipeline


def build_feature_dict(sam_tokens: Tensor, camera_RT: Tensor, device: torch.device) -> dict:
    T = sam_tokens.shape[0]
    feature = sam_tokens.unsqueeze(0).to(device, dtype=torch.float32)
    camera_R = camera_RT[:, :3, :3].reshape(T, 9).unsqueeze(0).to(device, dtype=torch.float32)
    return {"feature": feature, "camera_R": camera_R}


def forward_smpl_batch(body_model, rot6d, shapes, trans):
    B, L = trans.shape[:2]
    J = rot6d.shape[2]
    rot6d_flat = rot6d.reshape(B * L, J, 6)
    shapes_expanded = shapes.expand(B, L, -1).reshape(B * L, shapes.shape[-1])
    trans_flat = trans.reshape(B * L, 3)
    out = body_model({"rot6d": rot6d_flat, "shapes": shapes_expanded, "trans": trans_flat})
    k3d = out["keypoints3d"]
    return k3d.reshape(B, L, k3d.shape[1], 3)


def fit_floor_height(k3d: Tensor, axis: str = "y") -> Tensor:
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    zs = k3d[..., axis_idx].amin(dim=-1).reshape(-1)
    zs, _ = torch.sort(zs)

    min_z = zs.min()
    max_z = zs.max()
    zs = zs[zs <= min_z + (max_z - min_z) * 1.0]

    inlier_thresh = 0.05
    best_inliers = -1
    best_z = zs[0]
    n = zs.numel()
    for _ in range(10_000):
        z = zs[torch.randint(0, n, (1,), device=zs.device)]
        inliers = (zs - z).abs() < inlier_thresh
        cnt = int(inliers.sum().item())
        if cnt > best_inliers:
            best_inliers = cnt
            best_z = z

    offset_val = zs[(zs - best_z).abs() < inlier_thresh].median()
    height_offset = torch.zeros(3, device=k3d.device, dtype=k3d.dtype)
    height_offset[axis_idx] = offset_val
    return height_offset


def _rotate_smplh_params_y180(poses_aa: np.ndarray, trans: np.ndarray):
    from scipy.spatial.transform import Rotation as R

    # 180° rotation around Y axis
    R_y180 = R.from_euler("y", 180, degrees=True)

    # Rotate root orientation: R_new = R_y180 @ R_old
    root_aa = poses_aa[:, 0, :]
    root_rot = R.from_rotvec(root_aa)
    new_root_rot = R_y180 * root_rot
    poses_aa[:, 0, :] = new_root_rot.as_rotvec().astype(poses_aa.dtype)

    # Rotate root translation
    trans[:] = (R_y180.apply(trans)).astype(trans.dtype)

    return poses_aa, trans


def save_fbx(output: dict, length: int, output_path: str, save_npz: bool = True, export_mesh: bool = True):
    """Write the SMPL-H npz and, if the Autodesk FBX SDK is installed, the .fbx."""
    from flowhmr.core.math.geometry import (
        rot6d_to_rotation_matrix,
        rotation_matrix_to_angle_axis,
    )

    rot6d = output["rot6d"][0, :length]
    shapes = output["shapes"][0, :1]
    trans = output["trans"][0, :length]

    rotations = rot6d_to_rotation_matrix(rot6d)
    poses = rotation_matrix_to_angle_axis(rotations)

    poses_np = poses.cpu().numpy()
    trans_np = trans.cpu().numpy()

    # Rotate 180° around Y axis before saving
    _rotate_smplh_params_y180(poses_np, trans_np)

    params = {
        "poses": poses_np,
        "betas": shapes.cpu().numpy(),
        "trans": trans_np,
        "Rh": poses_np[:, 0, :],
        "mocap_framerate": 30,
        "num_frames": length,
        "gender": "neutral",
    }

    if "foot_contacts" in output:
        fc = output["foot_contacts"][0, :length]  # (T, 4)
        params["foot_contacts"] = fc.cpu().numpy()

    if "local_joints_positions" in output:
        params["local_joints_positions"] = output["local_joints_positions"][0, :length].cpu().numpy()
    if "smooth_root_pos" in output:
        params["smooth_root_pos"] = output["smooth_root_pos"][0, :length].cpu().numpy()
    if "pelvis_world" in output:
        params["pelvis_world"] = output["pelvis_world"][0, :length].cpu().numpy()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if save_npz:
        npz_path = output_path.replace(".fbx", ".npz")
        np.savez(npz_path, **params)

    try:
        from flowhmr.core.export.smplh2fbx import SMPLH2FBX
    except ImportError:  # fbxsdkpy not installed: npz only
        return
    smplh2fbx = SMPLH2FBX()
    smplh2fbx.convert_npz_to_fbx(params, output_path, export_mesh=export_mesh)


# Bbox visualization helpers

def draw_bbox_on_frame(frame_bgr: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray:
    vis = frame_bgr.copy()
    x1, y1, x2, y2 = bbox_xyxy.astype(int)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return vis


def save_bbox_vis_video(
    video_path: str,
    bbox_xyxy: Tensor,
    output_path: str,
    fps: float = 30.0,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")

    bbox_np = bbox_xyxy.cpu().numpy()
    total = bbox_np.shape[0]

    writer = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i >= total:
            break
        vis = draw_bbox_on_frame(frame, bbox_np[i])
        if writer is None:
            h, w = vis.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        writer.write(vis)
        i += 1

    cap.release()
    if writer is not None:
        writer.release()
    print(f"    Bbox vis video saved: {output_path} ({i} frames)")


# SAM sparse visualization

def save_sam_sparse_vis(
    sam_extractor,
    video_path: str,
    bbox_xyxy: Tensor,
    K_all: Tensor,
    output_dir: str,
    vis_frame_indices: list,
):
    from flowhmr.utils.runtime_sam_features import get_visualize_fn

    visualize_sample_together = get_visualize_fn()
    faces = sam_extractor.faces

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")

    bbox_np = bbox_xyxy.cpu()
    K_cpu = K_all.cpu()

    vis_set = set(vis_frame_indices)
    i = 0
    saved = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if i in vis_set:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            b = bbox_np[min(i, bbox_np.shape[0] - 1)].reshape(1, 4).numpy()
            cam_int = K_cpu[min(i, K_cpu.shape[0] - 1)].reshape(1, 3, 3)
            _token, outputs = sam_extractor.extract_frame(
                frame_rgb,
                bboxes=b,
                cam_int=cam_int,
                inference_type="full",
                is_vis=True,
            )
            if outputs:
                rend = visualize_sample_together(frame_bgr, outputs, faces)
                rend = np.clip(rend, 0, 255).astype(np.uint8)
                out_path = os.path.join(output_dir, f"sam_vis_frame{i:06d}.jpg")
                cv2.imwrite(out_path, rend)
                saved += 1
        i += 1
    cap.release()
    print(f"    SAM sparse vis saved: {saved} frames to {output_dir}")


def pick_vis_frame_indices(total_frames: int, max_vis: int = 5) -> list:
    if total_frames <= 0:
        return []
    indices = [0]
    if total_frames > 1 and max_vis > 1:
        step = max(1, (total_frames - 1) // max_vis)
        for j in range(step, total_frames, step):
            indices.append(j)
            if len(indices) >= max_vis:
                break
    return sorted(set(indices))


class V2MInferenceEngine:

    def __init__(self, pipeline, proc, sam_extractor, body_model,
                 smpl_mesh, postprocess, device):
        self.pipeline = pipeline
        self.proc = proc
        self.sam_extractor = sam_extractor
        self.body_model = body_model
        self.smpl_mesh = smpl_mesh
        self.postprocess = postprocess
        self.device = device

    @classmethod
    def build_lite(cls, ckpt: str, model_cfg_path, device: str = "cuda",
                   validation_steps: int = None,
                   postprocess: bool = False,
                   smpl_model_path: str = None,
                   j_regressor_path: str = None) -> "V2MInferenceEngine":
        device_obj = torch.device(device)

        print("[V2MInferenceEngine] Loading models (lite mode, skip YOLOX & SAM3D)...")
        t0 = time.time()

        from flowhmr.core.bodymodels.smpl_skeleton import SMPLSkeleton, SMPLMesh

        if isinstance(model_cfg_path, dict):
            model_cfg = model_cfg_path
        else:
            model_cfg = load_yaml(model_cfg_path)

        bp_kwargs = {}
        if smpl_model_path is not None:
            bp_kwargs["smpl_model_path"] = smpl_model_path
        if j_regressor_path is not None:
            bp_kwargs["j_regressor_path"] = j_regressor_path

        pipe = build_pipeline(
            model_cfg, device=device, ckpt=ckpt,
            validation_steps=validation_steps,
            **bp_kwargs,
        )
        print(f"  Pipeline loaded ({sum(p.numel() for p in pipe.parameters()):,} params)")

        smpl_kw = {"model_path": smpl_model_path} if smpl_model_path else {}
        body_model = SMPLSkeleton(**smpl_kw).to(device_obj)
        print(f"  SMPL body model loaded")

        smpl_mesh = SMPLMesh(**smpl_kw).to(device_obj)
        print(f"  SMPL mesh model loaded")

        if postprocess:
            from flowhmr.utils.runtime_v2m_postprocess_simple import PostprocessPipeline
            postprocess = PostprocessPipeline.default(body_model=body_model, smpl_mesh=smpl_mesh)
            print("  Postprocess: on")
        else:
            postprocess = None
            print("  Postprocess: off")

        print(f"  All models loaded in lite mode ({time.time() - t0:.1f}s)")

        return cls(
            pipeline=pipe,
            proc=None,
            sam_extractor=None,
            body_model=body_model,
            smpl_mesh=smpl_mesh,
            postprocess=postprocess,
            device=device_obj,
        )

    @classmethod
    def build(cls, ckpt: str, model_cfg_path: str, device: str = "cuda",
              validation_steps: int = None,
              postprocess: bool = False,
              sam_device_ids: list[int] | None = None,
              sam_base_port: int = 29500) -> "V2MInferenceEngine":
        device_obj = torch.device(device)

        print("[V2MInferenceEngine] Loading models...")
        t0 = time.time()

        from flowhmr.utils.runtime_video_processing import (
            VideoProcessor,
            build_human_detector,
        )
        from flowhmr.core.bodymodels.smpl_skeleton import SMPLSkeleton, SMPLMesh

        detector = build_human_detector(device_obj)
        proc = VideoProcessor(detector=detector)
        print(f"  YOLOX detector loaded")

        if sam_device_ids and len(sam_device_ids) > 1:
            from flowhmr.utils.runtime_sam_worker import SamWorkerPool
            sam_extractor = SamWorkerPool(
                device_ids=sam_device_ids, base_port=sam_base_port,
            )
            sam_extractor.start()
            print(f"  SAM3D worker pool loaded ({len(sam_device_ids)} GPUs: {sam_device_ids})")
        else:
            from flowhmr.utils.runtime_sam_features import build_sam3d_extractor
            sam_dev = device_obj
            if sam_device_ids and len(sam_device_ids) == 1:
                sam_dev = torch.device(f"cuda:{sam_device_ids[0]}")
            sam_extractor = build_sam3d_extractor(sam_dev)
            print(f"  SAM3D extractor loaded (single GPU: {sam_dev})")

        if isinstance(model_cfg_path, dict):
            model_cfg = model_cfg_path
        else:
            model_cfg = load_yaml(model_cfg_path)
        pipe = build_pipeline(
            model_cfg, device=device, ckpt=ckpt,
            validation_steps=validation_steps,
        )
        print(f"  Pipeline loaded ({sum(p.numel() for p in pipe.parameters()):,} params)")

        body_model = SMPLSkeleton().to(device_obj)
        print(f"  SMPL body model loaded")

        smpl_mesh = SMPLMesh().to(device_obj)
        print(f"  SMPL mesh model loaded")

        if postprocess:
            from flowhmr.utils.runtime_v2m_postprocess_simple import PostprocessPipeline
            postprocess = PostprocessPipeline.default(body_model=body_model, smpl_mesh=smpl_mesh)
            print("  Postprocess: on")
        else:
            postprocess = None
            print("  Postprocess: off")

        print(f"  All models loaded ({time.time() - t0:.1f}s)")

        return cls(
            pipeline=pipe,
            proc=proc,
            sam_extractor=sam_extractor,
            body_model=body_model,
            smpl_mesh=smpl_mesh,
            postprocess=postprocess,
            device=device_obj,
        )

    # Stage 1: Preprocess
    def run_preprocess(self, video_path: str, output_dir: str,
                       progress_cb: Callable[[int, str], None] = None) -> dict:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        sub_dir = os.path.join(output_dir, video_name)
        os.makedirs(sub_dir, exist_ok=True)

        _cb(progress_cb, 0, "Copying original video")
        orig_copy = os.path.join(sub_dir, f"{video_name}_original.mp4")
        if not os.path.exists(orig_copy):
            shutil.copy2(video_path, orig_copy)
        print(f"  [preprocess] Original copied: {orig_copy}")

        _cb(progress_cb, 30, "Transcoding to 30fps")
        transcoded_path = self.proc.transcode(
            video_path,
            backup_dir=sub_dir,
            force_fps=30,
        )
        print(f"  [preprocess] Transcoded: {transcoded_path}")

        _cb(progress_cb, 100, "Preprocess done")
        return {
            "working_video": transcoded_path,
            "sub_dir": sub_dir,
            "video_name": video_name,
        }

    # Stage 2: Detection
    def run_detection(self, working_video: str, sub_dir: str, *,
                      manual_select: bool = False,
                      progress_cb: Callable[[int, str], None] = None) -> dict:
        video_name = os.path.basename(sub_dir)
        bbox_npz_path = os.path.join(sub_dir, f"{video_name}_bbox.npz")
        camera_npz_path = os.path.join(sub_dir, f"{video_name}_camera.npz")

        if os.path.exists(bbox_npz_path) and os.path.exists(camera_npz_path):
            print(f"  [detection] CACHED, loading from npz")
            bbox_data = np.load(bbox_npz_path)
            bbox_xyxy = torch.from_numpy(bbox_data["bbox"])
            K_all = torch.from_numpy(np.load(camera_npz_path)["K"])
            frame_range = None
            if "frame_range" in bbox_data:
                fr = bbox_data["frame_range"]
                frame_range = (int(fr[0]), int(fr[1]))
            _cb(progress_cb, 100, "Detection cached")
            return {"bbox_xyxy": bbox_xyxy, "K_all": K_all, "frame_range": frame_range}

        _cb(progress_cb, 10, "Running human detection + tracking")
        t0 = time.time()

        from flowhmr.utils.runtime_video_processing import VideoProcessor
        tracking_result = self.proc.detect_and_track(working_video)

        if not tracking_result.tracks:
            print(f"  [detection] No tracks found, falling back to full-image bbox")
            bbox_xyxy = self.proc.compute_bboxes(working_video)
            K_all, _ = self.proc.compute_camera_K(working_video)
            self.proc.save_bbox_npz(bbox_npz_path, bbox_xyxy)
            self.proc.save_camera_npz(camera_npz_path, K_all)
            _cb(progress_cb, 100, "Detection done (no tracks, full-image fallback)")
            return {"bbox_xyxy": bbox_xyxy, "K_all": K_all}

        _cb(progress_cb, 50, f"Found {len(tracking_result.tracks)} person(s)")

        if manual_select and len(tracking_result.tracks) > 1:
            detections = []
            for i, track in enumerate(tracking_result.tracks):
                thumbnail_path = self._save_track_thumbnail(
                    working_video, track, sub_dir, person_id=i,
                )
                detections.append({
                    "person_id": i,
                    "track_id": track.track_id,
                    "num_frames": track.length,
                    "avg_area": track.avg_area,
                    "thumbnail_path": thumbnail_path,
                })
            _cb(progress_cb, 60, "Waiting for user selection")
            return {
                "detections": detections,
                "needs_user_select": True,
                "_tracking_result": tracking_result,
                "_sub_dir": sub_dir,
                "_working_video": working_video,
                "_bbox_npz_path": bbox_npz_path,
                "_camera_npz_path": camera_npz_path,
            }

        best_idx = VideoProcessor.select_best_track(tracking_result)
        bbox_xyxy, K_all, frame_range = self._finalize_detection(
            tracking_result, best_idx, working_video, sub_dir,
            bbox_npz_path, camera_npz_path,
        )
        print(f"  [detection] Auto-selected track {tracking_result.tracks[best_idx].track_id} "
              f"({time.time() - t0:.1f}s)")
        _cb(progress_cb, 100, "Detection done")
        return {"bbox_xyxy": bbox_xyxy, "K_all": K_all, "frame_range": frame_range}

    def confirm_detection(self, person_id: int, detection_result: dict,
                          progress_cb: Callable[[int, str], None] = None) -> dict:
        tracking_result = detection_result["_tracking_result"]
        sub_dir = detection_result["_sub_dir"]
        working_video = detection_result["_working_video"]
        bbox_npz_path = detection_result["_bbox_npz_path"]
        camera_npz_path = detection_result["_camera_npz_path"]

        _cb(progress_cb, 70, f"Computing bbox for person {person_id}")
        bbox_xyxy, K_all, frame_range = self._finalize_detection(
            tracking_result, person_id, working_video, sub_dir,
            bbox_npz_path, camera_npz_path,
        )
        print(f"  [detection] User selected person {person_id}")
        _cb(progress_cb, 100, "Detection confirmed")
        return {"bbox_xyxy": bbox_xyxy, "K_all": K_all, "frame_range": frame_range}

    def _finalize_detection(self, tracking_result, track_idx: int,
                            working_video: str, sub_dir: str,
                            bbox_npz_path: str, camera_npz_path: str):
        from flowhmr.utils.runtime_video_processing import VideoProcessor

        track = tracking_result.tracks[track_idx]

        frame_start = int(track.frames.min())
        frame_end = int(track.frames.max()) + 1 # exclusive

        bbox_arr = VideoProcessor.interpolate_track(track, tracking_result.total_frames)
        bbox_arr = VideoProcessor.smooth_bboxes(bbox_arr, tracking_result.width, tracking_result.height)

        bbox_arr = bbox_arr[frame_start:frame_end]
        bbox_xyxy = torch.from_numpy(bbox_arr).to(torch.float32)

        K_all_full, _ = self.proc.compute_camera_K(working_video)
        K_all = K_all_full[frame_start:frame_end]

        frame_range = None
        if frame_start > 0 or frame_end < tracking_result.total_frames:
            frame_range = (frame_start, frame_end)
            print(f"  [detection] Track frame range: {frame_start}-{frame_end} "
                  f"({frame_end - frame_start} frames) / total {tracking_result.total_frames}")

        bbox_np = bbox_xyxy.detach().cpu().numpy().astype(np.float32)
        save_kw = {"bbox": bbox_np, "start_end": np.array([0, bbox_np.shape[0]], dtype=np.int64)}
        if frame_range is not None:
            save_kw["frame_range"] = np.array(frame_range, dtype=np.int64)
        np.savez(bbox_npz_path, **save_kw)

        self.proc.save_camera_npz(camera_npz_path, K_all)

        return bbox_xyxy, K_all, frame_range

    def _save_track_thumbnail(self, video_path: str, track, sub_dir: str,
                              person_id: int) -> str:
        mid_idx = track.length // 2
        frame_idx = int(track.frames[mid_idx])
        bbox = track.bboxes[mid_idx].astype(int)

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()

        if not ok:
            return ""

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]

        thumb_path = os.path.join(sub_dir, f"person_{person_id}_thumb.jpg")
        cv2.imwrite(thumb_path, crop)
        return thumb_path

    def _trim_video(self, video_path: str, frame_start: int, frame_end: int,
                    sub_dir: str) -> str:
        meta = self.proc.get_video_meta(video_path)
        fps = meta.fps
        start_time = frame_start / fps
        duration = (frame_end - frame_start) / fps

        video_name = os.path.basename(sub_dir)
        trimmed_path = os.path.join(sub_dir, f"{video_name}_trimmed.mp4")

        if os.path.exists(trimmed_path):
            print(f"  [trim] CACHED: {trimmed_path}")
            return trimmed_path

        def build_cmd(encoder_name: str) -> List[str]:
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-ss", str(start_time),
                "-t", str(duration),
                "-c:v", encoder_name, "-vsync", "1", "-pix_fmt", "yuv420p", "-an",
            ]
            if "libx264" in encoder_name or "libx265" in encoder_name:
                cmd += ["-preset", "fast", "-crf", "22"]
            cmd.append(trimmed_path)
            return cmd

        candidate_encoders: List[str] = ["libx264", "h264_nvenc", "mpeg4", "libx265"]
        last_error = ""
        for enc in candidate_encoders:
            cmd = build_cmd(enc)
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                break
            last_error = result.stderr.strip() if result.stderr else str(result.returncode)
        else:
            raise RuntimeError(
                f"FFmpeg trim FAILED (tried encoders: {candidate_encoders}): {last_error}"
            )

        trimmed_meta = self.proc.get_video_meta(trimmed_path)
        expected = frame_end - frame_start
        actual = trimmed_meta.total_frames
        print(f"  [trim] frames {frame_start}-{frame_end} ({expected} expected), "
              f"actual trimmed: {actual} frames")

        return trimmed_path

    # Stage 3: Feature Extract
    def run_feature_extract(self, working_video: str, bbox_xyxy: Tensor, K_all: Tensor,
                        sub_dir: str, max_frames: int = None,
                        sam_vis_frames: int = 5,
                        progress_cb: Callable[[int, str], None] = None) -> dict:
        video_name = os.path.basename(sub_dir)
        sam_npz_path = os.path.join(sub_dir, f"{video_name}_sam_tokens.npz")

        if os.path.exists(sam_npz_path):
            print(f"  [feature_extract] CACHED, loading from npz")
            sam_tokens = torch.from_numpy(np.load(sam_npz_path)["tokens"])
            _cb(progress_cb, 100, "Feature tokens cached")
            return {"sam_tokens": sam_tokens}

        _cb(progress_cb, 0, "Starting feature extraction")
        t0 = time.time()

        def _sam_progress(stage, current, total, message):
            if total and total > 0:
                pct = min(95, int(current / total * 95))
                _cb(progress_cb, pct, f"SAM: {current}/{total} frames")

        sam_tokens = self.sam_extractor.extract_video_tokens(
            working_video,
            bbox_xyxy=bbox_xyxy,
            K_all=K_all,
            max_frames=max_frames,
            progress_cb=_sam_progress,
        )
        print(f"  [feature_extract] tokens shape: {tuple(sam_tokens.shape)}")

        np.savez(sam_npz_path, tokens=sam_tokens.cpu().numpy())

        from flowhmr.utils.runtime_sam_features import Sam3DTokenExtractor
        if isinstance(self.sam_extractor, Sam3DTokenExtractor):
            T_vis = sam_tokens.shape[0]
            vis_indices = pick_vis_frame_indices(T_vis, max_vis=sam_vis_frames)
            print(f"  [feature_extract] vis frames: {vis_indices}")
            sam_vis_dir = os.path.join(sub_dir, "sam_vis")
            os.makedirs(sam_vis_dir, exist_ok=True)
            save_sam_sparse_vis(
                self.sam_extractor, working_video, bbox_xyxy, K_all,
                output_dir=sam_vis_dir,
                vis_frame_indices=vis_indices,
            )
        else:
            print(f"  [feature_extract] vis skipped (pool mode)")

        print(f"  [feature_extract] Done ({time.time() - t0:.1f}s)")
        _cb(progress_cb, 100, "Feature extraction done")
        return {"sam_tokens": sam_tokens}

    # Stage 4: Generation
    def run_generation(self, sam_tokens: Tensor, sub_dir: str, video_name: str, *,
                       seed: int = 0, cfg_scale: float = 1.0,
                       export_mesh: bool = True,
                       progress_cb: Callable[[int, str], None] = None) -> dict:
        T = sam_tokens.shape[0]

        _cb(progress_cb, 0, "ODE sampling")
        t0 = time.time()

        camera_RT = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(T, 1, 1)
        feature_dict = build_feature_dict(sam_tokens, camera_RT, self.device)
        print(f"  [generation] Feature dict: { {k: tuple(v.shape) for k, v in feature_dict.items()} }")

        with torch.no_grad():
            output = self.pipeline.generate(
                feature=feature_dict,
                seeds=[seed],
                length=T,
                cfg_scale=cfg_scale,
            )

        print(f"  [generation] Output: { {k: tuple(v.shape) for k, v in output.items() if isinstance(v, Tensor)} }")
        _cb(progress_cb, 50, "ODE sampling done, exporting")

        origin_output = {k: v.clone() if isinstance(v, Tensor) else v for k, v in output.items()}
        with torch.no_grad():
            first_rot6d = origin_output["rot6d"][:, :1]
            first_shapes = origin_output["shapes"]
            first_trans = origin_output["trans"][:, :1]
            B_o = first_rot6d.shape[0]
            first_mesh_out = self.smpl_mesh(
                {"rot6d": first_rot6d.reshape(B_o, 52, 6),
                 "shapes": first_shapes.reshape(B_o, -1),
                 "trans": first_trans.reshape(B_o, 3)},
            )
            first_verts = first_mesh_out["vertices"]
            floor_y = first_verts[:, :, 1].min(dim=-1)[0]
            origin_output["trans"][:, :, 1] -= floor_y[:, None]
            print(f"  [generation] Origin floor offset: {floor_y[0].item()*1000:.1f}mm")

        origin_fbx_path = os.path.join(sub_dir, f"{video_name}_seed{seed}_origin.fbx")
        save_fbx(origin_output, length=T, output_path=origin_fbx_path, save_npz=True, export_mesh=export_mesh)
        _cb(progress_cb, 70, "Origin FBX saved")

        fbx_path = origin_fbx_path
        if self.postprocess is not None:
            output = self.postprocess.run(output)
            fbx_path = os.path.join(sub_dir, f"{video_name}_seed{seed}.fbx")
            save_fbx(output, length=T, output_path=fbx_path, save_npz=True, export_mesh=export_mesh)
        _cb(progress_cb, 95, "Post-processed FBX saved")

        print(f"  [generation] Done ({time.time() - t0:.1f}s)")
        _cb(progress_cb, 100, "Generation complete")
        return {
            "output": output,
            "fbx_path": fbx_path,
            "origin_fbx_path": origin_fbx_path,
        }

    def process_video(self, video_path: str, output_dir: str, *,
                      seed: int = 0, cfg_scale: float = 1.0,
                      max_frames: int = None,
                      vggt_frame_interval: int = 1,
                      person_selector: Callable = None,
                      progress_callback: Callable[[StageName, int, str], None] = None) -> str:
        """Video -> SMPL-H motion.

        person_selector: optional ``f(result, candidates, default) -> candidate index``,
        called after tracking when more than one person track (>= 15 frames) is found, e.g.
        to let a user pick. ``candidates`` lists those tracks (id, frame range, boxes);
        returning None / an out-of-range index keeps ``default``. Without it the longest /
        largest track is used.
        """
        from flowhmr.utils.runtime_video_processing import VideoProcessor

        def _pcb(stage_name, progress, desc):
            if progress_callback is not None:
                progress_callback(stage_name, progress, desc)

        vggt_frame_interval = max(1, int(vggt_frame_interval))
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        sub_dir = os.path.join(output_dir, video_name)
        os.makedirs(sub_dir, exist_ok=True)

        bbox_npz_path = os.path.join(sub_dir, f"{video_name}_bbox.npz")
        feat_pt_path = os.path.join(sub_dir, f"{video_name}_sam3d_feat.pt")
        vggt_camera_npz_path = os.path.join(sub_dir, f"{video_name}_vggt_camera.npz")

        print(f"\n  === Step 1: Bbox ===")
        _pcb(StageName.DETECTION, 0, "Starting bbox extraction")

        if os.path.exists(bbox_npz_path):
            print(f"  [bbox] skip (exists): {bbox_npz_path}")
            working_video = self.proc.transcode(video_path, backup_dir=sub_dir, force_fps=30)
            _pcb(StageName.DETECTION, 100, "Bbox cached")
        else:
            working_video = self.proc.transcode(video_path, backup_dir=sub_dir, force_fps=30)
            _pcb(StageName.DETECTION, 20, "Transcoded, detecting")

            result = self.proc.detect_and_track(working_video)
            _pcb(StageName.DETECTION, 60, f"Found {len(result.tracks)} person(s)")

            if not result.tracks:
                track_idx = -1
                print(f"  [bbox] WARN no_tracks -> full-image bbox x {result.total_frames} frames")
            else:
                track_idx = VideoProcessor.select_best_track(result)
                # Fragments shorter than this (ByteTrack id switches, passers-by) are not offered.
                min_len = 15
                offered = [i for i, t in enumerate(result.tracks) if t.length >= min_len or i == track_idx]
                if person_selector is not None and len(offered) > 1:
                    candidates = []
                    for i in offered:
                        t = result.tracks[i]
                        order = np.argsort(t.frames)
                        candidates.append({
                            "index": len(candidates), "track_index": i, "track_id": int(t.track_id),
                            "start": int(t.frames.min()), "end": int(t.frames.max()),
                            "num_frames": int(t.length), "avg_area": float(t.avg_area),
                            "frames": t.frames[order].astype(int).tolist(),
                            "boxes": np.round(t.bboxes[order].astype(np.float64), 1).tolist(),
                        })
                    _pcb(StageName.DETECTION, 70, "Waiting for person selection")
                    default_pos = offered.index(track_idx)
                    chosen = person_selector(result, candidates, default_pos)
                    if chosen is not None and 0 <= int(chosen) < len(offered):
                        track_idx = offered[int(chosen)]
                track = result.tracks[track_idx]
                print(f"  [bbox] track {track.track_id}: length={track.length} "
                      f"frame_range=[{int(track.frames.min())},{int(track.frames.max())}] "
                      f"/ total={result.total_frames}")
            bbox_np, start_end = VideoProcessor.track_bboxes(result, track_idx)

            np.savez(bbox_npz_path, bbox=bbox_np, start_end=start_end)
            print(f"  [bbox] saved → {bbox_npz_path}  (shape={bbox_np.shape})")
            _pcb(StageName.DETECTION, 100, "Bbox done")

        # ==== Step 2: VGGT Camera ====
        print(f"\n  === Step 2: VGGT Camera ===")
        _pcb(StageName.CAMERA, 0, "Starting VGGT camera extraction")

        data = np.load(bbox_npz_path)
        start_end_arr = data["start_end"].astype(np.int64)
        s, e = int(start_end_arr[0]), int(start_end_arr[1])
        t_track = e - s + 1

        from flowhmr.utils.runtime_vggt_camera import extract_vggt_camera

        _ = extract_vggt_camera(
            working_video,
            output_npz=vggt_camera_npz_path,
            frame_range=(s, e),
            expected_len=t_track,
            device=self.device,
            frame_interval=vggt_frame_interval,
        )
        _pcb(StageName.CAMERA, 100, "VGGT camera ready")

        # ==== Step 3: SAM feature extraction ====
        print(f"\n  === Step 3: SAM Feature ===")
        _pcb(StageName.FEATURE_EXTRACT, 0, "Starting SAM extraction")

        if os.path.exists(feat_pt_path):
            print(f"  [sam] skip (exists): {feat_pt_path}")
            _pcb(StageName.FEATURE_EXTRACT, 100, "SAM cached")
        else:
            data = np.load(bbox_npz_path)
            bbox_track = data["bbox"].astype(np.float32)
            start_end_arr = data["start_end"].astype(np.int64)
            s, e = int(start_end_arr[0]), int(start_end_arr[1])
            t_track = e - s + 1

            meta = VideoProcessor.get_video_meta(working_video)
            total_frames = meta.total_frames
            if total_frames <= 0:
                cap = cv2.VideoCapture(working_video)
                total_frames = 0
                while cap.read()[0]:
                    total_frames += 1
                cap.release()

            bbox_full = np.empty((total_frames, 4), dtype=np.float32)
            if s > 0:
                bbox_full[:s] = bbox_track[0]
            bbox_full[s:e + 1] = bbox_track
            if e + 1 < total_frames:
                bbox_full[e + 1:] = bbox_track[-1]

            max_frames_needed = e + 1
            if max_frames is not None and max_frames > 0:
                max_frames_needed = min(max_frames_needed, max_frames)

            print(
                f"  [sam] bbox.shape={bbox_track.shape}, start_end=[{s},{e}], "
                f"total_frames={total_frames}, max_frames={max_frames_needed}"
            )

            bbox_full_t = torch.from_numpy(bbox_full).float()
            K_all, _ = self.proc.compute_camera_K(working_video)

            camera_data = np.load(vggt_camera_npz_path)
            K_track = torch.from_numpy(camera_data["K"].astype(np.float32))
            assert K_track.shape[0] == t_track, (
                f"VGGT K length {K_track.shape[0]} != track length {t_track}"
            )
            K_all = K_all.to(torch.float32)
            K_all[s:e + 1] = K_track
            print(f"  [sam] using Stage-2 VGGT original-frame K for track frames: {tuple(K_track.shape)}")

            def _sam_progress(stage, current, total, message):
                if total and total > 0:
                    pct = min(95, int(current / total * 95))
                    _pcb(StageName.FEATURE_EXTRACT, pct, f"SAM: {current}/{total} frames")

            tokens_full = self.sam_extractor.extract_video_tokens(
                working_video,
                bbox_xyxy=bbox_full_t,
                K_all=K_all,
                token_dim=3072,
                max_frames=max_frames_needed,
                progress_cb=_sam_progress,
            )

            tokens = tokens_full[s:e + 1].contiguous().cpu().to(torch.float32)
            torch.save(tokens, feat_pt_path)
            print(f"  [sam] saved → {feat_pt_path}  (shape={tuple(tokens.shape)})")
            _pcb(StageName.FEATURE_EXTRACT, 100, "SAM done")

        # ==== Step 4: Generation ====
        print(f"\n  === Step 4: Generation ===")
        _pcb(StageName.GENERATION, 0, "Starting generation")

        origin_fbx_path = os.path.join(sub_dir, f"{video_name}_seed{seed}_origin.fbx")
        final_fbx_path = os.path.join(sub_dir, f"{video_name}_seed{seed}.fbx")

        data = np.load(bbox_npz_path)
        start_end_arr = data["start_end"].astype(np.int64)
        s, e = int(start_end_arr[0]), int(start_end_arr[1])
        expected_len = e - s + 1

        sam_tokens = torch.load(feat_pt_path, map_location="cpu", weights_only=True)
        sam_tokens = sam_tokens.to(torch.float32).contiguous()
        T = sam_tokens.shape[0]
        assert T == expected_len, f"sam_tokens.shape[0]={T} != expected {expected_len}"
        print(f"  [gen] T={T}, range=[{s},{e}], seed={seed}, cfg={cfg_scale}")

        from flowhmr.utils.runtime_vggt_camera import load_or_extract_vggt_camera_rt

        camera_RT = load_or_extract_vggt_camera_rt(
            working_video,
            vggt_camera_npz_path,
            frame_range=(s, e),
            expected_len=T,
            device=self.device,
            frame_interval=vggt_frame_interval,
        )
        feature_dict = build_feature_dict(sam_tokens, camera_RT, self.device)

        with torch.no_grad():
            output = self.pipeline.generate(
                feature=feature_dict,
                seeds=[seed],
                length=T,
                cfg_scale=cfg_scale,
            )
        _pcb(StageName.GENERATION, 50, "ODE sampling done, exporting")

        origin_output = {k: v.clone() if isinstance(v, Tensor) else v
                        for k, v in output.items()}
        with torch.no_grad():
            first_rot6d = origin_output["rot6d"][:, :1]
            first_shapes = origin_output["shapes"]
            first_trans = origin_output["trans"][:, :1]
            B_o = first_rot6d.shape[0]
            first_mesh_out = self.smpl_mesh(
                {"rot6d": first_rot6d.reshape(B_o, 52, 6),
                 "shapes": first_shapes.reshape(B_o, -1),
                 "trans": first_trans.reshape(B_o, 3)},
            )
            first_verts = first_mesh_out["vertices"]
            floor_y = first_verts[:, :, 1].min(dim=-1)[0]
            origin_output["trans"][:, :, 1] -= floor_y[:, None]

        save_fbx(origin_output, length=T, output_path=origin_fbx_path,
                 save_npz=True, export_mesh=True)
        _pcb(StageName.GENERATION, 70, "Origin FBX saved")

        if self.postprocess is not None:
            output = self.postprocess.run(output)
            save_fbx(output, length=T, output_path=final_fbx_path,
                     save_npz=True, export_mesh=True)
        _pcb(StageName.GENERATION, 100, "Generation complete")


        return sub_dir

    def process_videos(self, video_paths: list, output_dir: str, **kwargs) -> list:
        results = []
        for vi, video_path in enumerate(video_paths):
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            print(f"\n{'=' * 70}")
            print(f"  [{vi + 1}/{len(video_paths)}] {video_name}")
            print(f"{'=' * 70}")

            t_video = time.time()
            try:
                sub_dir = self.process_video(video_path, output_dir, **kwargs)
                elapsed = time.time() - t_video
                results.append((video_name, "ok", elapsed, None))
                print(f"  [{video_name}] ok ({elapsed:.1f}s)")
            except Exception as e:
                elapsed = time.time() - t_video
                results.append((video_name, "failed", elapsed, str(e)))
                print(f"  [{video_name}] FAILED: {e} ({elapsed:.1f}s)")
                import traceback
                traceback.print_exc()

        # Summary
        print(f"\n{'=' * 70}")
        print(f"  Batch Summary")
        print(f"{'=' * 70}")
        print(f"  {'Video':<30s} {'Status':<10s} {'Time':>8s}  Error")
        print(f"  {'-' * 70}")
        for name, status, elapsed, error_msg in results:
            err = error_msg or ""
            print(f"  {name:<30s} {status:<10s} {elapsed:>7.1f}s  {err}")
        print(f"{'=' * 70}")

        return results


# Helper

def _cb(progress_cb: Callable | None, progress: int, description: str):
    if progress_cb is not None:
        progress_cb(progress, description)

import os
import os.path as osp
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
import torch
from torch import Tensor


# Repo root (for locating checkpoints)
_repo_root = osp.normpath(osp.join(osp.dirname(__file__), "..", ".."))
_YOLOX_CKPT_PATH = osp.join(_repo_root, "ckpts", "yolox", "yolox_l.pth")


def find_ffmpeg() -> str:
    """Return an ffmpeg executable: $FFMPEG, then PATH, then the imageio-ffmpeg binary."""
    exe = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError(
            "ffmpeg not found: install it system-wide, `pip install imageio-ffmpeg`, "
            "or set $FFMPEG") from e


# Type definitions
@dataclass
class VideoMeta:
    """Metadata extracted from a video file."""
    width: int
    height: int
    fps: float
    total_frames: int


@dataclass
class TrackInfo:
    """Single-person tracking info across frames."""
    track_id: int
    frames: np.ndarray
    bboxes: np.ndarray
    length: int = 0
    avg_area: float = 0.0


@dataclass
class TrackingResult:
    """Result from detect_and_track()."""
    tracks: List[TrackInfo]
    total_frames: int
    width: int
    height: int


# YOLOXHumanDetector
class YOLOXHumanDetector:
    """YOLOX-based human detector, drop-in replacement for the old detectron2 HumanDetector."""

    def __init__(self, ckpt_path: str, device: torch.device, test_size: tuple = (640, 640)):
        import sys as _sys
        _yolox_dir = osp.join(_repo_root, "third_party", "YOLOX")
        if _yolox_dir not in _sys.path:
            _sys.path.insert(0, _yolox_dir)

        from yolox.exp import get_exp # type: ignore
        from yolox.data.data_augment import ValTransform # type: ignore
        from yolox.utils import postprocess # type: ignore

        self.device = device
        self.test_size = test_size
        self._postprocess = postprocess
        self.preproc = ValTransform(legacy=False)

        exp = get_exp(exp_name="yolox-l")
        model = exp.get_model()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        model.to(device)
        model.eval()
        self.model = model
        self.num_classes = exp.num_classes

    def run_human_detection(
        self,
        img: np.ndarray,
        det_cat_id: int = 0,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
        default_to_full_image: bool = True,
    ) -> Optional[np.ndarray]:
        h, w = img.shape[:2]
        ratio = min(self.test_size[0] / h, self.test_size[1] / w)

        img_pre, _ = self.preproc(img, None, self.test_size)
        img_tensor = torch.from_numpy(img_pre).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            outputs = self._postprocess(outputs, self.num_classes, bbox_thr, nms_thr, class_agnostic=True)

        if outputs[0] is None:
            if default_to_full_image:
                return np.array([[0, 0, w, h, 1.0]], dtype=np.float32)
            return None

        det = outputs[0].cpu().numpy() # (N, 7): x1,y1,x2,y2,obj_conf,cls_conf,cls_id
        person_mask = det[:, 6].astype(int) == det_cat_id
        det = det[person_mask]

        if len(det) == 0:
            if default_to_full_image:
                return np.array([[0, 0, w, h, 1.0]], dtype=np.float32)
            return None

        bboxes = det[:, :4] / ratio
        scores = det[:, 4] * det[:, 5] # obj_conf * cls_conf
        result = np.hstack([bboxes, scores[:, None]]).astype(np.float32)
        return result


def build_human_detector(device: torch.device) -> YOLOXHumanDetector:
    """Build YOLOX-L human detector. Raises FileNotFoundError if weights missing."""
    if not osp.exists(_YOLOX_CKPT_PATH):
        raise FileNotFoundError(f"YOLOX checkpoint not found: {_YOLOX_CKPT_PATH}")
    return YOLOXHumanDetector(ckpt_path=_YOLOX_CKPT_PATH, device=device)


class VideoProcessor:
    """Standalone video processing: transcode, detect+track, camera K, etc."""

    def __init__(self, detector: YOLOXHumanDetector):
        self.detector = detector

    # Static helpers
    @staticmethod
    def get_video_meta(video_path: str) -> VideoMeta:
        """Read video metadata via OpenCV."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {video_path}")
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        return VideoMeta(width=width, height=height, fps=fps, total_frames=total_frames)

    @staticmethod
    def default_K_from_hw(height: int, width: int) -> np.ndarray:
        """Default camera intrinsics from image dimensions (diagonal focal length)."""
        f = float((height**2 + width**2) ** 0.5)
        K = np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        return K

    # Camera
    def compute_camera_K(self, video_path: str) -> Tuple[Tensor, None]:
        """Compute per-frame camera intrinsics K (T, 3, 3). RT is always None (placeholder)."""
        meta = self.get_video_meta(video_path)
        total_frames = meta.total_frames
        if total_frames <= 0:
            cap = cv2.VideoCapture(video_path)
            cnt = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                cnt += 1
            cap.release()
            total_frames = cnt
        K0 = self.default_K_from_hw(meta.height, meta.width).astype(np.float32)
        K = np.repeat(K0[None, ...], total_frames, axis=0)
        return torch.from_numpy(K).to(torch.float32), None

    # Transcode
    @staticmethod
    def transcode(
        video_path: str,
        backup_dir: str,
        start_time: float = 0.0,
        duration: Optional[float] = None,
        force_fps: Optional[int] = 30,
    ) -> str:
        """Transcode video to a cached mp4, trying multiple encoders."""
        os.makedirs(backup_dir, exist_ok=True)
        base = osp.splitext(osp.basename(video_path))[0]
        fps_tag = f"_{force_fps}fps" if force_fps else ""
        cache_base = f"{base}{fps_tag}"
        backup_path = osp.join(backup_dir, f"{cache_base}.mp4")

        if os.path.exists(backup_path):
            print(f">>> using cached transcoded video: {backup_path}")
            return backup_path

        ffmpeg_exe = find_ffmpeg()
        # Write to a temp file and rename when done, so a reader (e.g. the web demo serving
        # this file) never sees a half-written mp4.
        tmp_path = osp.join(backup_dir, f"{cache_base}.part.mp4")

        def build_cmd(encoder_name: Optional[str]) -> List[str]:
            cmd = [ffmpeg_exe, "-y", "-ss", str(start_time), "-i", video_path]
            if duration is not None:
                cmd += ["-t", str(duration)]
            if force_fps and encoder_name:
                cmd += ["-vf", f"fps={int(force_fps)}", "-c:v", encoder_name, "-vsync", "1", "-pix_fmt", "yuv420p"]
                if "libx264" in encoder_name or "libx265" in encoder_name:
                    cmd += ["-preset", "fast", "-crf", "22"]
                elif "nvenc" in encoder_name:
                    cmd += ["-preset", "p4", "-cq", "24"]
                else:
                    cmd += ["-q:v", "5"]
            else:
                cmd += ["-c:v", "copy"]
            cmd += ["-an", "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", tmp_path]
            return cmd

        if force_fps:
            candidate_encoders: List[Optional[str]] = ["libx264", "h264_nvenc", "mpeg4", "libx265"]
        else:
            candidate_encoders = [None]

        last_error = ""
        for enc in candidate_encoders:
            cmd = build_cmd(enc)
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                os.replace(tmp_path, backup_path)
                print(f">>> video transcoding done (encoder: {enc if enc else 'copy'}): {backup_path}")
                return backup_path
            except subprocess.CalledProcessError as e:
                err_msg = e.stderr.strip() if e.stderr else str(e)
                last_error = err_msg
                continue

        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"FFmpeg transcoding FAILED (tried encoders: {candidate_encoders}): {last_error}")

    # Detect + Track
    def detect_and_track(
        self,
        video_path: str,
        *,
        bbox_thr: float = 0.4,
        nms_thr: float = 0.3,
        bytetrack_thresh: float = 0.25,
        bytetrack_match: float = 0.8,
    ) -> TrackingResult:
        """Run per-frame detection + ByteTrack, returning all tracks."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = None

        tracker = sv.ByteTrack(
            track_activation_threshold=bytetrack_thresh,
            minimum_matching_threshold=bytetrack_match,
        )
        raw_tracks: Dict[int, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
        frame_idx = 0

        pbar_total = total_frames if total_frames else "?"
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                results = self.detector.run_human_detection(
                    frame_bgr, det_cat_id=0, bbox_thr=bbox_thr, nms_thr=nms_thr, default_to_full_image=False,
                )

                if results is not None and len(results) > 0:
                    if not isinstance(results, np.ndarray):
                        results = np.array(results)
                    xyxy = results[:, :4].astype(np.float32)
                    confidence = results[:, 4].astype(np.float32)
                    class_id = np.zeros((len(results),), dtype=int)
                else:
                    xyxy = np.empty((0, 4), dtype=np.float32)
                    confidence = np.empty((0,), dtype=np.float32)
                    class_id = np.empty((0,), dtype=int)

                detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
                detections = tracker.update_with_detections(detections)

                tids = detections.tracker_id
                if tids is None:
                    tids = np.empty((0,), dtype=np.int64)
                for tid, box in zip(tids, detections.xyxy):
                    tid_i = int(tid)
                    raw_tracks[tid_i]["bboxes"].append(box.astype(np.float32))
                    raw_tracks[tid_i]["frames"].append(frame_idx)

                frame_idx += 1
                if frame_idx % 100 == 0:
                    print(f">>> bbox detection: {frame_idx}/{pbar_total} frames")
        finally:
            cap.release()

        if frame_idx <= 0:
            raise ValueError(f"no frames could be read from video: {video_path}")

        # Build TrackInfo list
        tracks: List[TrackInfo] = []
        for tid, v in raw_tracks.items():
            bboxes = np.stack(v["bboxes"], axis=0).astype(np.float32)
            frames = np.asarray(v["frames"], dtype=np.int64)
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            tracks.append(TrackInfo(
                track_id=int(tid),
                frames=frames,
                bboxes=bboxes,
                length=int(bboxes.shape[0]),
                avg_area=float(np.mean(areas)),
            ))

        print(f">>> bbox detection done: {frame_idx} frames, {len(tracks)} tracks")
        return TrackingResult(tracks=tracks, total_frames=frame_idx, width=width, height=height)

    # Post-processing (all static)
    @staticmethod
    def select_best_track(result: TrackingResult) -> int:
        """Select best track index by (length, avg_area) descending. Returns -1 if no tracks."""
        if not result.tracks:
            return -1
        scored = [(i, t.length, t.avg_area) for i, t in enumerate(result.tracks)]
        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return scored[0][0]

    @staticmethod
    def interpolate_track(track: TrackInfo, total_frames: int) -> np.ndarray:
        """Linearly interpolate a track's bboxes to every frame. Returns (total_frames, 4) float32."""
        order = np.argsort(track.frames)
        frames_sorted = track.frames[order].astype(np.float32)
        bboxes_sorted = track.bboxes[order]

        all_frames = np.arange(total_frames, dtype=np.float32)
        bbox_arr = np.empty((total_frames, 4), dtype=np.float32)

        if frames_sorted.size == 0:
            bbox_arr[:] = 0.0
            return bbox_arr
        if frames_sorted.size == 1:
            bbox_arr[:] = bboxes_sorted[0]
            return bbox_arr

        for j in range(4):
            bbox_arr[:, j] = np.interp(
                all_frames, frames_sorted, bboxes_sorted[:, j].astype(np.float32),
                left=float(bboxes_sorted[0, j]), right=float(bboxes_sorted[-1, j]),
            ).astype(np.float32)
        return bbox_arr

    @staticmethod
    def track_bboxes(result: TrackingResult, track_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Per-frame boxes of one track over its own frame range.

        Returns (bbox (N, 4) float32 xyxy, start_end int64 [first, last] frame, inclusive).
        track_idx < 0 (or no tracks) gives a full-image box over the whole video.
        """
        if track_idx < 0 or not result.tracks:
            full_box = np.array([0.0, 0.0, float(result.width), float(result.height)], dtype=np.float32)
            bbox = np.tile(full_box[None, :], (result.total_frames, 1)).astype(np.float32)
            return bbox, np.array([0, max(result.total_frames - 1, 0)], dtype=np.int64)
        track = result.tracks[track_idx]
        start, end = int(track.frames.min()), int(track.frames.max())
        bbox_full = VideoProcessor.interpolate_track(track, result.total_frames)
        bbox_full = VideoProcessor.smooth_bboxes(bbox_full, result.width, result.height)
        return bbox_full[start:end + 1].astype(np.float32), np.array([start, end], dtype=np.int64)

    @staticmethod
    def smooth_bboxes(bbox_arr: np.ndarray, width: int, height: int) -> np.ndarray:
        """Clamp bboxes to image bounds. In-place modification + return."""
        bbox_arr[:, 0] = np.clip(bbox_arr[:, 0], 0, width - 1)
        bbox_arr[:, 1] = np.clip(bbox_arr[:, 1], 0, height - 1)
        bbox_arr[:, 2] = np.clip(bbox_arr[:, 2], 1, width)
        bbox_arr[:, 3] = np.clip(bbox_arr[:, 3], 1, height)
        return bbox_arr

    # High-level: detect → track → select → interpolate → clamp → Tensor
    def compute_bboxes(
        self,
        video_path: str,
        *,
        bbox_thr: float = 0.4,
        nms_thr: float = 0.3,
        bytetrack_thresh: float = 0.25,
        bytetrack_match: float = 0.8,
        bbox_interp: bool = True,
    ) -> Tensor:
        """End-to-end bbox pipeline: detect+track → select best → interpolate → clamp.

        Returns (T, 4) float32 Tensor of xyxy bboxes, one per frame.
        Falls back to full-image bbox when no tracks are found.
        """
        result = self.detect_and_track(
            video_path,
            bbox_thr=bbox_thr,
            nms_thr=nms_thr,
            bytetrack_thresh=bytetrack_thresh,
            bytetrack_match=bytetrack_match,
        )
        full_box = np.array([0.0, 0.0, float(result.width), float(result.height)], dtype=np.float32)

        if not result.tracks:
            print(">>> no track found, using full-image bbox")
            bbox_arr = np.tile(full_box[None, :], (result.total_frames, 1)).astype(np.float32)
        else:
            best_idx = self.select_best_track(result)
            track = result.tracks[best_idx]
            print(f">>> selected track {track.track_id}: {track.length} frames, avg_area={track.avg_area:.0f}")

            if track.frames.size == 0:
                bbox_arr = np.tile(full_box[None, :], (result.total_frames, 1)).astype(np.float32)
            elif bbox_interp:
                bbox_arr = self.interpolate_track(track, result.total_frames)
            else:
                # Forward-fill: use last observed box
                frame_to_box = {int(f): track.bboxes[i] for i, f in enumerate(track.frames.tolist())}
                bbox_arr = np.empty((result.total_frames, 4), dtype=np.float32)
                last_box = full_box.copy()
                for t in range(result.total_frames):
                    if t in frame_to_box:
                        last_box = frame_to_box[t].astype(np.float32)
                    bbox_arr[t] = last_box

            bbox_arr = self.smooth_bboxes(bbox_arr, result.width, result.height)

        return torch.from_numpy(bbox_arr).to(torch.float32)

    # I/O helpers (static)
    @staticmethod
    def save_bbox_npz(bbox_npz_path: str, bbox_xyxy: Tensor) -> None:
        bbox = bbox_xyxy.detach().cpu().numpy().astype(np.float32)
        np.savez(bbox_npz_path, bbox=bbox, start_end=np.array([0, bbox.shape[0]], dtype=np.int64))

    @staticmethod
    def save_camera_npz(camera_npz_path: str, K_all: Tensor) -> None:
        K = K_all.detach().cpu().numpy().astype(np.float32)
        np.savez(camera_npz_path, K=K, start_end=np.array([0, K.shape[0]], dtype=np.int64))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="runtime_video_processing standalone test")
    parser.add_argument("--video", required=True, help="Path to input video")
    args = parser.parse_args()

    video_path = args.video
    assert osp.exists(video_path), f"video not found: {video_path}"

    # 1. Video meta
    meta = VideoProcessor.get_video_meta(video_path)
    print(f"[video_meta] {meta.width}x{meta.height}, {meta.fps:.1f}fps, {meta.total_frames} frames")

    # 2. Build detector
    print(f"[detector] YOLOX ckpt: {_YOLOX_CKPT_PATH}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    detector = build_human_detector(device)
    print(f"[detector] OK on {device}")

    # 3. Full bbox pipeline
    proc = VideoProcessor(detector=detector)
    bbox = proc.compute_bboxes(video_path)
    print(f"[compute_bboxes] shape={tuple(bbox.shape)}")

    # 4. Camera K
    K_all, _ = proc.compute_camera_K(video_path)
    print(f"[camera_K] shape={tuple(K_all.shape)}, fx={K_all[0, 0, 0]:.1f}")

    print("\n[DONE] All steps completed successfully.")

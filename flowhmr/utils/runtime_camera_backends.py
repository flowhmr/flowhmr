"""Camera-only video backends. All public results use OpenCV world-to-camera poses."""
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile

import cv2
import numpy as np
import torch

from .perception_options import CAMERA_WEIGHTS
from .runtime_vggt_camera import VGGTCameraResult, _interpolate_camera_sequence
from .runtime_video_processing import VideoProcessor


def _sample_video(video, start, indices):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open {video}")
    try:
        if start and not capture.set(cv2.CAP_PROP_POS_FRAMES, start):
            raise ValueError(f"Cannot seek to frame {start} in {video}")
        position = 0
        for index in indices:
            while position < index:
                if not capture.grab():
                    raise ValueError(f"Video ended at frame {start + position}: {video}")
                position += 1
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode frame {start + index} in {video}")
            position += 1
            yield frame
    finally:
        capture.release()


def _simplevo(frames, hw):
    """GVHMR's half-resolution SIFT / calibrated two-view pipeline, keeping rotations."""
    import pycolmap

    height, width = hw
    size = (max(1, width // 2), max(1, height // 2))
    # GVHMR's 24 mm full-frame focal prior, expressed in pixels at each resolution.
    focal_ratio = 24 / np.hypot(36, 24)
    camera = pycolmap.Camera(model="SIMPLE_PINHOLE", width=size[0], height=size[1],
                             params=[np.hypot(*size) * focal_ratio, size[0] / 2, size[1] / 2])
    options = pycolmap.TwoViewGeometryOptions(min_num_inliers=10, min_E_F_inlier_ratio=0.8,
                                               max_H_inlier_ratio=0.9, compute_relative_pose=True)
    sift = cv2.SIFT_create()
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    poses = []
    rotation = np.eye(3)
    prev_desc = prev_points = None
    failed = 0
    for frame in frames:
        gray = cv2.cvtColor(cv2.resize(frame, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        points = np.array([p.pt for p in keypoints], dtype=np.float64).reshape(-1, 2)
        relative = None
        if prev_desc is not None and descriptors is not None and len(descriptors) >= 2:
            pairs = matcher.knnMatch(prev_desc, descriptors, k=2)
            matches = [(m.queryIdx, m.trainIdx) for m, n in pairs if m.distance < 0.7 * n.distance]
            if len(matches) >= options.min_num_inliers:
                solution = pycolmap.estimate_calibrated_two_view_geometry(
                    camera, prev_points, camera, points, np.asarray(matches, dtype=np.uint32), options)
                if len(solution.inlier_matches) >= options.min_num_inliers:
                    relative = solution.cam2_from_cam1
        if relative is not None:
            rotation = relative.rotation.matrix() @ rotation
        elif poses:
            failed += 1
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rotation
        poses.append(pose)
        prev_points, prev_desc = points, descriptors
    if failed:
        print(f"  [simplevo] held previous rotation for {failed}/{len(poses)-1} low-match pairs")
    K = VideoProcessor.default_K_from_hw(height, width)
    K[0, 0] = K[1, 1] = np.hypot(width, height) * focal_ratio
    return torch.from_numpy(np.stack(poses)), K


def _fit_ray_intrinsics(rays, pixels):
    """Fit u = fx * ray_x + cx and v = fy * ray_y + cy in original-frame pixels."""
    rays = np.asarray(rays, dtype=np.float64).reshape(-1, 2)
    pixels = pixels.reshape(-1, 2)
    valid = np.isfinite(rays).all(axis=1)
    rays, pixels = rays[valid], pixels[valid]
    if len(rays) < 2:
        raise ValueError("Pi4 produced too few finite camera rays to estimate intrinsics")
    weights = np.ones_like(rays)
    # Independent Huber fits keep outlying rays from shifting either focal length.
    for iteration in range(6):
        ray_center = np.average(rays, axis=0, weights=weights)
        pixel_center = np.average(pixels, axis=0, weights=weights)
        centered = rays - ray_center
        variance = (weights * centered ** 2).sum(axis=0)
        if np.any(variance <= np.finfo(np.float64).eps):
            raise ValueError("Pi4 produced degenerate camera rays")
        focal = (weights * centered * (pixels - pixel_center)).sum(axis=0) / variance
        principal = pixel_center - focal * ray_center
        if iteration < 5:
            residual = pixels - (focal * rays + principal)
            mad = np.median(abs(residual - np.median(residual, axis=0)), axis=0)
            cutoff = 1.345 * np.maximum(0.1, 1.4826 * mad)
            weights = np.minimum(1., cutoff / np.maximum(abs(residual), 1e-8))
    if not np.isfinite([focal, principal]).all() or np.any(focal <= 0):
        raise ValueError("Pi4 camera rays did not yield positive finite focal lengths")
    return np.array([[focal[0], 0, principal[0]], [0, focal[1], principal[1]], [0, 0, 1]], dtype=np.float32)


@torch.inference_mode()
def _pi4(frames, hw, device):
    from PIL import Image
    from safetensors import safe_open
    try:
        from third_party.pi4_camera.encoder import Encoder
        from third_party.pi4_camera.decoder import GeometryDecoder
        from third_party.pi4_camera.heads.camera import CameraHead
        from third_party.pi4_camera.heads.metric import MetricHead
        from third_party.pi4_camera.layers import ConvPyramid, feature_pyramid, patch_positions
    except ModuleNotFoundError as exc:
        raise ImportError("Prepare the Pi4 camera modules with python tools/setup_pi4_camera.py") from exc

    checkpoint = Path(__file__).resolve().parents[2] / CAMERA_WEIGHTS["pi4"]
    with torch.device("meta"):
        # Pi4's point decoder shares its trunk between ray X/Y and log-depth.
        # Restrict the output convolution to the two pretrained ray channels.
        point_head = torch.nn.ModuleDict({"points": ConvPyramid((512, 256, 128, 64, 32), 2)})
        model = torch.nn.ModuleDict(dict(encoder=Encoder(), geometry_decoder=GeometryDecoder(),
                                         camera_head=CameraHead(), metric_head=MetricHead(), point_head=point_head))
    with safe_open(str(checkpoint), framework="pt", device="cpu") as weights:
        state = {}
        for key in model.state_dict():
            if key.startswith("point_head.points.output."):
                state[key] = weights.get_slice(key)[:2]
            else:
                state[key] = weights.get_tensor(key)
        model.load_state_dict(state, strict=True, assign=True)
    del state
    model.eval().to(device)
    height, width = hw
    ratio = 512 / max(hw)
    size = tuple(max(16, round(n * ratio / 16) * 16) for n in (width, height))
    images = [np.asarray(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).resize(
        size, Image.Resampling.LANCZOS)) for f in frames]
    rgb = torch.from_numpy(np.stack(images)).to(device).permute(0, 3, 1, 2).float().div_(255)
    rgb.sub_(rgb.new_tensor([0.485, 0.456, 0.406])[:, None, None])
    rgb.div_(rgb.new_tensor([0.229, 0.224, 0.225])[:, None, None])
    count = len(rgb)
    positions = patch_positions(count, size[1] // 16, size[0] // 16, rgb.device)
    with torch.autocast(rgb.device.type, dtype=torch.bfloat16, enabled=rgb.is_cuda):
        tokens = model.encoder(rgb)
        geometry, initial = model.geometry_decoder(tokens, positions, 1, count)
    del initial, tokens, rgb
    with torch.autocast(geometry.device.type, enabled=False):
        cameras = model.camera_head(geometry.float(), positions).reshape(count, 4, 4)
        scale = model.metric_head(geometry[:, 5:].float(), positions[:, 5:], 1)
        # PIL's resize maps pixel centers separately along the two axes.
        grid = np.moveaxis(np.indices(size[::-1])[::-1], 0, -1)[4:-4:4, 4:-4:4]
        pixels = (grid + 0.5) * (np.array([width, height]) / size) - 0.5
        intrinsics = []
        for frame_tokens in geometry[:, 5:].split(1):
            pyramid = feature_pyramid(frame_tokens.float(), size[1], size[0], 5)
            rays, _ = model.point_head.points(pyramid)
            sampled = rays[0, :, 4:-4:4, 4:-4:4].permute(1, 2, 0).cpu().numpy()
            intrinsics.append(_fit_ray_intrinsics(sampled, pixels))
    # Pi4 predicts camera-to-world; its metric scale multiplies translations.
    cameras[:, :3, 3] *= scale.item()
    world_to_camera = torch.linalg.inv(cameras) @ cameras[0]
    return world_to_camera.cpu(), np.stack(intrinsics)


def extract_camera(video, mode, *, frame_range, output_npz, device, frame_interval):
    start, end = frame_range
    length = end - start + 1
    interval = 8 if mode == "simplevo" else frame_interval
    if mode not in ("pi4", "simplevo") or start < 0 or length < 1 or interval < 1:
        raise ValueError(f"Invalid camera request: {mode}, {frame_range}, stride={interval}")
    meta = VideoProcessor.get_video_meta(video)
    hw = (meta.height, meta.width)
    path = Path(output_npz)
    k_source = "pi4_rays" if mode == "pi4" else "24mm_full_frame"
    if path.is_file():
        try:
            with np.load(path, allow_pickle=False) as data:
                poses, K = data["camera_RT"], data["K"]
                if (str(data["camera_mode"]) == mode and str(data["k_space"]) == "original_frame"
                        and (mode != "pi4" or str(data.get("k_source", "")) == k_source)
                        and int(data["camera_sampling_interval"]) == interval
                        and np.array_equal(data["frame_range"], frame_range)
                        and np.array_equal(data["image_size_hw"], hw)
                        and poses.shape == (length, 4, 4) and K.shape == (length, 3, 3)
                        and np.isfinite(poses).all() and np.isfinite(K).all()):
                    return VGGTCameraResult(poses, poses, K, hw)
        except (KeyError, OSError, ValueError, EOFError, BadZipFile):
            pass
    indices = list(range(0, length, interval))
    if indices[-1] != length - 1:
        indices.append(length - 1)
    frames = _sample_video(video, start, indices)
    try:
        poses, sampled_K = _simplevo(frames, hw) if mode == "simplevo" else _pi4(frames, hw, device)
    finally:
        frames.close()
    intrinsics = torch.from_numpy(sampled_K).expand(len(indices), -1, -1)
    rt, k = _interpolate_camera_sequence(poses[:, :3], intrinsics, indices, length)
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], length, axis=0)
    poses[:, :3] = rt.numpy()
    K = k.numpy()
    if not np.isfinite(poses).all() or not np.isfinite(K).all():
        raise ValueError(f"{mode} returned non-finite camera parameters")
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        np.savez(temp_path, camera_RT=poses, RT=poses, K=K, frame_range=frame_range,
                 image_size_hw=hw, camera_mode=mode, camera_sampling_interval=interval,
                 k_space="original_frame", k_source=k_source)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)
    return VGGTCameraResult(poses, poses, K, hw)

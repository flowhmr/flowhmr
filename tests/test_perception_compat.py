"""Regression coverage for detector/camera defaults, legacy caches and saved Web jobs."""
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from flowhmr.pipeline.v2m_inference import _bbox_cache_matches
from flowhmr.utils.perception_options import CAMERA_WEIGHTS, DETECTOR_WEIGHTS
from flowhmr.utils import runtime_camera_backends as cameras
from flowhmr.utils.runtime_video_processing import VideoProcessor
import run_v2m_demo_generation as cli


class PerceptionCompatibilityTest(unittest.TestCase):
    def test_pi4_focal_and_off_center_principal_point_from_rays(self):
        y, x = np.mgrid[4:140:4, 4:252:4]
        pixels = (np.stack([x, y], axis=-1) + 0.5) * [2.625, 2.667] - 0.5
        rays = (pixels - [340., 184.]) / [315., 322.]
        rays[4::11, 2::9] += 0.15
        rays[0, 0] = np.nan
        estimated = cameras._fit_ray_intrinsics(rays, pixels)
        np.testing.assert_allclose(estimated, [[315, 0, 340], [0, 322, 184], [0, 0, 1]], atol=.05)
        with self.assertRaisesRegex(ValueError, "degenerate"):
            cameras._fit_ray_intrinsics(np.zeros_like(rays), pixels)

    def test_pi4_replaces_diagonal_cache_and_interpolates_estimated_intrinsics(self):
        meta = SimpleNamespace(height=128, width=256)
        rt = torch.eye(4).repeat(3, 1, 1)
        sampled = np.array([[[fx, 0, cx], [0, fx + 8, 60], [0, 0, 1]]
                            for fx, cx in ((120, 126), (130, 127), (134, 128))], dtype=np.float32)
        with TemporaryDirectory() as tmp, patch.object(VideoProcessor, "get_video_meta", return_value=meta):
            cache = Path(tmp) / "pi4.npz"
            np.savez(cache, camera_RT=np.repeat(np.eye(4)[None], 6, axis=0), K=np.tile(np.eye(3), (6, 1, 1)),
                     frame_range=[2, 7], image_size_hw=[128, 256], camera_mode="pi4", camera_sampling_interval=4,
                     k_space="original_frame", k_source="image_diagonal")
            options = dict(frame_range=(2, 7), output_npz=cache, device="cpu", frame_interval=4)
            with patch.object(cameras, "_pi4", return_value=(rt, sampled)) as predict:
                first = cameras.extract_camera("video.mp4", "pi4", **options)
                second = cameras.extract_camera("video.mp4", "pi4", **options)
            predict.assert_called_once()
            np.testing.assert_allclose(first.K[[0, 4, 5]], sampled)
            np.testing.assert_allclose(first.K[2], (sampled[0] + sampled[1]) / 2)
            np.testing.assert_array_equal(first.K, second.K)
            with np.load(cache) as saved:
                self.assertEqual(str(saved["k_source"]), "pi4_rays")

    def test_cli_defaults_and_legacy_flags(self):
        with patch.object(sys, "argv", ["demo", "video.mp4"]):
            args = cli.parse_args()
        self.assertEqual((args.detector, args.camera), ("yolo26m", "pi4"))
        self.assertEqual(VideoProcessor().backend, args.detector)
        with patch.object(sys, "argv", ["demo", "video.mp4", "--static_camera", "--vggt_interval", "8"]):
            args = cli.parse_args()
        self.assertTrue(args.static_camera)
        self.assertEqual(args.vggt_interval, 8)
        for detector in DETECTOR_WEIGHTS:
            for camera in CAMERA_WEIGHTS:
                with self.subTest(detector=detector, camera=camera):
                    with patch.object(sys, "argv", ["demo", "video.mp4", "--detector", detector, "--camera", camera]):
                        args = cli.parse_args()
                    self.assertEqual((args.detector, args.camera), (detector, camera))

    def test_legacy_and_backend_specific_bbox_caches(self):
        old = {"bbox": np.zeros((3, 4)), "start_end": [2, 4]}
        self.assertTrue(_bbox_cache_matches(old, "yolox", inclusive=True))
        self.assertFalse(_bbox_cache_matches(old, "yolo26m", inclusive=True))
        self.assertFalse(_bbox_cache_matches(old, "yolox", inclusive=False))
        staged = {**old, "start_end": [0, 3], "detector_backend": "yolo26m"}
        self.assertTrue(_bbox_cache_matches(staged, "yolo26m", inclusive=False))
        self.assertFalse(_bbox_cache_matches(staged, "yolo26m", inclusive=True))

    def test_cli_result_path_survives_backend_switch(self):
        args = SimpleNamespace(overwrite=False, seed=0, cfg_scale=1, vggt_interval=8,
                               static_camera=False, sam_chunk_size=16)

        def generate(video, root, **options):
            folder = Path(root) / Path(video).stem
            folder.mkdir(exist_ok=True)
            np.savez(folder / "clip_seed0_origin.npz", camera=options["camera_backend"])

        engine = SimpleNamespace(postprocess=None, process_video=generate)
        with TemporaryDirectory() as tmp, patch.object(cli, "OUTPUT_ROOT", tmp):
            for camera in ("pi4", "vggt"):
                args.camera = camera
                output = cli.run_video(engine, "clip.mp4", args, "motion", variant_tag=f"motion_{camera}")
                self.assertEqual(Path(output), Path(tmp) / "clip/motion.npz")
                with np.load(output) as data:
                    self.assertEqual(str(data["camera"]), camera)
            with np.load(Path(tmp) / "clip/motion_pi4.npz") as first:
                self.assertEqual(str(first["camera"]), "pi4")

    def test_restore_jobs_and_rerun(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("flowhmr_web_server", root / "tools/web/server.py")
        web = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(web)
        self.assertEqual(web.parse_params({})["camera"], "pi4")
        fixtures = [
            (None, ("yolox", "vggt")),  # Old output directory without job.json
            ({}, ("yolox", "vggt")),
            ({"static_camera": True}, ("yolox", "static")),
            ({"detector": "yolo26m", "camera": "simplevo"}, ("yolo26m", "simplevo")),
        ]
        with TemporaryDirectory() as tmp:
            web.ARGS = SimpleNamespace(work_dir=tmp)
            for i, (params, _) in enumerate(fixtures):
                folder = Path(tmp) / f"20261101_100000_{i:06x}"
                folder.mkdir()
                (folder / "input.mp4").touch()
                if params is not None:
                    (folder / "job.json").write_text(json.dumps({"params": params, "status": "done"}))
            web.restore_jobs()
            for i, (_, expected) in enumerate(fixtures):
                params = web.JOBS[f"20261101_100000_{i:06x}"]["params"]
                self.assertEqual((params["detector"], params["camera"]), expected)
                self.assertEqual(web.parse_params({}, params)["camera"], expected[1])
                for cam in CAMERA_WEIGHTS:
                    changed = web.parse_params({"camera": cam}, params)
                    self.assertEqual(changed["camera"], cam)
                    self.assertEqual(changed["static_camera"], cam == "static")


if __name__ == "__main__":
    unittest.main()

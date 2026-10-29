"""CPU integration checks for fixed-camera inference, frame ranges and caches.

Only model inference and media export are replaced. The production camera,
tracking-range, intrinsics, feature-preparation and CLI paths execute normally.
"""

import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_v2m_demo_generation as cli
from flowhmr.pipeline import v2m_inference as inference
from flowhmr.utils import runtime_vggt_camera as camera
from flowhmr.utils.runtime_video_processing import TrackInfo, TrackingResult, VideoMeta, VideoProcessor


class TemporaryTest(unittest.TestCase):
    def enterContext(self, manager):
        # unittest.TestCase.enterContext was added after the supported Python 3.10.
        value = manager.__enter__()
        self.addCleanup(manager.__exit__, None, None, None)
        return value

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="flowhmr_static_camera_")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))


class StaticCameraResultTests(TemporaryTest):
    def test_identity_pose_original_frame_intrinsics_and_saved_metadata(self):
        path = self.root / "nested/static_camera.npz"
        with patch.object(camera, "_load_vggt_model", side_effect=AssertionError("VGGT must not load")):
            result = camera.make_static_camera((60, 80), frame_range=(5, 11), output_npz=str(path))
        expected_k = np.array([[100, 0, 40], [0, 100, 30], [0, 0, 1]], dtype=np.float32)
        np.testing.assert_array_equal(result.K, np.repeat(expected_k[None], 7, axis=0))
        np.testing.assert_array_equal(result.camera_RT, np.repeat(np.eye(4)[None], 7, axis=0))
        np.testing.assert_array_equal(result.RT_vggt, result.camera_RT)
        self.assertEqual(result.image_size_hw, (60, 80))
        self.assertEqual(result.K.dtype, np.float32)
        self.assertEqual(result.camera_RT.dtype, np.float32)
        with np.load(path, allow_pickle=False) as saved:
            self.assertEqual(str(saved["camera_mode"]), "static")
            self.assertEqual(str(saved["k_space"]), "original_frame")
            np.testing.assert_array_equal(saved["frame_range"], [5, 11])
            np.testing.assert_array_equal(saved["image_size_hw"], [60, 80])
            np.testing.assert_array_equal(saved["K"], result.K)
            np.testing.assert_array_equal(saved["RT"], result.camera_RT)
            np.testing.assert_array_equal(saved["camera_RT"], result.camera_RT)
        yup_rt = camera._vggt_opencv_to_yup_rt(torch.from_numpy(result.camera_RT))
        features = inference.build_feature_dict(torch.zeros(7, 3072), yup_rt, torch.device("cpu"))
        torch.testing.assert_close(features["camera_R"].reshape(7, 3, 3),
                                   torch.eye(3).repeat(7, 1, 1), rtol=0, atol=0)

    def test_matching_cache_preserves_mtime_but_range_and_dimensions_rebuild(self):
        path = self.root / "camera.npz"
        camera.make_static_camera((60, 80), frame_range=(5, 11), output_npz=path)
        os.utime(path, ns=(1_000_000_000, 1_000_000_000))
        before = path.read_bytes(), path.stat().st_mtime_ns
        camera.make_static_camera((60, 80), frame_range=(5, 11), output_npz=path)
        self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)
        # Same sequence length must not hide a changed absolute frame range.
        camera.make_static_camera((60, 80), frame_range=(6, 12), output_npz=path)
        with np.load(path) as saved:
            np.testing.assert_array_equal(saved["frame_range"], [6, 12])
        self.assertNotEqual(path.stat().st_mtime_ns, before[1])
        os.utime(path, ns=(1_000_000_000, 1_000_000_000))
        result = camera.make_static_camera((120, 160), frame_range=(6, 12), output_npz=path)
        self.assertNotEqual(path.stat().st_mtime_ns, 1_000_000_000)
        np.testing.assert_array_equal(result.K[0], [[200, 0, 80], [0, 200, 60], [0, 0, 1]])

    def test_wrong_mode_intrinsics_or_motion_are_not_accepted_as_static_cache(self):
        path = self.root / "camera.npz"
        for broken_field in ("camera_mode", "K", "camera_RT"):
            with self.subTest(broken_field=broken_field):
                camera.make_static_camera((60, 80), frame_range=(0, 2), output_npz=path)
                with np.load(path) as saved:
                    payload = {key: saved[key].copy() for key in saved.files}
                if broken_field == "camera_mode":
                    payload[broken_field] = "vggt"
                elif broken_field == "K":
                    payload[broken_field][1, 0, 0] = 999
                else:
                    payload[broken_field][1, 0, 3] = 1
                np.savez(path, **payload)
                os.utime(path, ns=(1_000_000_000, 1_000_000_000))
                result = camera.make_static_camera((60, 80), frame_range=(0, 2), output_npz=path)
                self.assertNotEqual(path.stat().st_mtime_ns, 1_000_000_000)
                with np.load(path) as saved:
                    self.assertEqual(str(saved["camera_mode"]), "static")
                    np.testing.assert_array_equal(saved["K"], result.K)
                    np.testing.assert_array_equal(saved["camera_RT"], result.camera_RT)

    def test_invalid_dimensions_or_frame_range_are_rejected(self):
        for size, frame_range in (((0, 80), (0, 2)), ((60, -1), (0, 2)),
                                  ((60, 80), (-1, 2)), ((60, 80), (3, 2))):
            with self.subTest(size=size, frame_range=frame_range), self.assertRaises(ValueError):
                camera.make_static_camera(size, frame_range=frame_range)

    def test_truncated_static_cache_is_rebuilt(self):
        path = self.root / "camera.npz"
        expected = camera.make_static_camera((60, 80), frame_range=(5, 11), output_npz=path)
        complete = path.read_bytes()
        path.write_bytes(complete[:len(complete) // 2])
        actual = camera.make_static_camera((60, 80), frame_range=(5, 11), output_npz=path)
        np.testing.assert_array_equal(actual.camera_RT, expected.camera_RT)
        np.testing.assert_array_equal(actual.K, expected.K)
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["frame_range"], [5, 11])
            np.testing.assert_array_equal(saved["K"], expected.K)


class StaticCameraPipelineTests(TemporaryTest):
    def setUp(self):
        super().setUp()
        self.output = self.root / "output"
        self.folder = self.output / "clip"
        self.meta = VideoMeta(width=80, height=60, fps=30, total_frames=12)
        frames = np.arange(2, 10)
        boxes = np.array([[frame + 2, 5, frame + 40, 50] for frame in frames], dtype=np.float32)
        track = TrackInfo(track_id=7, frames=frames, bboxes=boxes, length=8, avg_area=1710)
        tracking = TrackingResult(tracks=[track], total_frames=12, width=80, height=60)
        self.engine = object.__new__(inference.V2MInferenceEngine)
        self.engine.device = torch.device("cpu")
        self.engine.postprocess = None
        self.engine.proc = VideoProcessor(detector=None)
        self.engine.proc.transcode = Mock(return_value="clip_30fps.mp4")
        self.engine.proc.detect_and_track = Mock(return_value=tracking)
        self.sam_inputs, self.generated = [], []
        self.engine.sam_extractor = SimpleNamespace(extract_video_tokens=Mock(side_effect=self.fake_sam))
        self.engine.pipeline = SimpleNamespace(generate=Mock(side_effect=self.fake_generate))
        self.engine.smpl_mesh = Mock(return_value={"vertices": torch.zeros(1, 4, 3)})
        self.enterContext(patch.object(VideoProcessor, "get_video_meta", side_effect=lambda _: self.meta))
        self.enterContext(patch.object(camera, "_load_vggt_model", side_effect=AssertionError("No VGGT weights")))
        self.enterContext(patch.object(camera, "load_or_extract_vggt_camera_rt",
                                      side_effect=AssertionError("Generation must reuse Stage 2 camera")))
        self.vggt = self.enterContext(patch.object(camera, "extract_vggt_camera", side_effect=self.fake_vggt))
        self.export = self.enterContext(patch.object(inference, "save_fbx"))
        self.progress = Mock()
        self.allow_vggt = False

    def fake_vggt(self, _video, *, output_npz, frame_range, expected_len, **kwargs):
        self.assertTrue(self.allow_vggt, "Static mode called VGGT extraction")
        self.assertEqual(expected_len, frame_range[1] - frame_range[0] + 1)
        if Path(output_npz).exists():
            with np.load(output_npz) as data:
                return camera.VGGTCameraResult(data["camera_RT"], data["camera_RT"], data["K"], (60, 80))
        rt = np.repeat(np.eye(4, dtype=np.float32)[None], expected_len, axis=0)
        for index, angle in enumerate(np.arange(expected_len) * 0.1):
            c, s = np.cos(angle), np.sin(angle)
            rt[index, :3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        k = np.repeat(np.array([[[200, 0, 40], [0, 210, 30], [0, 0, 1]]], dtype=np.float32), expected_len, axis=0)
        np.savez(output_npz, camera_RT=rt, K=k, frame_range=frame_range,
                 image_size_hw=[60, 80], k_space="original_frame")
        return camera.VGGTCameraResult(rt, rt, k, (60, 80))

    def fake_sam(self, _video, *, bbox_xyxy, K_all, max_frames, **kwargs):
        self.sam_inputs.append({"boxes": bbox_xyxy.clone(), "K": K_all.clone(), "max_frames": max_frames})
        tokens = torch.zeros(max_frames, 3072)
        tokens[:, 0] = torch.arange(max_frames)
        tokens[:, 1] = K_all[:max_frames, 0, 0]
        return tokens

    def fake_generate(self, *, feature, length, seeds, cfg_scale, step_callback=None):
        if step_callback is not None:
            step_callback()
        self.generated.append({"feature": {key: value.clone() for key, value in feature.items()},
                               "length": length, "seeds": seeds, "cfg_scale": cfg_scale})
        return {"rot6d": torch.tensor([1., 0, 0, 0, 1., 0]).repeat(1, length, 52, 1),
                "trans": torch.zeros(1, length, 3), "shapes": torch.zeros(1, 10)}

    def run_pipeline(self, *, static=True, max_frames=None, **kwargs):
        self.allow_vggt = not static
        return self.engine.process_video("clip.mp4", str(self.output), static_camera=static,
                                         max_frames=max_frames, seed=3, cfg_scale=1.5,
                                         progress_callback=self.progress, **kwargs)

    def test_static_pipeline_preserves_nonzero_frame_indices_and_limit(self):
        result = self.run_pipeline(max_frames=7)
        self.assertEqual(result, str(self.folder))
        self.vggt.assert_not_called()
        self.engine.proc.detect_and_track.assert_called_once()
        generated = self.generated[-1]
        self.assertEqual((generated["length"], generated["seeds"], generated["cfg_scale"]), (5, [3], 1.5))
        torch.testing.assert_close(generated["feature"]["feature"][0, :, 0], torch.arange(2, 7).float())
        torch.testing.assert_close(generated["feature"]["feature"][0, :, 1], torch.full((5,), 100.))
        torch.testing.assert_close(generated["feature"]["camera_R"].reshape(5, 3, 3),
                                   torch.eye(3).repeat(5, 1, 1), rtol=0, atol=0)
        self.assertEqual(self.sam_inputs[-1]["max_frames"], 7)
        sam_kwargs = self.engine.sam_extractor.extract_video_tokens.call_args.kwargs
        self.assertEqual(sam_kwargs["frame_batch_size"], 16)
        self.assertEqual(tuple(self.sam_inputs[-1]["boxes"].shape), (12, 4))
        self.assertEqual(tuple(self.sam_inputs[-1]["K"].shape), (12, 3, 3))
        with np.load(self.folder / "clip_static_camera.npz") as data:
            np.testing.assert_array_equal(data["frame_range"], [2, 6])
            self.assertEqual(data["K"].shape, (5, 3, 3))
        self.assertEqual(torch.load(self.folder / "clip_sam3d_feat_static.pt", weights_only=True).shape, (5, 3072))
        self.export.assert_called_once()
        self.assertEqual(self.export.call_args.kwargs["length"], 5)
        self.assertEqual(self.progress.call_args.args[:2], (inference.StageName.GENERATION, 100))

    def test_custom_sam_batches_reach_extractor(self):
        self.run_pipeline(sam_chunk_size=32)
        kwargs = self.engine.sam_extractor.extract_video_tokens.call_args.kwargs
        self.assertEqual(kwargs["frame_batch_size"], 32)

    def test_invalid_sam_cache_recomputes_in_both_camera_modes(self):
        for static in (False, True):
            for corruption in ("truncated", "wrong_type", "wrong_shape"):
                with self.subTest(static=static, corruption=corruption):
                    self.run_pipeline(static=static)
                    suffix = "_static" if static else ""
                    cache = self.folder / f"clip_sam3d_feat{suffix}.pt"
                    if corruption == "truncated":
                        complete = cache.read_bytes()
                        cache.write_bytes(complete[:len(complete) // 2])
                    elif corruption == "wrong_type":
                        torch.save({"tokens": torch.zeros(8, 3072)}, cache)
                    else:
                        torch.save(torch.zeros(8, 2048), cache)
                    previous_sam_calls = len(self.sam_inputs)
                    self.run_pipeline(static=static)
                    self.assertEqual(len(self.sam_inputs), previous_sam_calls + 1)
                    recovered = torch.load(cache, map_location="cpu", weights_only=True)
                    self.assertEqual(recovered.shape, (8, 3072))
                    torch.testing.assert_close(recovered[:, 0], torch.arange(2, 10).float())

    def test_invalid_sam_batches_fail_before_preprocessing(self):
        for key, value in (("sam_chunk_size", 0), ("sam_chunk_size", True),
                           ("sam_chunk_size", 1.5)):
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                self.run_pipeline(**{key: value})
        self.engine.proc.transcode.assert_not_called()

    def test_switching_camera_modes_keeps_separate_reusable_features(self):
        self.run_pipeline(static=False)
        moving_paths = [self.folder / "clip_vggt_camera.npz", self.folder / "clip_sam3d_feat.pt"]
        moving_cache = [(path.read_bytes(), path.stat().st_mtime_ns) for path in moving_paths]
        moving_features = self.generated[-1]["feature"]
        self.assertFalse(torch.allclose(moving_features["camera_R"][0, -1], torch.eye(3).flatten()))
        self.run_pipeline(static=True)
        self.assertEqual(self.vggt.call_count, 1)
        self.assertEqual(len(self.sam_inputs), 2)
        self.assertEqual(self.generated[-1]["feature"]["feature"][0, 0, 1].item(), 100.)
        for path, before in zip(moving_paths, moving_cache):
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)
        static_paths = [self.folder / "clip_static_camera.npz", self.folder / "clip_sam3d_feat_static.pt"]
        static_stamps = [path.stat().st_mtime_ns for path in static_paths]
        self.run_pipeline(static=True)
        self.assertEqual(len(self.sam_inputs), 2)
        self.assertEqual([path.stat().st_mtime_ns for path in static_paths], static_stamps)
        self.run_pipeline(static=False)
        self.assertEqual(len(self.sam_inputs), 2)
        torch.testing.assert_close(self.generated[-1]["feature"]["feature"], moving_features["feature"])
        torch.testing.assert_close(self.generated[-1]["feature"]["camera_R"], moving_features["camera_R"])

    def test_changing_max_frames_invalidates_both_modes_and_restores_full_track(self):
        self.run_pipeline(static=False)
        self.run_pipeline(static=True)
        self.run_pipeline(static=True, max_frames=7)
        self.assertFalse((self.folder / "clip_vggt_camera.npz").exists())
        self.assertFalse((self.folder / "clip_sam3d_feat.pt").exists())
        self.assertEqual(self.generated[-1]["length"], 5)
        self.run_pipeline(static=False, max_frames=None)
        self.assertEqual(self.generated[-1]["length"], 8)
        self.assertFalse((self.folder / "clip_static_camera.npz").exists())
        self.assertFalse((self.folder / "clip_sam3d_feat_static.pt").exists())
        self.assertEqual(self.engine.proc.detect_and_track.call_count, 1)
        torch.testing.assert_close(self.generated[-1]["feature"]["feature"][0, :, 0], torch.arange(2, 10).float())

    def test_changed_image_dimensions_recompute_static_features(self):
        self.run_pipeline()
        camera_path = self.folder / "clip_static_camera.npz"
        os.utime(camera_path, ns=(1_000_000_000, 1_000_000_000))
        self.meta = VideoMeta(width=160, height=120, fps=30, total_frames=12)
        self.run_pipeline()
        self.assertEqual(len(self.sam_inputs), 2)
        self.assertEqual(self.generated[-1]["feature"]["feature"][0, 0, 1].item(), 200.)
        self.vggt.assert_not_called()

    def test_unknown_frame_count_is_counted_without_vggt(self):
        self.meta.total_frames = 0

        def capture(_path):
            frames = iter([True] * 12 + [False])
            return SimpleNamespace(read=lambda: (next(frames, False), None), release=Mock())

        with patch.object(inference.cv2, "VideoCapture", side_effect=capture):
            self.run_pipeline(max_frames=7)
        self.assertEqual(self.sam_inputs[-1]["K"].shape[0], 12)
        self.assertEqual(self.generated[-1]["length"], 5)
        self.vggt.assert_not_called()


class StaticCameraCliTests(TemporaryTest):
    def setUp(self):
        super().setUp()
        self.ckpt = self.root / "model.ckpt"
        self.ckpt.touch()
        self.video = self.root / "clip.mp4"
        self.video.touch()
        self.output = self.root / "output"
        self.enterContext(patch.object(cli, "REPO_ROOT", str(self.root)))
        self.enterContext(patch.object(cli, "OUTPUT_ROOT", str(self.output)))
        for relative in cli.REQUIRED_FILES:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        self.vggt_weight = self.root / "ckpts/vggt-omega/vggt_omega_1b_512.pt"

    def parse(self, *extra):
        with patch.object(sys, "argv", ["run_v2m_demo_generation.py", str(self.video),
                                        "--ckpt", str(self.ckpt), "--device", "cpu", *extra]):
            return cli.parse_args()

    def test_cli_default_and_explicit_static_flag(self):
        self.assertFalse(self.parse().static_camera)
        self.assertTrue(self.parse("--static_camera").static_camera)
        args = self.parse()
        self.assertEqual(args.sam_chunk_size, 16)

    def test_cli_rejects_invalid_sam_batches(self):
        for option, value in (("--sam_chunk_size", "0"), ("--sam_chunk_size", "1.5"),
                              ("--sam_chunk_size", "-2")):
            with self.subTest(option=option, value=value), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as status:
                self.parse(option, value)
            self.assertEqual(status.exception.code, 2)

    def test_static_mode_skips_only_vggt_weight_check(self):
        self.vggt_weight.unlink()
        cli.check_files(str(self.ckpt), static_camera=True)
        with self.assertRaisesRegex(SystemExit, "vggt_omega"):
            cli.check_files(str(self.ckpt))
        sam = self.root / "ckpts/sam-3d-body-dinov3/model.ckpt"
        sam.unlink()
        with self.assertRaisesRegex(SystemExit, "sam-3d-body-dinov3"):
            cli.check_files(str(self.ckpt), static_camera=True)
        sam.touch()
        self.ckpt.unlink()
        with self.assertRaisesRegex(SystemExit, "model.ckpt"):
            cli.check_files(str(self.ckpt), static_camera=True)

    def make_engine(self, expected_mode, removed_paths=()):
        def process(video, output, *, seed, static_camera, **kwargs):
            self.assertEqual(static_camera, expected_mode)
            for path in removed_paths:
                self.assertFalse(path.exists(), f"Overwrite retained {path.name}")
            folder = Path(output) / Path(video).stem
            folder.mkdir(parents=True, exist_ok=True)
            np.savez(folder / f"clip_seed{seed}_origin.npz", trans=np.zeros((3, 3)))
        return SimpleNamespace(process_video=Mock(side_effect=process), postprocess=None)

    def test_overwrite_removes_both_camera_modes_and_passes_flag(self):
        folder = self.output / "clip"
        folder.mkdir(parents=True)
        paths = [folder / f"clip_{suffix}" for suffix in (
            "30fps.mp4", "bbox.npz", "vggt_camera.npz", "static_camera.npz",
            "sam3d_feat.pt", "sam3d_feat_static.pt")]
        for path in paths:
            path.write_bytes(b"old cache")
        unrelated = folder / "keep.txt"
        unrelated.write_text("keep")
        engine = self.make_engine(True, paths)
        result = cli.run_video(engine, str(self.video), self.parse("--static_camera", "--overwrite"), "latest")
        self.assertTrue(Path(result).is_file())
        self.assertEqual(unrelated.read_text(), "keep")
        self.assertTrue(engine.process_video.call_args.kwargs["static_camera"])

    def test_main_propagates_static_mode_through_checks_and_generation(self):
        self.vggt_weight.unlink()
        engine = self.make_engine(True)
        args = self.parse("--static_camera", "--sam_chunk_size", "32")
        with patch.object(cli, "parse_args", return_value=args), \
                patch.object(cli.os, "chdir"), \
                patch.object(inference.V2MInferenceEngine, "build", return_value=engine) as build, \
                self.assertRaises(SystemExit) as exit_status:
            cli.main()
        self.assertEqual(exit_status.exception.code, 0)
        build.assert_called_once()
        self.assertEqual(build.call_args.kwargs["device"], "cpu")
        engine.process_video.assert_called_once()
        self.assertTrue(engine.process_video.call_args.kwargs["static_camera"])
        self.assertEqual(engine.process_video.call_args.kwargs["sam_chunk_size"], 32)


if __name__ == "__main__":
    unittest.main(verbosity=2)

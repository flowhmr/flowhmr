"""CPU scheduling checks with real SAM crop geometry and lightweight decoders."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np
import torch
from torchvision.transforms import ToTensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "sam-3d-body"))

from sam_3d_body.data.transforms import (
    Compose, GetBBoxCenterScale, TopdownAffine, VisionTransformWrapper,
)
from sam_3d_body.data.utils.prepare_batch import prepare_batch
from sam_3d_body.models.meta_arch.base_model import BaseModel
from sam_3d_body.models.meta_arch.sam3d_body import SAM3DBody

from flowhmr.utils.lite_sam_infer import extract_tokens
from flowhmr.utils.runtime_sam_features import Sam3DTokenExtractor, build_sam3d_extractor
from flowhmr.utils.runtime_sam_worker import SamWorkerPool, _do_extract
from flowhmr.utils.sam_gpu_preprocess import prepare_gpu_batch


class FakeDecoder(torch.nn.Module):
    def forward(self, batch):
        # Encode each frame's actual camera, crop and pixels into its token.
        # This makes camera reuse, incorrect flips and reordered crops observable.
        fields = torch.cat([
            batch["cam_int"].flatten(1),
            batch["bbox"].flatten(1),
            batch["img"].flatten(3).mean(-1).flatten(1),
        ], dim=1)
        return fields.repeat(1, 64)[:, :1024, None].transpose(1, 2), {}


class FakeModel:
    _initialize_batch = BaseModel._initialize_batch
    _flatten_person = BaseModel._flatten_person

    def __init__(self):
        self.decoder = FakeDecoder()
        self.decoder_hand = FakeDecoder()
        self.cfg = SimpleNamespace(MODEL=SimpleNamespace(IMAGE_SIZE=(32, 32)))
        self.calls = []
        self.hand_postprocessing_calls = 0
        self.fail_next_hand = False
        self.hand_geometry = []

    def forward_step(self, batch, decoder_type="body"):
        batch_size, persons = batch["img"].shape[:2]
        self.calls.append((decoder_type, batch_size, persons))
        self.body_batch_idx = list(range(batch_size * persons)) if decoder_type == "body" else []
        self.hand_batch_idx = list(range(batch_size * persons)) if decoder_type == "hand" else []
        if decoder_type == "hand":
            if self.fail_next_hand:
                self.fail_next_hand = False
                raise RuntimeError("injected hand failure")
            self.decoder_hand(batch)
            self.hand_postprocessing_calls += 1
        else:
            self.decoder(batch)
        return {"mhr": {"hand_box": torch.tensor([
            [0.30, 0.40, 0.20, 0.18], [0.70, 0.60, 0.16, 0.22],
        ]).repeat(batch_size * persons, 1, 1)}}

    def _get_hand_box(self, output, batch):
        self.hand_geometry.append(batch["affine_trans"].clone())
        return SAM3DBody._get_hand_box(self, output, batch)


class FakeEstimator:
    def __init__(self):
        self.model = FakeModel()
        self.device = torch.device("cpu")
        self.faces = np.array([[0, 1, 2]])
        self.transform = self.make_transform(1.25)
        self.transform_hand = self.make_transform(0.9)
        self.full_calls = 0
        self.fusion_calls = 0

    @staticmethod
    def make_transform(padding):
        return Compose([
            GetBBoxCenterScale(padding=padding),
            TopdownAffine(input_size=(32, 32)),
            VisionTransformWrapper(ToTensor()),
        ])

    def process_one_image(self, img, *, bboxes, cam_int, **kwargs):
        # Match the full estimator's body/left/right/refinement order while
        # keeping a recognizable mesh payload for visualization consumers.
        self.full_calls += 1
        body = prepare_batch(img, self.transform, bboxes, cam_int=cam_int)
        self.model._initialize_batch(body)
        output = self.model.forward_step(body)
        left, right = self.model._get_hand_box(output, body)
        left[:, [0, 2]] = img.shape[1] - left[:, [2, 0]] - 1
        for image, boxes in ((img[:, ::-1], left), (img, right)):
            hand = prepare_batch(image, self.transform_hand, boxes, cam_int=cam_int)
            self.model.forward_step(hand, decoder_type="hand")
        self.fusion_calls += 1
        self.model.forward_step(body)
        return [{"pred_vertices": np.ones((3, 3)), "full_pose": True}]


class LiteSAMInferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.estimator = FakeEstimator()
        self.images, self.boxes, self.cameras = self.make_inputs(9)

    @staticmethod
    def make_inputs(count):
        rng = np.random.default_rng(42)
        images = [rng.integers(0, 256, (48 + i, 64 + i, 3), dtype=np.uint8) for i in range(count)]
        boxes = torch.tensor([[i + 2., i + 1., 42. + i * 2, 41. + i] for i in range(count)])
        cameras = torch.stack([
            torch.tensor([[80. + i, 0, 30. + i], [0, 90. + 2 * i, 20. + i], [0, 0, 1.]])
            for i in range(count)
        ])
        return images, boxes, cameras

    def single_frame_reference(self, images=None, boxes=None, cameras=None):
        if images is None:
            images, boxes, cameras = self.images, self.boxes, self.cameras
        wrapper = Sam3DTokenExtractor(FakeEstimator())
        try:
            return torch.stack([
                wrapper.extract_frame(img, bboxes=box[None].numpy(), cam_int=cam[None])[0]
                for img, box, cam in zip(images, boxes, cameras)
            ])
        finally:
            wrapper.close()

    def test_explicit_cpu_single_frame_uses_original_estimator_and_stops_before_fusion(self):
        with patch.object(self.estimator, "process_one_image", wraps=self.estimator.process_one_image) as entry:
            actual = extract_tokens(self.estimator, self.images[:1], self.boxes[:1], self.cameras[:1])
        entry.assert_called_once()
        self.assertIs(entry.call_args.args[0], self.images[0])
        np.testing.assert_array_equal(entry.call_args.kwargs["bboxes"], self.boxes[:1].numpy())
        torch.testing.assert_close(entry.call_args.kwargs["cam_int"], self.cameras[:1], rtol=0, atol=0)
        self.assertEqual(entry.call_args.kwargs["inference_type"], "full")
        torch.testing.assert_close(actual, self.single_frame_reference()[:1], rtol=0, atol=0)
        # Left-hand output still completes, so the unchanged estimator can
        # reach the right-hand decoder. Its fusion/refinement never executes.
        self.assertEqual(self.estimator.model.hand_postprocessing_calls, 1)
        self.assertEqual(self.estimator.fusion_calls, 0)
        self.assertEqual(self.estimator.model.calls, [("body", 1, 1), ("hand", 1, 1), ("hand", 1, 1)])

    def test_merged_hands_keep_frame_order_cameras_and_all_geometry(self):
        actual = extract_tokens(self.estimator, self.images, self.boxes, self.cameras,
                                merge_hands=True)
        torch.testing.assert_close(actual, self.single_frame_reference(), rtol=0, atol=0)
        self.assertEqual(self.estimator.model.calls, [("body", 9, 1), ("hand", 18, 1)])
        self.assertEqual(self.estimator.full_calls, 0)
        self.assertEqual(self.estimator.model.hand_postprocessing_calls, 0)
        geometry = self.estimator.model.hand_geometry[0]
        self.assertEqual(tuple(geometry.shape), (1, 9, 2, 3))
        self.assertFalse(torch.equal(geometry[:, :1], geometry[:, 1:2]))

    def test_separate_hand_batches_preserve_per_frame_camera(self):
        actual = extract_tokens(self.estimator, self.images, self.boxes, self.cameras)
        torch.testing.assert_close(actual, self.single_frame_reference(), rtol=0, atol=0)
        self.assertEqual(self.estimator.model.calls, [("body", 9, 1), ("hand", 9, 1), ("hand", 9, 1)])

    def test_tensor_preprocessing_path_keeps_crops_and_hand_order_on_cpu(self):
        actual = extract_tokens(self.estimator, self.images, self.boxes, self.cameras,
                                gpu_preprocess=True, merge_hands=True)
        torch.testing.assert_close(actual, self.single_frame_reference(), rtol=0, atol=0)

    def test_failure_restores_hooks_and_model_state_then_full_inference_works(self):
        wrapper = Sam3DTokenExtractor(self.estimator)
        self.addCleanup(wrapper.close)
        model = self.estimator.model
        original = {"_batch_size": 7, "_max_num_person": 2,
                    "_person_valid": torch.tensor([True]), "body_batch_idx": [3]}
        for name, value in original.items():
            setattr(model, name, value)
        counts = (len(model.decoder._forward_hooks), len(model.decoder_hand._forward_hooks))
        model.fail_next_hand = True
        with self.assertRaisesRegex(RuntimeError, "injected hand failure"):
            wrapper.extract_frames(self.images[:2], self.boxes[:2], self.cameras[:2], frame_batch_size=2)
        self.assertEqual(counts, (len(model.decoder._forward_hooks), len(model.decoder_hand._forward_hooks)))
        for name, value in original.items():
            self.assertIs(getattr(model, name), value)
        self.assertFalse(hasattr(model, "hand_batch_idx"))
        tokens, outputs = wrapper.extract_frame(self.images[0], bboxes=self.boxes[:1].numpy(), cam_int=self.cameras[:1])
        self.assertEqual(tuple(tokens.shape), (3072,))
        self.assertTrue(outputs[0]["full_pose"])
        self.assertEqual(outputs[0]["pred_vertices"].shape, (3, 3))
        self.assertFalse(wrapper._capture_tokens)
        # Subsequent lite extraction must not append to full-output hook lists.
        wrapper.extract_frames(self.images[:1], self.boxes[:1], self.cameras[:1])
        self.assertEqual(wrapper._body_tokens, [])
        self.assertEqual(wrapper._hand_tokens, [])

    def test_cpu_single_frame_failure_removes_temporary_hooks_and_can_retry(self):
        model = self.estimator.model
        model.fail_next_hand = True
        with self.assertRaisesRegex(RuntimeError, "injected hand failure"):
            extract_tokens(self.estimator, self.images[:1], self.boxes[:1], self.cameras[:1])
        self.assertEqual(len(model.decoder._forward_hooks), 0)
        self.assertEqual(len(model.decoder_hand._forward_hooks), 0)
        for name in ("_batch_size", "_max_num_person", "_person_valid", "body_batch_idx", "hand_batch_idx"):
            self.assertFalse(hasattr(model, name), name)
        actual = extract_tokens(self.estimator, self.images[:1], self.boxes[:1], self.cameras[:1])
        torch.testing.assert_close(actual, self.single_frame_reference()[:1], rtol=0, atol=0)
        self.assertEqual(self.estimator.fusion_calls, 0)
        wrapper = Sam3DTokenExtractor(self.estimator)
        self.addCleanup(wrapper.close)
        _, outputs = wrapper.extract_frame(self.images[0], bboxes=self.boxes[:1].numpy(), cam_int=self.cameras[:1])
        self.assertEqual(self.estimator.fusion_calls, 1)
        self.assertTrue(outputs[0]["full_pose"])

    def test_default_wrapper_tensor_crops_keep_per_frame_cameras_and_tail_frame(self):
        images, boxes, cameras = self.make_inputs(17)
        wrapper = Sam3DTokenExtractor(self.estimator)
        self.addCleanup(wrapper.close)
        with patch("flowhmr.utils.sam_gpu_preprocess.prepare_gpu_batch", wraps=prepare_gpu_batch) as preprocess:
            actual = wrapper.extract_frames(images, boxes, cameras)
        torch.testing.assert_close(actual, self.single_frame_reference(images, boxes, cameras), rtol=0, atol=0)
        self.assertEqual(preprocess.call_count, 3 * len(images))
        self.assertEqual(self.estimator.model.calls, [
            ("body", 16, 1), ("hand", 32, 1),
            ("body", 1, 1), ("hand", 2, 1),
        ])
        self.assertEqual(self.estimator.full_calls, 0)

    def test_default_video_tensor_crops_preserve_rgb_order_max_frames_and_tail(self):
        images, boxes, cameras = self.make_inputs(17)
        reference = self.single_frame_reference(images, boxes, cameras)
        for limit, expected_progress in ((15, [0, 15, 15]), (17, [0, 16, 17, 17])):
            with self.subTest(max_frames=limit):
                estimator = FakeEstimator()
                wrapper = Sam3DTokenExtractor(estimator)
                self.addCleanup(wrapper.close)
                capture = SimpleNamespace(isOpened=lambda: True, release=unittest.mock.Mock())
                capture.read = unittest.mock.Mock(side_effect=[
                    (True, img[:, :, ::-1].copy()) for img in images
                ] + [(False, None)])
                progress = []
                with patch.object(wrapper, "_get_total_frames", return_value=len(images)), patch(
                    "flowhmr.utils.runtime_sam_features.cv2.VideoCapture", return_value=capture,
                ), patch("flowhmr.utils.sam_gpu_preprocess.prepare_gpu_batch", wraps=prepare_gpu_batch) as preprocess:
                    actual = wrapper.extract_video_tokens("fake.mp4", bbox_xyxy=boxes,
                        K_all=cameras, max_frames=limit, progress_cb=lambda *args: progress.append(args))
                torch.testing.assert_close(actual, reference[:limit], rtol=0, atol=0)
                self.assertEqual(preprocess.call_count, 3 * limit)
                self.assertEqual(capture.read.call_count, limit)
                capture.release.assert_called_once()
                self.assertEqual([item[1] for item in progress], expected_progress)
                self.assertEqual(progress[-1][2], limit)
                self.assertEqual(estimator.full_calls, 0)

    def test_chunks_preserve_tokens_hand_order_tail_and_defaults(self):
        images, boxes, cameras = self.make_inputs(33)
        wrapper = Sam3DTokenExtractor(self.estimator, use_gpu_preprocess=False)
        self.addCleanup(wrapper.close)
        reference = self.single_frame_reference(images, boxes, cameras)
        for chunk in (1, 4, 16, 32):
            with self.subTest(chunk=chunk):
                self.estimator.model.calls.clear()
                actual = wrapper.extract_frames(images, boxes, cameras, frame_batch_size=chunk)
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                expected_calls = []
                for start in range(0, len(images), chunk):
                    count = min(chunk, len(images) - start)
                    expected_calls.extend([("body", count, 1), ("hand", 2 * count, 1)])
                self.assertEqual(self.estimator.model.calls, expected_calls)
                self.assertEqual(wrapper.frame_batch_size, 16)
                self.assertEqual(self.estimator.full_calls, 0)

    def test_video_and_worker_overrides_use_two_hands_per_frame(self):
        wrapper = Sam3DTokenExtractor(self.estimator)
        self.addCleanup(wrapper.close)
        capture = SimpleNamespace(isOpened=lambda: True, release=unittest.mock.Mock())
        decoded = iter([
            (True, img[:, :, ::-1].copy()) for img in self.images
        ])
        capture.read = unittest.mock.Mock(side_effect=lambda: next(decoded, (False, None)))
        with patch.object(wrapper, "_get_total_frames", return_value=len(self.images)), patch(
            "flowhmr.utils.runtime_sam_features.cv2.VideoCapture", return_value=capture,
        ):
            video_tokens = wrapper.extract_video_tokens(
                "fake.mp4", bbox_xyxy=self.boxes, K_all=self.cameras,
                frame_batch_size=4,
            )
        expected_calls = [
            ("body", 4, 1), ("hand", 8, 1), ("body", 4, 1), ("hand", 8, 1),
            ("body", 1, 1), ("hand", 2, 1),
        ]
        self.assertEqual(self.estimator.model.calls, expected_calls)
        self.estimator.model.calls.clear()
        worker_tokens = _do_extract(wrapper, {
            "frames": self.images, "bboxes": self.boxes.numpy(), "Ks": self.cameras.numpy(),
            "frame_batch_size": 4,
        })
        self.assertEqual(self.estimator.model.calls, expected_calls)
        reference = self.single_frame_reference()
        torch.testing.assert_close(video_tokens, reference, rtol=0, atol=0)
        torch.testing.assert_close(worker_tokens, reference, rtol=0, atol=0)

    def test_worker_pool_transports_batch_settings(self):
        pool = SamWorkerPool([0])
        pool._started = True
        self.addCleanup(setattr, pool, "_started", False)
        with patch("flowhmr.utils.runtime_sam_worker.decode_video_frames", return_value=self.images), patch(
            "flowhmr.utils.runtime_sam_worker.socket.socket",
        ), patch("flowhmr.utils.runtime_sam_worker.send_msg") as send, patch(
            "flowhmr.utils.runtime_sam_worker.recv_msg",
            return_value={"tokens": np.zeros((len(self.images), 3072), dtype=np.float32)},
        ):
            pool.extract_video_tokens(
                "fake.mp4", bbox_xyxy=self.boxes, K_all=self.cameras,
                frame_batch_size=32,
            )
        request = send.call_args.args[1]
        self.assertEqual(request["frame_batch_size"], 32)
        self.assertEqual(set(request), {"cmd", "frames", "bboxes", "Ks", "token_dim", "frame_batch_size"})

    def test_invalid_frame_batches_fail_before_inference(self):
        wrapper = Sam3DTokenExtractor(self.estimator)
        self.addCleanup(wrapper.close)
        for value in (0, -2, 8.0, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "frame_batch_size"):
                wrapper.extract_frames(self.images, self.boxes, self.cameras, frame_batch_size=value)
        self.assertEqual(self.estimator.model.calls, [])

    def test_worker_builder_defaults_use_tensor_crops_and_preserve_tail_and_token_dimension(self):
        images, boxes, cameras = self.make_inputs(17)
        # Exercise the builder used by _worker_main without loading weights.
        with patch("flowhmr.utils.runtime_sam_features.load_sam_3d_body", return_value=(self.estimator.model, None)), patch(
            "flowhmr.utils.runtime_sam_features._maybe_convert_backbone_to_fp16",
        ), patch("flowhmr.utils.runtime_sam_features.SAM3DBodyEstimator", return_value=self.estimator), patch(
            "flowhmr.utils.runtime_sam_features.osp.exists", return_value=True,
        ):
            wrapper = build_sam3d_extractor(torch.device("cpu"))
        self.addCleanup(wrapper.close)
        with patch("flowhmr.utils.sam_gpu_preprocess.prepare_gpu_batch", wraps=prepare_gpu_batch) as preprocess:
            actual = _do_extract(wrapper, {"frames": images, "bboxes": boxes.numpy(),
                "Ks": cameras.numpy(), "token_dim": 2048})
        torch.testing.assert_close(actual, self.single_frame_reference(images, boxes, cameras)[:, :2048], rtol=0, atol=0)
        self.assertEqual(preprocess.call_count, 3 * len(images))
        self.assertEqual([size for kind, size, _ in self.estimator.model.calls if kind == "body"], [16, 1])
        self.assertEqual([size for kind, size, _ in self.estimator.model.calls if kind == "hand"], [32, 2])
        empty = _do_extract(wrapper, {"frames": [], "bboxes": [], "Ks": [], "token_dim": 1024})
        self.assertEqual(tuple(empty.shape), (0, 1024))

    def test_invalid_inputs_leave_no_temporary_hooks(self):
        for images, boxes, cameras in (([], [], []), (self.images[:2], self.boxes[:1], self.cameras[:2])):
            with self.subTest(count=len(images)), self.assertRaises(ValueError):
                extract_tokens(self.estimator, images, boxes, cameras)
        self.assertEqual(len(self.estimator.model.decoder._forward_hooks), 0)
        self.assertEqual(len(self.estimator.model.decoder_hand._forward_hooks), 0)

    def test_model_hooks_do_not_retain_discarded_wrapper_or_captured_tokens(self):
        wrapper = Sam3DTokenExtractor(self.estimator)
        wrapper.extract_frame(self.images[0], bboxes=self.boxes[:1].numpy(), cam_int=self.cameras[:1])
        wrapper_ref = weakref.ref(wrapper)
        token_refs = [weakref.ref(token) for token in wrapper._body_tokens + wrapper._hand_tokens]
        del wrapper
        # Keep the estimator/model alive: their hooks must not own the wrapper
        # or its last captured GPU buffers. No close() or cyclic GC is needed.
        self.assertIsNone(wrapper_ref())
        self.assertTrue(all(token_ref() is None for token_ref in token_refs))
        outputs = self.estimator.process_one_image(
            self.images[0], bboxes=self.boxes[:1].numpy(), cam_int=self.cameras[:1],
        )
        self.assertTrue(outputs[0]["full_pose"])


if __name__ == "__main__":
    unittest.main()

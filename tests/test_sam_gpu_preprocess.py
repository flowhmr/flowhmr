"""CPU parity checks against SAM's released OpenCV preprocessing (no weights)."""

import unittest

import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor

from sam_3d_body.data.transforms import (
    Compose,
    GetBBoxCenterScale,
    TopdownAffine,
    VisionTransformWrapper,
)
from sam_3d_body.data.utils.prepare_batch import prepare_batch

from flowhmr.utils.sam_gpu_preprocess import _warp_affine_rgb, prepare_gpu_batch


def crop_transform(padding=1.25, size=(96, 128), use_udp=False):
    return Compose(
        [
            GetBBoxCenterScale(padding=padding),
            TopdownAffine(input_size=size, use_udp=use_udp),
            VisionTransformWrapper(ToTensor()),
        ]
    )


class SamGPUPreprocessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.image = np.random.default_rng(2026).integers(
            0, 256, size=(127, 193, 3), dtype=np.uint8
        )
        cls.camera = torch.tensor([[[250.0, 0, 96.5], [0, 260.0, 63.5], [0, 0, 1]]])
        cls.boxes = np.array(
            [
                [20.4, 8.75, 160.8, 110.5],
                [-45.25, -15.0, 98.75, 154.5],
                [175.0, 120.0, 240.0, 190.0],
                [31.45, 43.75, 31.55, 43.95],
                [-200.0, -150.0, -120.0, -70.0],
            ],
            dtype=np.float32,
        )

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def assert_batch_equal(self, image, boxes, transform):
        reference = prepare_batch(image, transform, boxes, cam_int=self.camera)
        actual = prepare_gpu_batch(
            torch.from_numpy(image.copy()), boxes, self.camera.numpy(), transform
        )
        for key, value in actual.items():
            with self.subTest(field=key):
                torch.testing.assert_close(value, reference[key], rtol=0, atol=0)
                self.assertEqual(value.device.type, "cpu")
                self.assertEqual(value.dtype, torch.float32)

    def test_body_crop_matches_opencv_including_padding_and_tiny_boxes(self):
        self.assert_batch_equal(self.image, self.boxes, crop_transform())

    def test_left_hand_flip_matches_original_hand_preprocessing(self):
        boxes = self.boxes.copy()
        boxes[:, [0, 2]] = self.image.shape[1] - self.boxes[:, [2, 0]] - 1
        self.assert_batch_equal(
            self.image[:, ::-1].copy(), boxes, crop_transform(padding=0.9)
        )

    def test_random_boxes_and_square_output(self):
        rng = np.random.default_rng(917)
        top_left = rng.uniform((-100, -90), (210, 140), size=(24, 2))
        dimensions = rng.uniform((0.1, 0.1), (280, 250), size=(24, 2))
        boxes = np.concatenate([top_left, top_left + dimensions], axis=1).astype(np.float32)
        self.assert_batch_equal(self.image, boxes, crop_transform(size=(256, 256)))

    def test_udp_crop_uses_the_original_geometry(self):
        self.assert_batch_equal(self.image, self.boxes, crop_transform(use_udp=True))

    def test_rotated_affine_pixel_sampling(self):
        matrices = [
            cv2.getRotationMatrix2D((80.25, 61.5), angle, scale)
            for angle, scale in ((-29.1, 0.8), (19.75, 1.25), (0, 0.07))
        ]
        size = (96, 128)
        reference = torch.from_numpy(
            np.stack([cv2.warpAffine(self.image, m, size) for m in matrices])
        ).permute(0, 3, 1, 2).float().div(255.0)
        actual = _warp_affine_rgb(torch.from_numpy(self.image), matrices, size)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_rejects_unsupported_transforms_instead_of_silently_ignoring_them(self):
        transform = crop_transform()
        transform.transforms.append(lambda item: item)
        with self.assertRaisesRegex(ValueError, "custom transforms"):
            prepare_gpu_batch(torch.from_numpy(self.image), self.boxes, self.camera, transform)

    def test_rejects_invalid_inputs(self):
        for boxes in (np.empty((0, 4)), [[0, 0, 0, 1]], [[0, 0, np.nan, 1]]):
            with self.subTest(boxes=boxes), self.assertRaises(ValueError):
                prepare_gpu_batch(
                    torch.from_numpy(self.image), boxes, self.camera, crop_transform()
                )


if __name__ == "__main__":
    unittest.main()

"""Device-side crops for the unmodified SAM 3D Body inference transforms.

Crop geometry still comes from SAM's own transforms. Pixel sampling reproduces
OpenCV's fixed-point INTER_LINEAR affine warp, including zero padding and uint8
rounding, rather than introducing the different sampling rules of grid_sample.
"""

import cv2
import numpy as np
import torch


def _crop_operations(transform):
    from sam_3d_body.data.transforms import (
        Compose,
        GetBBoxCenterScale,
        TopdownAffine,
        VisionTransformWrapper,
    )
    from torchvision.transforms import ToTensor

    operations = getattr(transform, "transforms", ())
    expected = (GetBBoxCenterScale, TopdownAffine, VisionTransformWrapper)
    if (
        type(transform) is not Compose
        or len(operations) != len(expected)
        or any(type(op) is not cls for op, cls in zip(operations, expected))
        or type(operations[-1].transform) is not ToTensor
    ):
        raise ValueError(
            "GPU SAM crops require Compose([GetBBoxCenterScale, "
            "TopdownAffine, VisionTransformWrapper(ToTensor())]); "
            "custom transforms must use the original CPU preprocessing."
        )
    return operations[:2]


def _warp_affine_rgb(image, matrices, size):
    """Return NCHW float32 crops with the same pixel scale as ToTensor."""
    out_width, out_height = size
    inverse = np.stack([cv2.invertAffineTransform(matrix) for matrix in matrices])
    inverse = torch.as_tensor(inverse, dtype=torch.float64, device=image.device)
    columns = torch.arange(out_width, dtype=torch.float64, device=image.device)
    rows = torch.arange(out_height, dtype=torch.float64, device=image.device)

    # OpenCV rounds the horizontal and vertical contributions separately at
    # 10 fractional bits, then looks up bilinear weights at 5 fractional bits.
    coordinates = []
    for axis in range(2):
        horizontal = torch.round(inverse[:, axis, 0, None] * columns * 1024)
        vertical = torch.round(
            (inverse[:, axis, 1, None] * rows + inverse[:, axis, 2, None]) * 1024
        )
        coordinates.append(
            (horizontal.long()[:, None, :] + vertical.long()[:, :, None] + 16) >> 5
        )
    source_x, source_y = coordinates
    x0, y0 = source_x >> 5, source_y >> 5
    fraction_x, fraction_y = source_x & 31, source_y & 31

    source_height, source_width = image.shape[:2]
    source_pixels = image.reshape(-1, 3)
    accumulated = torch.zeros(
        (*source_x.shape, 3), dtype=torch.int32, device=image.device
    )
    for dy in (0, 1):
        y = y0 + dy
        weight_y = fraction_y if dy else 32 - fraction_y
        for dx in (0, 1):
            x = x0 + dx
            inside = (x >= 0) & (x < source_width) & (y >= 0) & (y < source_height)
            indices = y.clamp(0, source_height - 1) * source_width
            indices = indices + x.clamp(0, source_width - 1)
            weights = weight_y * (fraction_x if dx else 32 - fraction_x) * inside
            accumulated.add_(source_pixels[indices].int() * weights[..., None].int())

    # Quantize before scaling: directly interpolating float images would produce
    # different network inputs from the released uint8 OpenCV preprocessing.
    crops = ((accumulated + 512) >> 10).to(torch.uint8)
    return crops.permute(0, 3, 1, 2).contiguous().float().div_(255.0)


@torch.no_grad()
def prepare_gpu_batch(image, boxes, camera, transform):
    """Prepare SAM's (1, N, ...) batch on the RGB image tensor's device.

    ``image`` is an HWC uint8 tensor; ``boxes`` contains N xyxy boxes in this
    image's coordinates. The caller must flip the image and boxes together for
    the left hand, as in the original SAM inference. ``camera`` is (1, 3, 3) or
    (3, 3). No image normalization is applied here: SAM's data_preprocess handles
    it. Segmentation masks are absent, matching prepare_batch without masks.
    """
    if not isinstance(image, torch.Tensor) or (
        image.dtype != torch.uint8
        or image.ndim != 3
        or image.shape[-1] != 3
        or min(image.shape[:2]) == 0
    ):
        raise ValueError("image must be a nonempty HWC RGB uint8 tensor")
    operations = _crop_operations(transform)
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not len(boxes):
        raise ValueError("boxes must have shape (N, 4) with at least one person")
    if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("boxes must be finite xyxy boxes with positive width and height")
    camera = torch.as_tensor(camera, dtype=torch.float32, device=image.device)
    if camera.shape not in ((3, 3), (1, 3, 3)):
        raise ValueError("camera must have shape (3, 3) or (1, 3, 3)")

    geometry = []
    for box in boxes:
        item = {"bbox": box.copy(), "bbox_format": "xyxy"}
        for operation in operations:
            item = operation(item)
        geometry.append(item)
    width, height = map(int, geometry[0]["input_size"])
    if width <= 0 or height <= 0:
        raise ValueError("SAM crop output dimensions must be positive")

    batch = {
        key: torch.as_tensor(
            np.stack([item[key] for item in geometry]),
            dtype=torch.float32,
            device=image.device,
        ).unsqueeze(0)
        for key in ("bbox", "bbox_center", "bbox_scale", "affine_trans")
    }
    batch["img"] = _warp_affine_rgb(
        image, [item["affine_trans"] for item in geometry], (width, height)
    ).unsqueeze(0)
    count = len(boxes)
    for name, dimensions in (
        ("img_size", (width, height)),
        ("ori_img_size", (image.shape[1], image.shape[0])),
    ):
        batch[name] = torch.tensor(
            dimensions, dtype=torch.float32, device=image.device
        ).expand(1, count, 2)
    batch["mask"] = torch.zeros(
        (1, count, 1, height, width), dtype=torch.float32, device=image.device
    )
    batch["mask_score"] = torch.zeros((1, count), dtype=torch.float32, device=image.device)
    batch["person_valid"] = torch.ones((1, count), dtype=torch.float32, device=image.device)
    batch["cam_int"] = camera.reshape(1, 3, 3)
    return batch

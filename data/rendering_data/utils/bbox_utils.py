"""Bounding-box and projection utilities for SMPL annotations."""

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


def find_longest_segment(valid_frames_log):
    """Find the longest consecutive run of valid frames.

    Args:
        valid_frames_log: list of bools, one per frame

    Returns:
        (best_start, best_end, max_len): inclusive frame range and length
    """
    max_len = 0
    best_start = 0
    best_end = 0

    current_start = -1
    current_len = 0

    for i, is_valid in enumerate(valid_frames_log):
        if is_valid:
            if current_start == -1:
                current_start = i
            current_len += 1
        else:
            if current_start != -1:
                if current_len > max_len:
                    max_len = current_len
                    best_start = current_start
                    best_end = i - 1
                # reset
                current_start = -1
                current_len = 0

    if current_start != -1 and current_len > max_len:
        max_len = current_len
        best_start = current_start
        best_end = len(valid_frames_log) - 1

    return best_start, best_end, max_len


def project(points, cam_int):
    """Project camera-space points (N, 3) to the image plane with K (3, 3).

    Returns:
        (N, 2+) numpy array of pixel coordinates
    """
    projected_points = points / points[:, -1].unsqueeze(-1)
    projected_points = torch.einsum('ij, kj->ki', cam_int, projected_points.float())

    return projected_points.detach().cpu().numpy()



def moving_average_smooth(x, window_size=5, dim=-1):
    """Sliding-window moving average along any dimension of a tensor."""
    kernel_smooth = torch.ones(window_size).float() / window_size
    kernel_smooth = kernel_smooth[None, None].to(x)  # (1, 1, window_size)
    rad = kernel_smooth.size(-1) // 2

    x = x.transpose(dim, -1)
    x_shape = x.shape[:-1]
    x = rearrange(x, "... f -> (...) 1 f")  # (NB, 1, f)
    x = F.pad(x[None], (rad, rad, 0, 0), mode="replicate")[0]
    x = F.conv1d(x, kernel_smooth)
    x = x.squeeze(1).reshape(*x_shape, -1)  # (..., f)
    x = x.transpose(-1, dim)
    return x


def get_bbox_valid(joints, img_height, img_width, rescale):
    """Compute a rescaled bbox from 2D joints, ignoring out-of-image points.

    Args:
        joints: (N, 2+) pixel coordinates
        img_height: image height in pixels
        img_width: image width in pixels
        rescale: bbox scale factor around the center

    Returns:
        (center [cx, cy], scale, num_valid_joints, bbox_xyxy)
        All set to -1 when no joint is inside the image.
    """
    valid_j = []
    joints = np.copy(joints)
    for j in joints:
        if j[0] > img_width or j[1] > img_height or j[0] < 0 or j[1] < 0:
            continue
        else:
            valid_j.append(j)

    if len(valid_j) < 1:
        return [-1, -1], -1, len(valid_j), [-1, -1, -1, -1]

    joints = np.array(valid_j)

    bbox_xyxy = [min(joints[:, 0]), min(joints[:, 1]), max(joints[:, 0]), max(joints[:, 1])]

    center = [(bbox_xyxy[2] + bbox_xyxy[0]) / 2, (bbox_xyxy[3] + bbox_xyxy[1]) / 2]
    width = bbox_xyxy[2] - bbox_xyxy[0]
    height = bbox_xyxy[3] - bbox_xyxy[1]

    scaled_width = width * rescale
    scaled_height = height * rescale

    scaled_bbox_xyxy = [
        center[0] - scaled_width / 2,
        center[1] - scaled_height / 2,
        center[0] + scaled_width / 2,
        center[1] + scaled_height / 2
    ]

    scale = max(width, height) / 200
    scale *= rescale

    return center, scale, len(valid_j), scaled_bbox_xyxy

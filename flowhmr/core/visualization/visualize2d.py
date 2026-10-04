import os
import shutil

import cv2
import numpy as np
import torch
import subprocess

# fmt: off

HML_JOINT_NAMES = [
    "pelvis",  # 0
    "left_hip",  # 1
    "right_hip",  # 2
    "spine1",  # 3
    "left_knee",  # 4
    "right_knee",  # 5
    "spine2",  # 6
    "left_ankle",  # 7
    "right_ankle",  # 8
    "spine3",  # 9
    "left_foot",  # 10
    "right_foot",  # 11
    "neck",  # 12
    "left_collar",  # 13
    "right_collar",  # 14
    "head",  # 15
    "left_shoulder",  # 16
    "right_shoulder",  # 17
    "left_elbow",  # 18
    "right_elbow",  # 19
    "left_wrist",  # 20
    "right_wrist",  # 21
]

HML3D_KINEMATIC_CHAINS = [
    [0, 2, 5, 8, 11], # pelvis, right_hip, right_knee, right_ankle, right_foot
    [0, 1, 4, 7, 10], # pelvis, left_hip, left_knee, left_ankle, left_foot
    [0, 3, 6, 9, 12, 15], # pelvis, spine1, spine2, spine3, neck, head
    [9, 14, 17, 19, 21], # spine3, right_collar, right_shoulder, right_elbow, right_wrist
    [9, 13, 16, 18, 20], # spine3, left_collar, left_shoulder, left_elbow, left_wrist
]

COLORS = [
    "red", "blue", "black", "red", "blue",
    "darkblue", "darkblue", "darkblue", "darkblue", "darkblue",
    "darkred", "darkred", "darkred", "darkred", "darkred",
]

COLORS_RGB255 = [
    (255, 0, 0), (0, 0, 255), (0, 0, 0), (255, 0, 0), (0, 0, 255),
    (0, 0, 139), (0, 0, 139), (0, 0, 139), (0, 0, 139), (0, 0, 139),
    (139, 0, 0), (139, 0, 0), (139, 0, 0), (139, 0, 0), (139, 0, 0),
]
# fmt: on


# fmt: off
def get_available_encoder(ffmpeg):
    # VP9
    try:
        subprocess.run([
            ffmpeg, "-f", "lavfi", "-i", "testsrc=duration=0.1:size=32x32:rate=1",
            "-c:v", "libvpx-vp9",
            "-f", "null", "-"
        ], capture_output=True, check=True, timeout=5)

        return "webm", "libvpx-vp9", "-crf 30 -b:v 0"
    except:
        pass
    # VP8
    try:
        subprocess.run([
            ffmpeg, "-f", "lavfi", "-i", "testsrc=duration=0.1:size=32x32:rate=1",
            "-c:v", "libvpx",
            "-f", "null", "-"
        ], capture_output=True, check=True, timeout=5)
        return "webm", "libvpx", "-crf 10 -b:v 1M"
    except:
        pass
    # MPEG-4
    try:
        subprocess.run([
            ffmpeg, "-f", "lavfi", "-i", "testsrc=duration=0.1:size=32x32:rate=1",
            "-c:v", "mpeg4",
            "-f", "null", "-"
        ], capture_output=True, check=True, timeout=5)
        return "mp4", "mpeg4", "-q:v 3"
    except:
        pass
    return "mp4", "mpeg4", "-q:v 3"
# fmt: on


def plot_skeleton(vis_xy, positions_2d, line_width, hand_width, circle_radius, ground_height):
    positions_2d = np.clip(positions_2d, 0, vis_xy.shape[0] - 1)
    positions_2d[:, 1] = vis_xy.shape[0] - positions_2d[:, 1]
    p2d = positions_2d.astype(np.int32)
    for i, (chain, color) in enumerate(zip(HML3D_KINEMATIC_CHAINS, COLORS_RGB255)):
        for jj in range(len(chain) - 1):
            cv2.line(
                vis_xy,
                (p2d[chain[jj], 0], p2d[chain[jj], 1]),
                (p2d[chain[jj + 1], 0], p2d[chain[jj + 1], 1]),
                color,
                line_width,
            )
        # ax.plot3D(
    return vis_xy


def merge(images, row=-1, col=-1, resize=False, ret_range=False, square=False, **kwargs):
    if row == -1:
        row = int(np.sqrt(len(images)))
    if col == -1:
        col = int(np.ceil(len(images) / row))
    assert row == col, "row and col should be the same"
    height = images[0].shape[0]
    width = images[0].shape[1]
    # special case
    if height > width:
        if len(images) == 3:
            row, col = 1, 3
    if len(images[0].shape) > 2:
        ret_img = np.zeros((height * row, width * col, images[0].shape[2]), dtype=np.uint8) + 255
    else:
        ret_img = np.zeros((height * row, width * col), dtype=np.uint8) + 255
    ranges = []
    for i in range(row):
        for j in range(col):
            if i * col + j >= len(images):
                break
            img = images[i * col + j]
            # resize the image size
            img = cv2.resize(img, (width, height))
            ret_img[height * i : height * (i + 1), width * j : width * (j + 1)] = img
            ranges.append((width * j, height * i, width * (j + 1), height * (i + 1)))
    if resize:
        min_height = 1000
        if ret_img.shape[0] > min_height:
            scale = min_height / ret_img.shape[0]
            ret_img = cv2.resize(ret_img, None, fx=scale, fy=scale)
    if ret_range:
        return ret_img, ranges
    return ret_img


def add_multiline_text(image, text, position=(10, 0), font_scale=1, font_thickness=2, max_width=None):
    # Split text into multiple lines based on image width
    font_scale = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    max_width = image.shape[1] - 20 # Leave some margin

    # Calculate text width and split if necessary
    text_lines = []
    current_line = ""
    for word in text.split():
        test_line = current_line + " " + word if current_line else word
        (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)

        if text_width <= max_width:
            current_line = test_line
        else:
            text_lines.append(current_line)
            current_line = word

    if current_line:
        text_lines.append(current_line)

    # Draw each line of text
    y_position = 30
    for line in text_lines:
        image = cv2.putText(
            image, line, (position[0], position[1] + y_position), font, font_scale, (0, 0, 0), thickness
        )
        y_position += 30 # Move to next line
    return image


def add_border(image):
    cv2.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), (0, 0, 0), 1)
    return image


def visualize_skeleton(
    gt=None,
    pred=None,
    text=None,
    length=None,
    output_dir=None,
    vis_direction=[0, 1],
    bbox_size=1.0,
    vis_size=256,
    fps=30,
    format="mp4",
    text_of_each_seed=None,
    add_border_of_each_seed=False,
):
    if gt is not None:
        if isinstance(gt, torch.Tensor):
            gt = gt.cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    os.makedirs(output_dir, exist_ok=True)
    # gt: (1, L, J, 3)
    if gt is not None:
        min_height = gt[..., vis_direction[1]].min(axis=-1).mean()
    else:
        min_height = 0

    map_world_to_2d = lambda x: ((x + bbox_size) / (2 * bbox_size)) * vis_size

    if text_of_each_seed is not None:
        assert (
            len(text_of_each_seed) == pred.shape[0]
        ), f"text_of_each_seed must have the same length as pred, but got {len(text_of_each_seed)} and {pred.shape[0]}"

    for nf in range(length):
        vis_all = []
        for bs in range(pred.shape[0]):
            kpts = pred[bs, nf, :, :]
            positions_2d = map_world_to_2d(kpts)
            ground_height = int(map_world_to_2d(min_height))
            # visualize
            vis_xy = np.zeros((vis_size, vis_size, 3), dtype=np.uint8) + 255
            vis_xy = plot_skeleton(
                vis_xy,
                positions_2d[:, vis_direction],
                line_width=2 * vis_size // 256,
                hand_width=1 * vis_size // 256,
                circle_radius=2 * vis_size // 256,
                ground_height=ground_height,
            )
            if text_of_each_seed is not None:
                vis_xy = add_multiline_text(vis_xy, text_of_each_seed[bs], position=(10, 50))
            if add_border_of_each_seed:
                vis_xy = add_border(vis_xy)
            vis_all.append(vis_xy)
        vis_all = merge(vis_all)

        if gt is not None:
            kpts_gt = gt[nf]
            positions_2d_gt = map_world_to_2d(kpts_gt) * 2
            ground_height_gt = int(map_world_to_2d(min_height))
            vis_xy_gt = np.zeros((vis_all.shape[0], vis_all.shape[1], 3), dtype=np.uint8) + 255
            vis_xy_gt = plot_skeleton(
                vis_xy_gt,
                positions_2d_gt[:, vis_direction],
                line_width=2 * vis_size // 256,
                hand_width=1 * vis_size // 256,
                circle_radius=2 * vis_size // 256,
                ground_height=ground_height_gt,
            )
            vis_all = np.concatenate([vis_xy_gt, vis_all], axis=1)
        if text is not None:
            vis_all = add_multiline_text(vis_all, text)
        cv2.imwrite(f"{output_dir}/{nf:06d}.jpg", vis_all)
        assert os.path.exists(f"{output_dir}/{nf:06d}.jpg"), f"failed to save {f'{output_dir}/{nf:06d}.jpg'}: {vis_all.shape}"
    ffmpeg = "/usr/bin/ffmpeg"

    if format.lower() == "gif":
        gif_cmd = f'{ffmpeg} -y -loglevel error -framerate {fps} -i {output_dir}/%06d.jpg -vf "fps={fps},scale=512:-1:flags=lanczos" -loop 0 {output_dir}_vis.gif'
        print(gif_cmd)
        os.system(gif_cmd)

        output_file = f"{output_dir}_vis.gif"
        if os.path.exists(output_file):
            shutil.rmtree(output_dir)
        return output_file
    else:
        container, video_codec, params = get_available_encoder(ffmpeg)
        output_file = f"{output_dir}_vis.{container}"

        cmd = f"{ffmpeg} -y -loglevel error -framerate {fps} -i {output_dir}/%06d.jpg -c:v {video_codec} {params} -pix_fmt yuv420p -an {output_file}"
        print(cmd)
        os.system(cmd)

        if os.path.exists(output_file):
            shutil.rmtree(output_dir)
        return output_file

def visualize_keypoints_video(video_name, seed_output, proj_kp2d, kp2d_metrics, output_dir="test_2dkp/test_0105"):
    import cv2
    import numpy as np
    from ..evaluation.vertex_ids import coco133tobody25

    basename = os.path.basename(video_name).replace('.mp4', '')
    os.makedirs(output_dir, exist_ok=True)

    try:
        from decord import VideoReader
        vr = VideoReader(video_name)

        gt_kp2d_coco133 = seed_output["gt"]["keypoints3d"][0].cpu().numpy()  # (F, 133, 3)
        gt_kp2d_body25 = coco133tobody25(gt_kp2d_coco133)
        pred_kp2d_body25 = proj_kp2d[0]

        num_frames = min(len(vr), gt_kp2d_body25.shape[0], pred_kp2d_body25.shape[0])

        output_video_path = os.path.join(output_dir, f"{basename}_keypoints.mp4")
        fps = 30

        first_frame = vr[0].asnumpy()
        height, width = first_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Body25: 0-Nose, 1-Neck, 2-RShoulder, 3-RElbow, 4-RWrist, 5-LShoulder, 6-LElbow, 7-LWrist,
        #         15-REye, 16-LEye, 17-REar, 18-LEar, 19-24: feet points
        BODY25_SKELETON = [
            [0, 1], # Nose -> Neck
            [1, 2], [1, 5], # Neck -> Shoulders
            [1, 8], # Neck -> MidHip
            [8, 9], [8, 12], # MidHip -> Hips

            [2, 3], [3, 4], # RShoulder -> RElbow -> RWrist

            [5, 6], [6, 7], # LShoulder -> LElbow -> LWrist

            [9, 10], [10, 11], # RHip -> RKnee -> RAnkle
            [11, 22], [11, 24], [22, 23], # RAnkle -> feet

            [12, 13], [13, 14], # LHip -> LKnee -> LAnkle
            [14, 19], [14, 21], [19, 20], # LAnkle -> feet

            [0, 15], [0, 16], # Nose -> Eyes
            [15, 17], [16, 18], # Eyes -> Ears
        ]

        vis_frames = num_frames
        for f in range(vis_frames):
            frame = vr[f].asnumpy().copy()

            gt_xy = gt_kp2d_body25[f, :, :2]
            gt_conf = gt_kp2d_body25[f, :, 2]

            for connection in BODY25_SKELETON:
                pt1_idx, pt2_idx = connection
                if (pt1_idx < len(gt_conf) and pt2_idx < len(gt_conf) and
                    gt_conf[pt1_idx] > 0.3 and gt_conf[pt2_idx] > 0.3):
                    pt1 = (int(gt_xy[pt1_idx, 0]), int(gt_xy[pt1_idx, 1]))
                    pt2 = (int(gt_xy[pt2_idx, 0]), int(gt_xy[pt2_idx, 1]))
                    cv2.line(frame, pt1, pt2, (0, 200, 0), 2)

            pred_xy = pred_kp2d_body25[f]
            for connection in BODY25_SKELETON:
                pt1_idx, pt2_idx = connection
                if (pt1_idx < len(pred_xy) and pt2_idx < len(pred_xy)):
                    x1, y1 = pred_xy[pt1_idx]
                    x2, y2 = pred_xy[pt2_idx]
                    if (not (np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2)) and
                        0 <= x1 < width and 0 <= y1 < height and
                        0 <= x2 < width and 0 <= y2 < height):
                        pt1 = (int(x1), int(y1))
                        pt2 = (int(x2), int(y2))
                        cv2.line(frame, pt1, pt2, (0, 0, 200), 2)

            for i, (x, y, c) in enumerate(zip(gt_xy[:, 0], gt_xy[:, 1], gt_conf)):
                if c > 0.3:
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
                    if i in [0, 1, 2, 5, 8, 9, 12]: # nose, neck, shoulders, hip, hips
                        cv2.putText(frame, str(i), (int(x)+5, int(y)-5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            for i, (x, y) in enumerate(pred_xy):
                if not (np.isnan(x) or np.isnan(y)) and 0 <= x < width and 0 <= y < height:
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)
                    if i in [0, 1, 2, 5, 8, 9, 12]:
                        cv2.putText(frame, str(i), (int(x)+5, int(y)+5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            cv2.rectangle(frame, (10, 10), (350, 80), (0, 0, 0), -1)
            cv2.putText(frame, "GT (Green) vs Pred (Red)", (15, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Frame {f+1}/{vis_frames}", (15, 50),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Reproj Error: {kp2d_metrics['reproj_error_px']:.1f}px", (15, 70),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

        video_writer.release()

        print(f"Saved keypoint video: {output_video_path} ({vis_frames} frames)")
        return output_video_path

    except Exception as e:
        print(f"Video visualization failed for {basename}: {e}")
        return None
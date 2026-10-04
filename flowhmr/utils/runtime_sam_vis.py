from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np


def parse_pose_metainfo(metainfo: Dict) -> Dict:
    """Parse pose metainfo dict. Only accepts dict input (no file-path loading)."""
    assert isinstance(metainfo, dict), (
        f"parse_pose_metainfo only accepts dict, got {type(metainfo)}. "
        "File-path loading requires detectron2 and is not supported here."
    )

    assert "pose_format" in metainfo
    assert "keypoint_info" in metainfo
    assert "skeleton_info" in metainfo
    assert "joint_weights" in metainfo
    assert "sigmas" in metainfo

    parsed = dict(
        pose_format=None,
        num_keypoints=None,
        keypoint_id2name={},
        keypoint_name2id={},
        upper_body_ids=[],
        lower_body_ids=[],
        flip_indices=[],
        flip_pairs=[],
        keypoint_colors=[],
        num_skeleton_links=None,
        skeleton_links=[],
        skeleton_link_colors=[],
        dataset_keypoint_weights=None,
        sigmas=None,
    )

    parsed["pose_format"] = metainfo["pose_format"]

    for optional_key in [
        "remove_teeth", "min_visible_keypoints", "teeth_keypoint_ids",
        "coco_wholebody_to_goliath_mapping", "coco_wholebody_to_goliath_keypoint_info",
    ]:
        if optional_key in metainfo:
            parsed[optional_key] = metainfo[optional_key]

    parsed["num_keypoints"] = len(metainfo["keypoint_info"])

    for kpt_id, kpt in metainfo["keypoint_info"].items():
        kpt_name = kpt["name"]
        parsed["keypoint_id2name"][kpt_id] = kpt_name
        parsed["keypoint_name2id"][kpt_name] = kpt_id
        parsed["keypoint_colors"].append(kpt.get("color", [255, 128, 0]))

        kpt_type = kpt.get("type", "")
        if kpt_type == "upper":
            parsed["upper_body_ids"].append(kpt_id)
        elif kpt_type == "lower":
            parsed["lower_body_ids"].append(kpt_id)

        swap_kpt = kpt.get("swap", "")
        if swap_kpt == kpt_name or swap_kpt == "":
            parsed["flip_indices"].append(kpt_name)
        else:
            parsed["flip_indices"].append(swap_kpt)
            pair = (swap_kpt, kpt_name)
            if pair not in parsed["flip_pairs"]:
                parsed["flip_pairs"].append(pair)

    parsed["num_skeleton_links"] = len(metainfo["skeleton_info"])
    for _, sk in metainfo["skeleton_info"].items():
        parsed["skeleton_links"].append(sk["link"])
        parsed["skeleton_link_colors"].append(sk.get("color", [96, 96, 255]))

    parsed["dataset_keypoint_weights"] = np.array(metainfo["joint_weights"], dtype=np.float32)
    parsed["sigmas"] = np.array(metainfo["sigmas"], dtype=np.float32)

    if "stats_info" in metainfo:
        parsed["stats_info"] = {
            name: np.array(val, dtype=np.float32)
            for name, val in metainfo["stats_info"].items()
        }

    def _map(src, mapping: dict):
        if isinstance(src, (list, tuple)):
            cls = type(src)
            return cls(_map(s, mapping) for s in src)
        return mapping[src]

    parsed["flip_pairs"] = _map(parsed["flip_pairs"], mapping=parsed["keypoint_name2id"])
    parsed["flip_indices"] = _map(parsed["flip_indices"], mapping=parsed["keypoint_name2id"])
    parsed["skeleton_links"] = _map(parsed["skeleton_links"], mapping=parsed["keypoint_name2id"])

    parsed["keypoint_colors"] = np.array(parsed["keypoint_colors"], dtype=np.uint8)
    parsed["skeleton_link_colors"] = np.array(parsed["skeleton_link_colors"], dtype=np.uint8)

    return parsed


class SkeletonVisualizer:
    def __init__(
        self,
        bbox_color: Optional[Union[str, Tuple[int]]] = "green",
        kpt_color: Optional[Union[str, Tuple[Tuple[int]]]] = "red",
        link_color: Optional[Union[str, Tuple[Tuple[int]]]] = None,
        text_color: Optional[Union[str, Tuple[int]]] = (255, 255, 255),
        line_width: Union[int, float] = 1,
        radius: Union[int, float] = 3,
        alpha: float = 1.0,
        show_keypoint_weight: bool = False,
    ):
        self.bbox_color = bbox_color
        self.kpt_color = kpt_color
        self.link_color = link_color
        self.line_width = line_width
        self.text_color = text_color
        self.radius = radius
        self.alpha = alpha
        self.show_keypoint_weight = show_keypoint_weight
        self.pose_meta = {}
        self.skeleton = None

    def set_pose_meta(self, pose_meta: Dict):
        parsed_meta = parse_pose_metainfo(pose_meta)
        self.pose_meta = parsed_meta.copy()
        self.bbox_color = parsed_meta.get("bbox_color", self.bbox_color)
        self.kpt_color = parsed_meta.get("keypoint_colors", self.kpt_color)
        self.link_color = parsed_meta.get("skeleton_link_colors", self.link_color)
        self.skeleton = parsed_meta.get("skeleton_links", self.skeleton)

    def draw_skeleton(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        kpt_thr: float = 0.3,
    ):
        image = image.copy()
        img_h, img_w, _ = image.shape
        if len(keypoints.shape) == 2:
            keypoints = keypoints[None, :, :]

        for cur_keypoints in keypoints:
            kpts = cur_keypoints[:, :-1]
            score = cur_keypoints[:, -1]

            if self.kpt_color is None or isinstance(self.kpt_color, str):
                kpt_color = [self.kpt_color] * len(kpts)
            elif len(self.kpt_color) == len(kpts):
                kpt_color = self.kpt_color
            else:
                raise ValueError(
                    f"the length of kpt_color ({len(self.kpt_color)}) does not match "
                    f"that of keypoints ({len(kpts)})"
                )

            if self.skeleton is not None and self.link_color is not None:
                if isinstance(self.link_color, str):
                    link_color = [self.link_color] * len(self.skeleton)
                elif len(self.link_color) == len(self.skeleton):
                    link_color = self.link_color
                else:
                    raise ValueError(
                        f"the length of link_color ({len(self.link_color)}) does not match "
                        f"that of skeleton ({len(self.skeleton)})"
                    )

                for sk_id, sk in enumerate(self.skeleton):
                    pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                    pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))
                    if (
                        pos1[0] <= 0 or pos1[0] >= img_w
                        or pos1[1] <= 0 or pos1[1] >= img_h
                        or pos2[0] <= 0 or pos2[0] >= img_w
                        or pos2[1] <= 0 or pos2[1] >= img_h
                        or score[sk[0]] < kpt_thr
                        or score[sk[1]] < kpt_thr
                        or link_color[sk_id] is None
                    ):
                        continue
                    color = link_color[sk_id]
                    if not isinstance(color, str):
                        color = tuple(int(c) for c in color)
                    image = cv2.line(image, pos1, pos2, color, thickness=self.line_width)

            for kid, kpt in enumerate(kpts):
                if score[kid] < kpt_thr or kpt_color[kid] is None:
                    continue
                color = kpt_color[kid]
                if not isinstance(color, str):
                    color = tuple(int(c) for c in color)
                image = cv2.circle(image, (int(kpt[0]), int(kpt[1])), int(self.radius), color, -1)

        return image


LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def build_visualize_fn(pose_info: Dict):
    """Build a visualize_sample_together function with the given pose_info.

    Returns a callable: (img_cv2, outputs, faces) -> np.ndarray
    """
    from sam_3d_body.visualization.renderer import Renderer

    visualizer = SkeletonVisualizer(line_width=2, radius=5)
    visualizer.set_pose_meta(pose_info)

    def visualize_sample_together(img_cv2, outputs, faces):
        img_keypoints = img_cv2.copy()
        img_mesh = img_cv2.copy()

        all_depths = np.stack([tmp["pred_cam_t"] for tmp in outputs], axis=0)[:, 2]
        outputs_sorted = [outputs[idx] for idx in np.argsort(-all_depths)]

        for person_output in outputs_sorted:
            keypoints_2d = person_output["pred_keypoints_2d"]
            keypoints_2d = np.concatenate(
                [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
            )
            img_keypoints = visualizer.draw_skeleton(img_keypoints, keypoints_2d)

        all_pred_vertices = []
        all_faces = []
        for pid, person_output in enumerate(outputs_sorted):
            all_pred_vertices.append(
                person_output["pred_vertices"] + person_output["pred_cam_t"]
            )
            all_faces.append(faces + len(person_output["pred_vertices"]) * pid)
        all_pred_vertices = np.concatenate(all_pred_vertices, axis=0)
        all_faces = np.concatenate(all_faces, axis=0)

        fake_pred_cam_t = (
            np.max(all_pred_vertices[-2 * 18439:], axis=0)
            + np.min(all_pred_vertices[-2 * 18439:], axis=0)
        ) / 2
        all_pred_vertices = all_pred_vertices - fake_pred_cam_t

        renderer = Renderer(focal_length=person_output["focal_length"], faces=all_faces)
        img_mesh = (
            renderer(
                all_pred_vertices, fake_pred_cam_t, img_mesh,
                mesh_base_color=LIGHT_BLUE, scene_bg_color=(1, 1, 1),
            ) * 255
        )

        white_img = np.ones_like(img_cv2) * 255
        img_mesh_side = (
            renderer(
                all_pred_vertices, fake_pred_cam_t, white_img,
                mesh_base_color=LIGHT_BLUE, scene_bg_color=(1, 1, 1),
                side_view=True,
            ) * 255
        )

        cur_img = np.concatenate([img_cv2, img_keypoints, img_mesh, img_mesh_side], axis=1)
        return cur_img

    return visualize_sample_together

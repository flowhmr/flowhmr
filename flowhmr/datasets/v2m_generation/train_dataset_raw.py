import json
import random
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from ...core.math.geometry import angle_axis_to_rotation_matrix
from .base_dataset import (
    process_r_t,
    compute_wv_rotation,
    compute_camera_features,
    compute_bbox_info,
    padding_or_clip,
)
from ...core.bodymodels.smpl_skeleton import SMPLSkeleton
from .train_dataset import (
    _parse_dirnames_from_root,
    _resolve_file_paths,
    _regroup_samples,
    _load_npz_from_bytes,
    _load_pt_from_bytes,
)


# --- TrainDatasetRaw ---


class TrainDatasetRaw(Dataset):

    FPS = 30

    def __init__(
        self,
        roots: List[dict],
        augmentation_type: str = "original",
        max_len: int = 360,
        round_frames: int = 4,
        split: str = "train",
        load_hand: bool = False,
        cfg_dropout_prob: float = 0.1,
        keyframe_dropout_prob: float = 0.3,
    ):
        super().__init__()
        self.augmentation_type = augmentation_type
        self.max_len = max_len
        self.round_frames = round_frames
        self.split = split
        self.load_hand = load_hand
        self.cfg_dropout_prob = cfg_dropout_prob
        self.keyframe_dropout_prob = keyframe_dropout_prob

        self.smpl_skeleton = SMPLSkeleton()

        self.filenames: List[Dict[str, str]] = []
        for root_config in roots:
            dirnames = _parse_dirnames_from_root(root_config)
            count_before = len(self.filenames)
            for dirname in dirnames:
                item = _resolve_file_paths(dirname, root_config)
                if item is not None:
                    self.filenames.append(item)
            print(
                f"[TrainDatasetRaw] {root_config['root']}: "
                f"{len(dirnames)} dirs -> {len(self.filenames) - count_before} valid samples"
            )
        print(f"[TrainDatasetRaw] {len(self.filenames)} training samples in total")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> dict:
        data = self.filenames[index]
        raw = self._load_raw_data(data)
        return self.process_item(raw, data, index)

    def process_item(self, raw: dict, data: dict, index: int) -> dict:
        total_frames = raw["total_frames"]
        if total_frames > self.max_len:
            raise ValueError(
                f"sequence has {total_frames} frames, exceeding max_len={self.max_len}"
            )

        camera = self._load_camera(raw["camera_data"])
        motion = self._load_motion(raw["motion_data"])

        root_rotation, transl, relative_transform, transl0 = self._transform_motion_to_wv(
            motion, camera["RT"]
        )
        camera_RT_wv = self._transform_camera_to_wv(camera["RT"], relative_transform, transl0)
        R_to_first_frame, center_velocity = compute_camera_features(camera_RT_wv)

        inputs_dict, feature_length = self._build_inputs(
            raw["full_feature"],
            R_to_first_frame, center_velocity, raw["bbox_data"], camera["K"],
            bbox_start=raw["bbox_start"],
            bbox_end=raw["bbox_end"],
        )

        target = self._build_target(motion, root_rotation, transl)

        inputs_dict["feature"] = padding_or_clip(
            inputs_dict["feature"],
            self.max_len,
            round_frames=1,
            keys=["feature", "camera_R", "camera_T", "bbox_info"],
        )

        target = padding_or_clip(
            target,
            self.max_len,
            round_frames=1,
            keys=["root_rotation", "body_rotations", "trans", "shapes"],
        )

        ret = {
            "target": target,
            "inputs": inputs_dict,
            "index": index,
            "length": feature_length,
            "meta": {
                "sequence_name": data["sequence_name"],
                "feature_name": data["feature_name"],
                "motion_name": data["motion_name"],
                "camera_name": data["camera_name"],
                "camera_origin_K": camera["K"],
                "camera_origin_RT": camera["RT"],
                "camera_wv_RT": camera_RT_wv,
            },
        }
        ret["meta"] = padding_or_clip(
            ret["meta"],
            self.max_len,
            round_frames=1,
            keys=["camera_origin_K", "camera_origin_RT", "camera_wv_RT"],
        )
        return ret


    def _random_crop(self, motion_data, camera_data, bbox_data, full_feature, total_frames, bbox_start_end):
        if self.split == "train" and self.max_len < total_frames:
            start_frame = random.randint(0, total_frames - self.max_len)
            end_frame = start_frame + self.max_len
            cropped_len = end_frame - start_frame

            motion_data = {
                "poses": motion_data["poses"][start_frame:end_frame],
                "trans": motion_data["trans"][start_frame:end_frame],
                "betas": motion_data["betas"],
            }
            camera_data = {
                "RT": camera_data["RT"][start_frame:end_frame],
                "K": camera_data["K"][start_frame:end_frame],
            }

            new_start = bbox_start_end[0] - start_frame
            new_end = bbox_start_end[1] - start_frame
            if new_start < 0:
                new_start = 0
            if new_end + 1 > cropped_len:
                new_end = cropped_len - 1
            bbox_start_end = [new_start, new_end]

            bbox_data = {
                "bbox": bbox_data["bbox"][start_frame:end_frame],
                "start_end": bbox_start_end,
            }
            full_feature = full_feature[start_frame:end_frame]
            total_frames = cropped_len

        return motion_data, camera_data, bbox_data, full_feature, total_frames, bbox_start_end

    def _load_raw_data(self, data: dict) -> dict:
        motion_data = dict(np.load(data["motion_name"]))
        camera_data = dict(np.load(data["camera_name"]))
        feature_data = torch.load(data["feature_name"], map_location="cpu", weights_only=True)
        bbox_data = dict(np.load(data["bbox_name"]))

        n_frames_motion = motion_data["poses"].shape[0]
        n_frames_camera = camera_data["RT"].shape[0]
        n_frames_bbox = bbox_data["bbox"].shape[0]
        assert n_frames_camera == n_frames_motion == n_frames_bbox, (
            f"frame count mismatch: camera={n_frames_camera}, motion={n_frames_motion}, "
            f"bbox={n_frames_bbox}, file={data['feature_name']}"
        )
        total_frames = n_frames_camera

        full_feature = torch.zeros(total_frames, *feature_data.shape[1:])
        bbox_start_end = bbox_data["start_end"]
        full_feature[bbox_start_end[0]: bbox_start_end[1] + 1] = feature_data

        motion_data, camera_data, bbox_data, full_feature, total_frames, bbox_start_end = \
            self._random_crop(motion_data, camera_data, bbox_data, full_feature, total_frames, bbox_start_end)

        return {
            "motion_data": motion_data,
            "camera_data": camera_data,
            "bbox_data": bbox_data,
            "full_feature": full_feature,
            "total_frames": total_frames,
            "bbox_start": int(bbox_start_end[0]),
            "bbox_end": int(bbox_start_end[1]) + 1, # exclusive end index
        }

    def _load_camera(self, camera_data: dict) -> dict:
        RT = torch.FloatTensor(camera_data["RT"])

        if RT.shape[1] == 3:
            n = RT.shape[0]
            RT_homo = torch.zeros((n, 4, 4), dtype=RT.dtype)
            RT_homo[:, :3, :4] = RT
            RT_homo[:, 3, 3] = 1
            RT = RT_homo

        return {
            "RT": RT,
            "K": torch.FloatTensor(camera_data["K"]),
        }

    def _load_motion(self, motion_data: dict) -> dict:
        poses = torch.FloatTensor(motion_data["poses"])
        transl = torch.FloatTensor(motion_data["trans"])
        betas = torch.FloatTensor(motion_data["betas"])

        if len(poses.shape) == 2:
            poses = poses.reshape(poses.shape[0], -1, 3)
        rotations = angle_axis_to_rotation_matrix(poses)

        root_rotations = rotations[:, 0]
        body_rotations = rotations[:, 1:]

        n_frames = poses.shape[0]
        shapes = betas[0][None].repeat(n_frames, 1)

        return {
            "root_rotations": root_rotations,
            "body_rotations": body_rotations,
            "transl": transl,
            "shapes": shapes,
        }


    def _transform_motion_to_wv(self, motion: dict, camera_RT: torch.Tensor):
        camera_R0 = camera_RT[0, :3, :3]
        relative_transform = compute_wv_rotation(camera_R0)

        j_shaped = self.smpl_skeleton.compute_j_shaped(motion["shapes"][:1])

        root_rotation, transl = process_r_t(
            relative_transform, motion["root_rotations"], motion["transl"], j_shaped[:, 0]
        )

        transl0 = transl[:1].clone()
        transl[:, 0] -= transl0[:, 0]
        transl[:, 2] -= transl0[:, 2]

        return root_rotation, transl, relative_transform, transl0

    @staticmethod
    def _transform_camera_to_wv(
        camera_RT: torch.Tensor, relative_transform: torch.Tensor, transl0: torch.Tensor
    ) -> torch.Tensor:
        camera_RT = camera_RT.clone()
        trans = torch.eye(4)
        trans[:3, :3] = relative_transform
        trans[:3, 3] = -transl0.reshape(3)
        camera_RT_wv = camera_RT @ torch.inverse(trans)
        return camera_RT_wv


    def _build_inputs(
        self,
        full_feature: torch.Tensor,
        R_to_first_frame: torch.Tensor,
        center_velocity: torch.Tensor,
        bbox_data: dict,
        camera_K: torch.Tensor,
        bbox_start: int = 0,
        bbox_end: int = -1,
    ):
        feature = full_feature
        if not self.load_hand:
            feature = feature[:, :1024]
        feature_length = feature.shape[0]
        feature[~torch.isfinite(feature)] = 0

        if bbox_end < 0:
            bbox_end = feature_length

        if self.split == "train":
            feature = self._augment_feature(feature, bbox_start, bbox_end)

        bbox = bbox_data["bbox"][..., :4]
        bbox = torch.FloatTensor(bbox)
        bbox_center = (bbox[:, :2] + bbox[:, 2:4]) / 2
        bbox_scale = (bbox[:, 2:4] - bbox[:, :2]).max(dim=-1, keepdim=True).values
        bbox_info = compute_bbox_info(bbox_center, bbox_scale, camera_K)

        inputs_dict = {
            "feature": {
                "feature": feature,
                "camera_R": R_to_first_frame.reshape(R_to_first_frame.shape[0], -1),
                "camera_T": center_velocity * self.FPS,
                "bbox_info": bbox_info,
            },
        }

        return inputs_dict, feature_length


    def _build_target(
        self,
        motion: dict,
        root_rotation: torch.Tensor,
        transl: torch.Tensor,
    ) -> dict:
        return {
            "root_rotation": root_rotation,         # (T, 3, 3)
            "body_rotations": motion["body_rotations"],  # (T, 51, 3, 3)
            "trans": transl,                         # (T, 3)
            "shapes": motion["shapes"],              # (T, 16)
        }


    def _apply_keyframe_mask(
        self, feature: torch.Tensor, bbox_start: int, bbox_end: int
    ) -> torch.Tensor:
        n_valid = bbox_end - bbox_start
        if n_valid < 2:
            return feature

        feature = feature.clone()

        inner_count = n_valid - 2
        boundary = max(1, n_valid // 30)

        if inner_count <= 0:
            keyframe_indices = {bbox_start, bbox_end - 1}
        else:
            if random.random() < 0.7:
                k = random.randint(0, boundary)
            else:
                k = random.randint(0, inner_count)

            if k == 0:
                mid_offsets = []
            else:
                k = min(k, inner_count)
                mid_offsets = random.sample(range(1, n_valid - 1), k)

            keyframe_indices = {bbox_start, bbox_end - 1}
            for off in mid_offsets:
                keyframe_indices.add(bbox_start + off)

        mask = torch.zeros(feature.shape[0], dtype=torch.bool)
        for idx in keyframe_indices:
            if 0 <= idx < feature.shape[0]:
                mask[idx] = True
        feature[~mask] = 0.0
        return feature

    def _augment_feature(
        self, feature: torch.Tensor, bbox_start: int = 0, bbox_end: int = -1
    ) -> torch.Tensor:
        if self.augmentation_type == "original":
            if random.random() < self.cfg_dropout_prob:
                feature = torch.zeros_like(feature)
        elif self.augmentation_type == "keyframe":
            r = random.random()
            if r < self.cfg_dropout_prob:
                feature = torch.zeros_like(feature)
            elif r < self.cfg_dropout_prob + self.keyframe_dropout_prob:
                feature = self._apply_keyframe_mask(feature, bbox_start, bbox_end)
        return feature


    @staticmethod
    def _log_data_shapes(data: dict, prefix: str = ""):
        for key, val in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(val, torch.Tensor):
                print(f"  {full_key}: shape={list(val.shape)}, dtype={val.dtype}")
            elif isinstance(val, dict):
                TrainDatasetRaw._log_data_shapes(val, prefix=full_key)
            elif isinstance(val, (int, float, bool, str)):
                print(f"  {full_key}: {val}")
            else:
                print(f"  {full_key}: {type(val).__name__}")


class WDSTrainDatasetRaw(TrainDatasetRaw):

    is_webdataset = True

    def __init__(
        self,
        tar_urls: Union[str, List[str]],
        augmentation_type: str = "original",
        max_len: int = 360,
        round_frames: int = 4,
        split: str = "train",
        load_hand: bool = False,
        cfg_dropout_prob: float = 0.1,
        keyframe_dropout_prob: float = 0.3,
        shuffle_buffer: int = 1000,
        resampled: bool = True,
        split_by_node: bool = False,
        quality_filter: bool = False,
        max_motion_accel_mps2: float = 150.0,
        max_camera_accel_mps2: float = 150.0,
        max_root_bbox_offset: float = 1.0,
    ):
        self.augmentation_type = augmentation_type
        self.max_len = max_len
        self.round_frames = round_frames
        self.split = split
        self.load_hand = load_hand
        self.cfg_dropout_prob = cfg_dropout_prob
        self.keyframe_dropout_prob = keyframe_dropout_prob
        self.quality_filter = quality_filter
        self.max_motion_accel_mps2 = max_motion_accel_mps2
        self.max_camera_accel_mps2 = max_camera_accel_mps2
        self.max_root_bbox_offset = max_root_bbox_offset
        self.smpl_skeleton = SMPLSkeleton()

        Dataset.__init__(self)

        import webdataset as wds

        if isinstance(tar_urls, str):
            tar_urls = [tar_urls]

        expanded_urls = []
        for url in tar_urls:
            expanded_urls.extend(wds.shardlists.expand_urls(url))

        print(f"[WDSTrainDatasetRaw] {len(expanded_urls)} shards")

        nodesplitter = wds.split_by_node if split_by_node else (lambda x: x)

        self.dataset = (
            wds.WebDataset(
                expanded_urls,
                resampled=resampled,
                shardshuffle=False,
                nodesplitter=nodesplitter,
                workersplitter=wds.split_by_worker,
                verbose=True,
            )
            .compose(_regroup_samples)
            .shuffle(shuffle_buffer)
            .map(self._decode_sample)
            .select(self._is_valid_sample)
            .map(self._wds_process_sample)
        )

        self._tar_urls = expanded_urls
        self._estimated_length = None


    def __len__(self):
        if self._estimated_length is None:
            self._estimated_length = len(self._tar_urls) * 1024
        return self._estimated_length

    def __iter__(self):
        return iter(self.dataset)

    def __getitem__(self, index):
        raise NotImplementedError(
            "WDSTrainDatasetRaw does not support random access; "
            "iterate with iter(dataset) or use get_dataloader()"
        )

    def set_epoch_length(self, length: int):
        self._estimated_length = length


    @staticmethod
    def _decode_sample(sample: dict) -> dict:
        decoded = {"__key__": sample["__key__"]}

        for prefix_key, no_prefix_key, loader, field_name in [
            (".motion.npz", "motion.npz", _load_npz_from_bytes, "motion_data"),
            (".camera.npz", "camera.npz", _load_npz_from_bytes, "camera_data"),
            (".feature.pt", "feature.pt", _load_pt_from_bytes, "feature_data"),
            (".bbox.npz", "bbox.npz", _load_npz_from_bytes, "bbox_data"),
        ]:
            key = prefix_key if prefix_key in sample else no_prefix_key
            if key in sample:
                decoded[field_name] = loader(sample[key])

        meta_key = ".metadata.json" if ".metadata.json" in sample else "metadata.json"
        if meta_key in sample:
            decoded["metadata"] = json.loads(sample[meta_key].decode("utf-8"))

        return decoded

    def _quality_flags(self, sample: dict) -> List[str]:
        """Return consistency failures under the unified 30-FPS, meter, W2C convention."""
        motion = sample["motion_data"]
        camera = sample["camera_data"]
        bbox_data = sample["bbox_data"]
        flags = []

        try:
            poses = np.asarray(motion["poses"])
            trans = np.asarray(motion["trans"], dtype=np.float64)
            betas = np.asarray(motion["betas"])
            camera_rt = np.asarray(camera["RT"], dtype=np.float64)
            camera_k = np.asarray(camera["K"], dtype=np.float64)
            bbox = np.asarray(bbox_data["bbox"], dtype=np.float64)[..., :4]
            start_end = np.asarray(bbox_data["start_end"]).reshape(-1)
        except (KeyError, TypeError, ValueError):
            return ["missing_or_invalid_fields"]

        frame_count = poses.shape[0] if poses.ndim else 0
        lengths = [
            frame_count,
            trans.shape[0] if trans.ndim else 0,
            camera_rt.shape[0] if camera_rt.ndim else 0,
            camera_k.shape[0] if camera_k.ndim else 0,
            bbox.shape[0] if bbox.ndim else 0,
        ]
        pose_shape_valid = (
            poses.ndim == 3 and poses.shape[1:] == (52, 3)
        ) or (
            poses.ndim == 2 and poses.shape[1] == 156
        )
        if frame_count < 10:
            flags.append("too_short")
        if len(set(lengths)) != 1:
            flags.append("length_mismatch")
        if not pose_shape_valid or trans.shape != (frame_count, 3):
            flags.append("invalid_motion_shape")
        if betas.ndim < 2 or betas.shape[-1] != 16:
            flags.append("invalid_shape_params")
        if camera_rt.shape[1:] not in ((3, 4), (4, 4)):
            flags.append("invalid_camera_shape")
        if camera_k.shape != (frame_count, 3, 3) or bbox.shape != (frame_count, 4):
            flags.append("invalid_camera_bbox_shape")
        if (
            start_end.size != 2
            or start_end[0] < 0
            or start_end[1] < start_end[0]
            or start_end[1] >= frame_count
        ):
            flags.append("invalid_feature_range")
        else:
            feature = sample["feature_data"]
            if isinstance(feature, dict):
                feature = feature.get("feature", next(iter(feature.values())))
            if len(feature) != int(start_end[1] - start_end[0] + 1):
                flags.append("feature_length_mismatch")

        arrays = (poses, trans, betas, camera_rt, camera_k, bbox)
        if not all(np.isfinite(value).all() for value in arrays):
            flags.append("nonfinite")
        if flags or not self.quality_filter:
            return flags

        declared_fps = float(np.asarray(motion.get("mocap_framerate", self.FPS)).reshape(-1)[0])
        if abs(declared_fps - self.FPS) > 1e-3:
            flags.append("fps_mismatch")

        rotation = camera_rt[:, :3, :3]
        translation = camera_rt[:, :3, 3]
        rotation_error = np.linalg.norm(
            rotation @ np.swapaxes(rotation, -1, -2) - np.eye(3), axis=(-2, -1)
        ).max()
        determinant_error = np.abs(np.linalg.det(rotation) - 1.0).max()
        if rotation_error > 1e-2 or determinant_error > 1e-2:
            flags.append("invalid_camera_rotation")

        if frame_count >= 3:
            motion_acceleration = np.linalg.norm(
                np.diff(trans, n=2, axis=0), axis=-1
            ) * self.FPS * self.FPS
            if np.quantile(motion_acceleration, 0.99) > self.max_motion_accel_mps2:
                flags.append("motion_jump")

            camera_center = -np.einsum("tji,tj->ti", rotation, translation)
            camera_acceleration = np.linalg.norm(
                np.diff(camera_center, n=2, axis=0), axis=-1
            ) * self.FPS * self.FPS
            if np.quantile(camera_acceleration, 0.99) > self.max_camera_accel_mps2:
                flags.append("camera_jump")

        width = bbox[:, 2] - bbox[:, 0]
        height = bbox[:, 3] - bbox[:, 1]
        bbox_size = np.maximum(width, height)
        focal = np.concatenate([camera_k[:, 0, 0], camera_k[:, 1, 1]])
        if (width <= 1.0).any() or (height <= 1.0).any() or (focal <= 0).any():
            flags.append("invalid_bbox_or_focal")
        else:
            camera_root = np.einsum("tij,tj->ti", rotation, trans) + translation
            depth = camera_root[:, 2]
            if (depth > 1e-4).mean() < 0.99:
                flags.append("root_behind_camera")
            else:
                projected = np.einsum("tij,tj->ti", camera_k, camera_root)
                root_uv = projected[:, :2] / depth[:, None]
                bbox_center = (bbox[:, :2] + bbox[:, 2:4]) * 0.5
                normalized_offset = np.linalg.norm(
                    root_uv - bbox_center, axis=-1
                ) / bbox_size
                if np.quantile(normalized_offset, 0.95) > self.max_root_bbox_offset:
                    flags.append("root_bbox_mismatch")
        return flags

    def _is_valid_sample(self, sample: dict) -> bool:
        required = ["motion_data", "camera_data", "feature_data", "bbox_data"]
        for key in required:
            if key not in sample:
                print(
                    f"[WDSTrainDatasetRaw] Warning: {sample.get('__key__', 'unknown')} "
                    f"is missing '{key}', skipped"
                )
                return False
        flags = self._quality_flags(sample)
        if flags:
            print(
                f"[WDSTrainDatasetRaw] Warning: {sample.get('__key__', 'unknown')} "
                f"failed quality checks {flags}, skipped"
            )
            return False
        return True


    def _wds_process_sample(self, sample: dict) -> dict:
        motion_data = sample["motion_data"]
        camera_data = sample["camera_data"]
        feature_data = sample["feature_data"]
        bbox_data = sample["bbox_data"]
        metadata = sample.get("metadata", {})

        n_frames_motion = motion_data["poses"].shape[0]
        n_frames_camera = camera_data["RT"].shape[0]
        n_frames_bbox = bbox_data["bbox"].shape[0]
        assert n_frames_camera == n_frames_motion == n_frames_bbox, (
            f"frame count mismatch: camera={n_frames_camera}, motion={n_frames_motion}, "
            f"bbox={n_frames_bbox}, key={sample['__key__']}"
        )
        total_frames = n_frames_camera

        if isinstance(feature_data, dict):
            feature_data = feature_data.get("feature", list(feature_data.values())[0])
        if isinstance(feature_data, np.ndarray):
            feature_data = torch.from_numpy(feature_data).float()

        full_feature = torch.zeros(total_frames, *feature_data.shape[1:])
        bbox_start_end = bbox_data["start_end"]
        start_idx = int(bbox_start_end[0])
        end_idx = int(bbox_start_end[1])
        full_feature[start_idx: end_idx + 1] = feature_data

        motion_data, camera_data, bbox_data, full_feature, total_frames, bbox_start_end = \
            self._random_crop(motion_data, camera_data, bbox_data, full_feature, total_frames,
                              [start_idx, end_idx])

        raw = {
            "motion_data": motion_data,
            "camera_data": camera_data,
            "bbox_data": bbox_data,
            "full_feature": full_feature,
            "total_frames": total_frames,
            "bbox_start": int(bbox_start_end[0]),
            "bbox_end": int(bbox_start_end[1]) + 1,
        }

        raw_key = sample["__key__"]
        sequence_name = metadata.get("sequence_name", raw_key.replace("_dot_", "."))

        data = {
            "sequence_name": sequence_name,
            "feature_name": metadata.get("feature_path", raw_key),
            "motion_name": metadata.get("motion_path", raw_key),
            "camera_name": metadata.get("camera_path", raw_key),
        }

        return self.process_item(raw, data, index=raw_key)

    # --- DataLoader ---

    def get_dataloader(
        self,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        steps_per_epoch: Optional[int] = None,
    ):
        import webdataset as wds

        loader = wds.WebLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        if steps_per_epoch is not None:
            loader = loader.with_epoch(steps_per_epoch)

        return loader


_V2M_TARGET_KEYS = [
    "smooth_root_pos", "smooth_root_vel", "global_root_heading",
    "local_joints_positions", "global_rot_data", "local_rot_data",
    "foot_contacts", "shapes",
]


class V2MTrainDatasetRaw(TrainDatasetRaw):

    def _build_target(
        self,
        motion: dict,
        root_rotation: torch.Tensor,
        transl: torch.Tensor,
    ) -> dict:
        from ...core.motion.motion_rep import encode_motion_v0_rotmat

        target_dict = {
            "root_rotation": root_rotation,
            "body_rotations": motion["body_rotations"],
            "trans": transl,
            "shapes": motion["shapes"],
        }

        with torch.no_grad():
            motion = encode_motion_v0_rotmat(target_dict, self.smpl_skeleton, fps=30.0)

        return {key: motion[key] for key in _V2M_TARGET_KEYS}

    def process_item(self, raw: dict, data: dict, index) -> dict:
        total_frames = raw["total_frames"]
        if total_frames > self.max_len:
            raise ValueError(
                f"sequence has {total_frames} frames, exceeding max_len={self.max_len}"
            )

        camera = self._load_camera(raw["camera_data"])
        motion = self._load_motion(raw["motion_data"])

        root_rotation, transl, relative_transform, transl0 = self._transform_motion_to_wv(
            motion, camera["RT"]
        )
        camera_RT_wv = self._transform_camera_to_wv(camera["RT"], relative_transform, transl0)
        R_to_first_frame, center_velocity = compute_camera_features(camera_RT_wv)

        inputs_dict, feature_length = self._build_inputs(
            raw["full_feature"],
            R_to_first_frame, center_velocity, raw["bbox_data"], camera["K"],
            bbox_start=raw["bbox_start"],
            bbox_end=raw["bbox_end"],
        )

        target = self._build_target(motion, root_rotation, transl)

        # pad or clip to max_len
        inputs_dict["feature"] = padding_or_clip(
            inputs_dict["feature"],
            self.max_len,
            round_frames=1,
            keys=["feature", "camera_R", "camera_T", "bbox_info"],
        )

        target = padding_or_clip(
            target,
            self.max_len,
            round_frames=1,
            keys=_V2M_TARGET_KEYS,
        )

        ret = {
            "target": target,
            "inputs": inputs_dict,
            "index": index,
            "length": feature_length,
            "meta": {
                "sequence_name": data["sequence_name"],
                "feature_name": data["feature_name"],
                "motion_name": data["motion_name"],
                "camera_name": data["camera_name"],
                "camera_origin_K": camera["K"],
                "camera_origin_RT": camera["RT"],
                "camera_wv_RT": camera_RT_wv,
            },
        }
        ret["meta"] = padding_or_clip(
            ret["meta"],
            self.max_len,
            round_frames=1,
            keys=["camera_origin_K", "camera_origin_RT", "camera_wv_RT"],
        )
        return ret


class WDSV2MTrainDatasetRaw(WDSTrainDatasetRaw, V2MTrainDatasetRaw):
    pass

import io
import json
import os
import random
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from ...core.math.geometry import angle_axis_to_rotation_matrix, rotation_matrix_to_rot6d
from .base_dataset import (
    process_r_t,
    get_local_transl_vel,
    compute_wv_rotation,
    compute_camera_features,
    compute_bbox_info,
    padding_or_clip,
)
from ...core.bodymodels.smpl_skeleton import SMPLSkeleton


def _parse_dirnames_from_root(root_config: dict) -> List[str]:
    root = root_config["root"]

    if isinstance(root, list):
        dirnames = list(root)
    elif os.path.isdir(root):
        dirnames = []
        for folder in root_config.get("folders", []):
            folder_path = os.path.join(root, folder)
            dirnames.extend([os.path.join(folder_path, x) for x in sorted(os.listdir(folder_path))])
    elif os.path.isfile(root) and root.endswith(".txt"):
        with open(root, "r", encoding="utf-8") as f:
            dirnames = [line.strip() for line in f.readlines() if line.strip()]
    else:
        raise ValueError(f"invalid data root (expected a directory or .txt list): {root}")

    sample_step = root_config.get("sample_step", 1)
    if sample_step > 1:
        dirnames = [d for i, d in enumerate(dirnames) if i % sample_step == 0]
    elif sample_step < 0:
        step = -sample_step
        dirnames = [d for i, d in enumerate(dirnames) if i % step != 0]

    return dirnames


def _resolve_file_paths(dirname: str, root_config: dict) -> Optional[Dict[str, str]]:
    name = dirname.split("/")[-1]
    basename = "_".join(name.split("_")[:-1])
    last_dirname = os.path.basename(os.path.dirname(dirname))

    motion_name = os.path.join(dirname, basename + ".npz")

    camera_format = root_config.get("camera_format", "{name}_camera.npz")
    camera_name = os.path.join(dirname, camera_format.format(name=name, last_dirname=last_dirname))

    feature_format = root_config.get("feature_format", "{name}_sam3d_feat.pt")
    feature_name = os.path.join(dirname, feature_format.format(name=name, last_dirname=last_dirname))

    bbox_format = root_config.get("bbox_format", None)
    if bbox_format is None:
        return None
    bbox_name = os.path.join(dirname, bbox_format.format(name=name, last_dirname=last_dirname))

    return {
        "sequence_name": name,
        "motion_name": motion_name,
        "camera_name": camera_name,
        "feature_name": feature_name,
        "bbox_name": bbox_name,
    }


# --- TrainDataset ---


class TrainDataset(Dataset):

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
    ):
        super().__init__()
        self.augmentation_type = augmentation_type
        self.max_len = max_len
        self.round_frames = round_frames
        self.split = split
        self.load_hand = load_hand
        self.cfg_dropout_prob = cfg_dropout_prob

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
                f"[TrainDataset] {root_config['root']}: "
                f"{len(dirnames)} dirs -> {len(self.filenames) - count_before} valid samples"
            )
        print(f"[TrainDataset] {len(self.filenames)} training samples in total")

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
            keys=["root_rot6d", "body_rot6d", "transl_vel", "trans", "shapes",
                  "end_effector_vel", "end_effector_horizontal_speed", "end_effector_vertical_vel"],
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
            "bbox_end": int(bbox_start_end[1])+1, # exclusive end index
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

        transl_vel = get_local_transl_vel(transl, root_rotations, fps=self.FPS)

        n_frames = poses.shape[0]
        shapes = betas[0][None].repeat(n_frames, 1)

        return {
            "root_rotations": root_rotations,
            "body_rotations": body_rotations,
            "transl": transl,
            "transl_vel": transl_vel,
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
        transl = transl - transl0

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
    ):
        feature = full_feature
        if not self.load_hand:
            feature = feature[:, :1024]
        feature_length = feature.shape[0]
        feature[feature.isnan()] = 0

        if self.split == "train":
            feature = self._augment_feature(feature)

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
        target = {
            "root_rot6d": rotation_matrix_to_rot6d(root_rotation),
            "body_rot6d": rotation_matrix_to_rot6d(motion["body_rotations"]),
            "transl_vel": motion["transl_vel"],
            "trans": transl,
            "shapes": motion["shapes"],
        }
        vel, h_speed, v_vel = self._compute_end_effector_vel(target)
        target["end_effector_vel"] = vel
        target["end_effector_horizontal_speed"] = h_speed
        target["end_effector_vertical_vel"] = v_vel

        return target

    def _compute_end_effector_vel(self, target: dict, fps: float = 30.0):
        joint_ids = [7, 10, 8, 11, 20, 21]

        smpl_params = {
            "rot6d": torch.cat([target["root_rot6d"][:, None], target["body_rot6d"]], dim=1),
            "trans": target["trans"],
            "shapes": target["shapes"].mean(dim=0, keepdim=True),
        }

        with torch.no_grad():
            joints = self.smpl_skeleton(smpl_params)["keypoints3d"]

        end_joints = joints[:, joint_ids, :]

        end_vel = end_joints[1:] - end_joints[:-1]
        end_vel = torch.cat([end_vel, end_vel[-1:]], dim=0)
        end_vel = end_vel * fps

        horizontal_speed = torch.sqrt(end_vel[..., 0] ** 2 + end_vel[..., 2] ** 2)
        vertical_velocity = end_vel[..., 1]

        return end_vel, horizontal_speed, vertical_velocity


    def _augment_feature(self, feature: torch.Tensor) -> torch.Tensor:
        if self.augmentation_type == "original":
            if random.random() < self.cfg_dropout_prob:
                feature = torch.zeros_like(feature)
            return feature

        return feature


    @staticmethod
    def _log_data_shapes(data: dict, prefix: str = ""):
        for key, val in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(val, torch.Tensor):
                print(f"  {full_key}: shape={list(val.shape)}, dtype={val.dtype}")
            elif isinstance(val, dict):
                TrainDataset._log_data_shapes(val, prefix=full_key)
            elif isinstance(val, (int, float, bool, str)):
                print(f"  {full_key}: {val}")
            else:
                print(f"  {full_key}: {type(val).__name__}")


_FILE_SUFFIXES = [".motion.npz", ".camera.npz", ".feature.pt", ".bbox.npz", ".metadata.json"]


def _load_npz_from_bytes(data: bytes) -> dict:
    buf = io.BytesIO(data)
    return dict(np.load(buf, allow_pickle=True))


def _load_pt_from_bytes(data: bytes) -> torch.Tensor:
    buf = io.BytesIO(data)
    return torch.load(buf, map_location="cpu", weights_only=True)


def _regroup_samples(src):
    for sample in src:
        if "__key__" not in sample:
            yield sample
            continue

        fixed_sample = {}
        original_key = sample.get("__key__", "")

        for k, v in sample.items():
            if k in ("__key__", "__url__"):
                fixed_sample[k] = v
                continue

            found_suffix = None
            for suffix in _FILE_SUFFIXES:
                clean_suffix = suffix.lstrip(".")
                if k.endswith(clean_suffix):
                    found_suffix = suffix
                    extra_key_part = k[: -len(clean_suffix)].rstrip(".")
                    if extra_key_part and not original_key.endswith(extra_key_part):
                        original_key = original_key + "." + extra_key_part
                    break

            if found_suffix:
                fixed_sample[found_suffix] = v
            else:
                fixed_sample[k] = v

        fixed_sample["__key__"] = original_key
        yield fixed_sample


class WDSTrainDataset(TrainDataset):

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
        shuffle_buffer: int = 1000,
        resampled: bool = True,
        split_by_node: bool = False,
    ):
        self.augmentation_type = augmentation_type
        self.max_len = max_len
        self.round_frames = round_frames
        self.split = split
        self.load_hand = load_hand
        self.cfg_dropout_prob = cfg_dropout_prob
        self.smpl_skeleton = SMPLSkeleton()

        Dataset.__init__(self)

        import webdataset as wds

        if isinstance(tar_urls, str):
            tar_urls = [tar_urls]

        expanded_urls = []
        for url in tar_urls:
            expanded_urls.extend(wds.shardlists.expand_urls(url))

        print(f"[WDSTrainDataset] {len(expanded_urls)} shards")

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
            "WDSTrainDataset does not support random access; "
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

    @staticmethod
    def _is_valid_sample(sample: dict) -> bool:
        required = ["motion_data", "camera_data", "feature_data", "bbox_data"]
        for key in required:
            if key not in sample:
                print(
                    f"[WDSTrainDataset] Warning: {sample.get('__key__', 'unknown')} "
                    f"is missing '{key}', skipped"
                )
                return False
        if sample["motion_data"]["poses"].shape[0] < 10:
            print(
                f"[WDSTrainDataset] Warning: {sample.get('__key__', 'unknown')} "
                f"motion shorter than 10 frames, skipped"
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
            "bbox_end": int(bbox_start_end[1]) + 1, # exclusive end index
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

from typing import List

import torch
from torch.utils.data import Dataset

from ...core.math.geometry import rotation_matrix_to_rot6d
from .train_dataset_raw import TrainDatasetRaw


class V2MGRPOGTDataset(TrainDatasetRaw):

    def __init__(
        self,
        roots: list,
        max_len: int = 360,
        load_hand: bool = True,
        sample_step: int = 1,
        round_frames: int = 1,
    ):
        super().__init__(
            roots=roots,
            augmentation_type="original",
            max_len=max_len,
            round_frames=round_frames,
            split="val",
            load_hand=load_hand,
            cfg_dropout_prob=0.0,
        )
        if sample_step > 1:
            self.filenames = self.filenames[::sample_step]
        self._virtual_len = None

    def set_train_iterations(self, virtual_len: int) -> None:
        self._virtual_len = int(virtual_len)

    def __len__(self) -> int:
        return self._virtual_len if self._virtual_len is not None else len(self.filenames)

    def _random_crop(self, motion_data, camera_data, bbox_data, full_feature,
                     total_frames, bbox_start_end):
        if self.max_len is not None and self.max_len < total_frames:
            start_frame, end_frame = 0, self.max_len
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

            new_start = max(bbox_start_end[0] - start_frame, 0)
            new_end = min(bbox_start_end[1] - start_frame, cropped_len - 1)
            bbox_start_end = [new_start, new_end]

            bbox_data = {
                "bbox": bbox_data["bbox"][start_frame:end_frame],
                "start_end": bbox_start_end,
            }
            full_feature = full_feature[start_frame:end_frame]
            total_frames = cropped_len

        return motion_data, camera_data, bbox_data, full_feature, total_frames, bbox_start_end

    def process_item(self, raw: dict, data: dict, index: int) -> dict:
        ret = super().process_item(raw, data, index)
        t = ret["target"]
        L = int(ret["length"])
        root_r = t["root_rotation"][:L]           # (T,3,3) WV
        body_r = t["body_rotations"][:L]          # (T,51,3,3) WV
        rotmats = torch.cat([root_r[:, None], body_r], dim=1)
        T = rotmats.shape[0]
        rot6d = rotation_matrix_to_rot6d(
            rotmats.reshape(T * 52, 3, 3)).reshape(T, 52, 6)
        with torch.no_grad():
            out = self.smpl_skeleton({
                "rot6d": rot6d,
                "shapes": t["shapes"][:L].reshape(T, -1),
                "trans": t["trans"][:L].reshape(T, 3),
            })
        gt_joints = out["keypoints3d"].reshape(T, -1, 3)[:, :52, :].float()
        return {
            "inputs": ret["inputs"],
            "reward_gt": {"gt_joints": gt_joints},
            "index": index,
            "length": L,
            "meta": ret["meta"],
        }

    def __getitem__(self, index: int) -> dict:
        real_index = index % len(self.filenames)
        data = self.filenames[real_index]
        raw = self._load_raw_data(data)
        return self.process_item(raw, data, real_index)


def grpo_gt_collate(batch: List[dict]) -> dict:
    feat_keys = batch[0]["inputs"]["feature"].keys()

    gts = [b["reward_gt"]["gt_joints"] for b in batch]
    T_max = max(g.shape[0] for g in gts)
    if any(g.shape[0] != T_max for g in gts):
        gts = [g if g.shape[0] == T_max else
               torch.cat([g, g.new_zeros((T_max - g.shape[0],) + g.shape[1:])], dim=0)
               for g in gts]

    return {
        "inputs": {"feature": {
            k: torch.stack([b["inputs"]["feature"][k] for b in batch], dim=0)
            for k in feat_keys
        }},
        "reward_gt": {
            "gt_joints": torch.stack(gts, dim=0),
        },
        "index": torch.tensor([b["index"] for b in batch], dtype=torch.long),
        "length": torch.tensor([b["length"] for b in batch], dtype=torch.long),
        "meta": {"sequence_name": [b["meta"]["sequence_name"] for b in batch]},
    }

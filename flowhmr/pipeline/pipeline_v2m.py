import json
import logging
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor

from .pipeline_v2m_simple import SimpleV2MPipeline
from ..core.motion.motion_rep import (
    _global_rots_to_local_rots,
    _run_fk,
    _SMPLH_PARENTS,
)
from ..core.math.geometry import rot6d_to_rotation_matrix, rotation_matrix_to_rot6d
from .utils import length_to_mask, randn_tensor
from .ode_solvers import odeint_custom

logger = logging.getLogger(__name__)


class V2MPipeline(SimpleV2MPipeline):

    COMPONENT_DIMS = {
        "smooth_root_vel_xz": 2,
        "smooth_root_pos_y": 1,
        "local_joints_positions": 52 * 3,   # 156
        "rotation_data": 52 * 6,             # 312
        "shapes": 16,
        "foot_contacts": 4,
    }
    LATENT_DIM = sum(COMPONENT_DIMS.values())

    DEFAULT_LOSS_WEIGHTS = {
        "smooth_root_vel_xz": 10.0,
        "smooth_root_pos_y": 10.0,
        "local_joints_positions": 10.0,
        "rotation_data": 10.0,
        "shapes": 1.0,
        "foot_contacts": 4.0,
        "fk_rotation": 5.0,
        "fk_consistency": 5.0,
    }

    def __init__(
        self,
        network_module: str,
        network_module_args: dict,
        losses_cfg: dict,
        noise_scheduler_cfg: dict,
        infer_noise_scheduler_cfg: dict,
        train_cfg: dict,
        test_cfg: dict,
        train_frames: int,
        mean_std: str = "assets/motion_stats.json",
        smpl_model_path: str = "assets/body_models/smplh/neutral/model.npz",
        j_regressor_path: str = "assets/body_models/smpl_neutral_J_regressor.pt",
        pred_type: str = "velocity",
        motion_style: str = "basic",
        fps: float = 30.0,
        **kwargs,
    ):
        assert motion_style in ("basic", "global", "mixed"), \
            f"motion_style must be 'basic'/'global'/'mixed', got '{motion_style}'"

        self.motion_style = motion_style
        self.fps = fps
        self.latent_dim = self.LATENT_DIM

        self._component_split_sizes = list(self.COMPONENT_DIMS.values())
        self._component_names = list(self.COMPONENT_DIMS.keys())

        parent_losses_cfg = {
            "recons": losses_cfg.get("recons", {"name": "SmoothL1Loss", "weight": 1.0}),
        }
        if "vertex" in losses_cfg:
            parent_losses_cfg["vertex"] = losses_cfg["vertex"]

        super().__init__(
            network_module=network_module,
            network_module_args=network_module_args,
            losses_cfg=parent_losses_cfg,
            noise_scheduler_cfg=noise_scheduler_cfg,
            infer_noise_scheduler_cfg=infer_noise_scheduler_cfg,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            train_frames=train_frames,
            mean_std=mean_std,
            smpl_model_path=smpl_model_path,
            j_regressor_path=j_regressor_path,
            pred_type=pred_type,
            **kwargs,
        )

        self.losses_cfg = losses_cfg


    def load_mean_std(self, mean_std_path: str) -> None:
        with open(mean_std_path, "r") as f:
            stats = json.load(f)

        def _register_pair(name: str, key: str, shape_suffix: tuple = ()):
            mean = torch.FloatTensor(stats[key]["mean"])
            std = torch.FloatTensor(stats[key]["std"])
            if shape_suffix:
                mean = mean.reshape(*shape_suffix)
                std = std.reshape(*shape_suffix)
            std = std.clamp(min=1e-5)
            self.register_buffer(f"{name}_mean", mean[None, None])  # (1,1,...)
            self.register_buffer(f"{name}_std", std[None, None])

        _register_pair("smooth_root_vel", "smooth_root_vel")                         # (1,1,3)
        _register_pair("smooth_root_pos", "smooth_root_pos")                         # (1,1,3)
        _register_pair("local_joints_positions", "local_joints_positions", (52, 3))  # (1,1,52,3)
        _register_pair("global_rot_data", "global_rot_data", (52, 6))               # (1,1,52,6)
        _register_pair("local_rot_data", "local_rot_data", (52, 6))                 # (1,1,52,6)
        _register_pair("rep_shapes", "shapes")                                    # (1,1,16)
        _register_pair("rep_foot_contacts", "foot_contacts")                      # (1,1,4)


    def _ensure_motion_format(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if "smooth_root_vel" in batch:
            return batch
        from ..core.motion.motion_rep import _encode_motion_base_style
        return _encode_motion_base_style(batch, self.body_model, fps=self.fps, n_joints_pos=52)

    # ==================== Encode ====================

    def encode_motion(self, batch: Dict[str, Tensor]) -> Tensor:
        batch = self._ensure_motion_format(batch)

        # 1. Root velocity XZ (2D)
        srv = batch["smooth_root_vel"]                         # (B,T,3)
        srv_xz = srv[..., [0, 2]]
        srv_xz_norm = (srv_xz - self.smooth_root_vel_mean[..., [0, 2]]) / \
                       self.smooth_root_vel_std[..., [0, 2]]

        # 2. Root position Y (1D)
        srp_y = batch["smooth_root_pos"][..., 1:2]            # (B,T,1)
        srp_y_norm = (srp_y - self.smooth_root_pos_mean[..., 1:2]) / \
                      self.smooth_root_pos_std[..., 1:2]

        # 3. Local joint positions (52×3=156)
        ljp = batch["local_joints_positions"]                  # (B,T,52,3)
        ljp_norm = (ljp - self.local_joints_positions_mean) / self.local_joints_positions_std
        B, T = ljp_norm.shape[:2]
        ljp_flat = ljp_norm.reshape(B, T, -1)

        if self.motion_style == "basic":
            rot = batch["local_rot_data"]                      # (B,T,52,6)
            rot_norm = (rot - self.local_rot_data_mean) / self.local_rot_data_std
        elif self.motion_style == "global":
            rot = batch["global_rot_data"]                     # (B,T,52,6)
            rot_norm = (rot - self.global_rot_data_mean) / self.global_rot_data_std
        elif self.motion_style == "mixed":
            rot_body = batch["global_rot_data"][:, :, :22]     # (B,T,22,6)
            rot_hand = batch["local_rot_data"][:, :, 22:]      # (B,T,30,6)
            rot_body_norm = (rot_body - self.global_rot_data_mean[:, :, :22]) / \
                             self.global_rot_data_std[:, :, :22]
            rot_hand_norm = (rot_hand - self.local_rot_data_mean[:, :, 22:]) / \
                             self.local_rot_data_std[:, :, 22:]
            rot_norm = torch.cat([rot_body_norm, rot_hand_norm], dim=2)
        rot_flat = rot_norm.reshape(B, T, -1)

        # 5. Shapes (16)
        shapes_norm = (batch["shapes"] - self.rep_shapes_mean) / self.rep_shapes_std

        # 6. Foot contacts (4)
        fc_norm = (batch["foot_contacts"] - self.rep_foot_contacts_mean) / \
                   self.rep_foot_contacts_std

        # 7. Concat: 2+1+156+312+16+4 = 491
        return torch.cat([srv_xz_norm, srp_y_norm, ljp_flat, rot_flat, shapes_norm, fc_norm], dim=-1)

    # ==================== Decode ====================

    def _decode_latent_to_motion(self, latent: Tensor, fps: float = None) -> Dict[str, Tensor]:
        fps = fps or self.fps
        B, T, D = latent.shape

        # 1. Split by component
        parts = torch.split(latent, self._component_split_sizes, dim=-1)
        srv_xz_norm = parts[0]
        srp_y_norm = parts[1]
        ljp_flat = parts[2]
        rot_flat = parts[3]
        shapes_norm = parts[4]
        fc_norm = parts[5]

        # 2. Denormalize root
        srv_xz = srv_xz_norm * self.smooth_root_vel_std[..., [0, 2]] + \
                  self.smooth_root_vel_mean[..., [0, 2]]
        srp_y = srp_y_norm * self.smooth_root_pos_std[..., 1:2] + \
                 self.smooth_root_pos_mean[..., 1:2]

        # 3. Denormalize joint positions
        ljp = ljp_flat.reshape(B, T, 52, 3)
        ljp = ljp * self.local_joints_positions_std + self.local_joints_positions_mean

        # 4. Denormalize rotations → recover local SMPL rotations
        rot_data = rot_flat.reshape(B, T, 52, 6)
        device = latent.device

        if self.motion_style == "basic":
            local_rot_data = rot_data * self.local_rot_data_std + self.local_rot_data_mean
            root_rot6d = local_rot_data[:, :, 0]
            body_rot6d = local_rot_data[:, :, 1:]

        elif self.motion_style == "global":
            global_rot_data = rot_data * self.global_rot_data_std + self.global_rot_data_mean
            # global → local rotations
            global_rot_mats = rot6d_to_rotation_matrix(
                global_rot_data.reshape(B * T * 52, 6)
            ).reshape(B, T, 52, 3, 3)
            local_rot_mats = _global_rots_to_local_rots(
                global_rot_mats, _SMPLH_PARENTS.to(device)
            )
            root_rot6d = rotation_matrix_to_rot6d(local_rot_mats[:, :, 0])
            body_rot6d = rotation_matrix_to_rot6d(local_rot_mats[:, :, 1:])

        elif self.motion_style == "mixed":
            # body(0:22) global, hand(22:52) local
            rot_body = rot_data[:, :, :22]
            rot_hand = rot_data[:, :, 22:]
            rot_body = rot_body * self.global_rot_data_std[:, :, :22] + \
                        self.global_rot_data_mean[:, :, :22]
            rot_hand = rot_hand * self.local_rot_data_std[:, :, 22:] + \
                        self.local_rot_data_mean[:, :, 22:]

            # body global → local
            global_body_mats = rot6d_to_rotation_matrix(
                rot_body.reshape(B * T * 22, 6)
            ).reshape(B, T, 22, 3, 3)
            local_body_mats = _global_rots_to_local_rots(
                global_body_mats, _SMPLH_PARENTS[:22].to(device)
            )
            root_rot6d = rotation_matrix_to_rot6d(local_body_mats[:, :, 0])
            body_body_rot6d = rotation_matrix_to_rot6d(local_body_mats[:, :, 1:])
            # hand already local
            body_rot6d = torch.cat([body_body_rot6d, rot_hand], dim=2)

        shapes = shapes_norm * self.rep_shapes_std + self.rep_shapes_mean
        shapes = shapes.mean(dim=1, keepdim=True)

        # 6. Foot contacts
        foot_contacts = fc_norm * self.rep_foot_contacts_std + self.rep_foot_contacts_mean

        # 7. Recover smooth_root_pos from velocity
        smooth_root_pos = torch.zeros(B, T, 3, device=device, dtype=latent.dtype)
        if T > 1:
            smooth_root_pos[:, 1:, 0] = torch.cumsum(srv_xz[:, 1:, 0] / fps, dim=1)
            smooth_root_pos[:, 1:, 2] = torch.cumsum(srv_xz[:, 1:, 1] / fps, dim=1)
        smooth_root_pos[..., 1] = srp_y.squeeze(-1)

        # 8. Recover pelvis world position from local_joints_positions + smooth_root
        pelvis_local = ljp[:, :, 0, :]
        pelvis_world = pelvis_local.clone()
        pelvis_world[..., 0] += smooth_root_pos[..., 0]
        pelvis_world[..., 2] += smooth_root_pos[..., 2]
        # Y is absolute (no adjustment needed)

        # 9. Full rot6d stack
        rot6d = torch.cat([root_rot6d[:, :, None], body_rot6d], dim=2)

        return {
            "root_rot6d": root_rot6d,
            "body_rot6d": body_rot6d,
            "rot6d": rot6d,
            "local_joints_positions": ljp,
            "foot_contacts": foot_contacts,
            "smooth_root_pos": smooth_root_pos,
            "shapes": shapes,
            "pelvis_world": pelvis_world,
        }

    def _motion_to_smplh(self, motion_fields: Dict[str, Tensor]) -> Dict[str, Tensor]:
        pelvis_world = motion_fields["pelvis_world"]    # (B, T, 3)
        shapes = motion_fields["shapes"]                # (B, 1, 16)
        rot6d = motion_fields["rot6d"]                  # (B, T, 52, 6)

        j_shaped = self.body_model.compute_j_shaped(shapes.squeeze(1))
        pelvis_rest = j_shaped[:, 0:1, :]

        trans = pelvis_world - pelvis_rest

        return {
            "rot6d": rot6d,
            "shapes": shapes,
            "trans": trans,
        }

    def decode_motion(self, latent: Tensor, fps: float = None, **kwargs) -> Dict[str, Tensor]:
        motion_fields = self._decode_latent_to_motion(latent, fps=fps)

        smplh_params = self._motion_to_smplh(motion_fields)

        return {
            "rot6d": smplh_params["rot6d"],
            "shapes": smplh_params["shapes"],
            "trans": smplh_params["trans"],
            "root_rot6d": motion_fields["root_rot6d"],
            "body_rot6d": motion_fields["body_rot6d"],
            "local_joints_positions": motion_fields["local_joints_positions"],
            "foot_contacts": motion_fields["foot_contacts"],
            "smooth_root_pos": motion_fields["smooth_root_pos"],
            "pelvis_world": motion_fields["pelvis_world"],
        }


    def compute_loss(
        self,
        pred: Tensor,
        gt: Tensor,
        pred_decode: Optional[Dict[str, Tensor]] = None,
        gt_decode: Optional[Dict[str, Tensor]] = None,
        data_mask_temporal: Optional[Tensor] = None,
    ) -> tuple:
        loss_fn = torch.nn.functional.smooth_l1_loss
        loss_dict = {}

        pred_parts = torch.split(pred, self._component_split_sizes, dim=-1)
        gt_parts = torch.split(gt, self._component_split_sizes, dim=-1)

        for i, name in enumerate(self._component_names):
            loss_dict[name] = loss_fn(
                pred_parts[i], gt_parts[i], reduction="none"
            ).mean(dim=-1)

        need_fk_rotation = self.losses_cfg.get("fk_rotation", {}).get("weight", 0.0) > 0
        need_fk_consistency = self.losses_cfg.get("fk_consistency", {}).get("weight", 0.0) > 0
        if (
            pred_decode is not None
            and gt_decode is not None
            and (need_fk_rotation or need_fk_consistency)
        ):
            # FK on predicted rotations
            pred_shapes_expanded = pred_decode["shapes"].expand(-1, pred_decode["root_rot6d"].shape[1], -1)
            gt_shapes_expanded = gt_decode["shapes"].expand(-1, gt_decode["root_rot6d"].shape[1], -1)

            pred_kp3d, pred_transforms = _run_fk(
                self.body_model,
                pred_decode["root_rot6d"], pred_decode["body_rot6d"],
                pred_shapes_expanded, pred_decode["trans"],
            )
            gt_kp3d, gt_transforms = _run_fk(
                self.body_model,
                gt_decode["root_rot6d"], gt_decode["body_rot6d"],
                gt_shapes_expanded, gt_decode["trans"],
            )

            B_fk, T_fk = pred_kp3d.shape[:2]
            if need_fk_rotation:
                pred_global_rot6d = rotation_matrix_to_rot6d(pred_transforms[:, :, :, :3, :3])
                gt_global_rot6d = rotation_matrix_to_rot6d(gt_transforms[:, :, :, :3, :3])
                fk_rot_loss = loss_fn(
                    pred_global_rot6d, gt_global_rot6d, reduction="none"
                )
                loss_dict["fk_rotation"] = fk_rot_loss.reshape(B_fk, T_fk, -1).mean(dim=-1)

            if need_fk_consistency:
                fk_pos_loss = loss_fn(pred_kp3d, gt_kp3d, reduction="none")
                loss_dict["fk_consistency"] = fk_pos_loss.reshape(B_fk, T_fk, -1).mean(dim=-1)

        if "vertex" in self.losses_cfg and pred_decode is not None and gt_decode is not None:
            loss_dict["vertex"] = self.vertex_loss(
                pred_decode, gt_decode, self.global_iteration
            )

        if "transroll" in self.losses_cfg and pred_decode is not None and gt_decode is not None:
            transroll_cfg = self.losses_cfg["transroll"]
            start_step = int(transroll_cfg.get("start_step", 0))
            overlap_step = int(transroll_cfg.get("overlap_step", 1))
            if self.global_iteration < start_step:
                warmup_scale = 0.0
            elif overlap_step <= 0 or self.global_iteration > start_step + overlap_step:
                warmup_scale = 1.0
            else:
                warmup_scale = (self.global_iteration - start_step) / overlap_step
            transroll_loss = loss_fn(
                pred_decode["trans"], gt_decode["trans"], reduction="none"
            ).mean(dim=-1)
            loss_dict["transroll"] = transroll_loss * warmup_scale

        loss_weight = {}
        for key in loss_dict:
            if key in self.losses_cfg:
                loss_weight[key] = self.losses_cfg[key].get("weight", self.DEFAULT_LOSS_WEIGHTS.get(key, 1.0))
            else:
                loss_weight[key] = self.DEFAULT_LOSS_WEIGHTS.get(key, 1.0)

        # temporal mask + mean
        loss_dict_mean = {}
        for key, val in loss_dict.items():
            if data_mask_temporal is not None:
                loss_dict_mean[key] = (val * data_mask_temporal).sum() / data_mask_temporal.sum()
            else:
                loss_dict_mean[key] = val.mean()

        return loss_dict_mean, loss_weight


    def _prepare_grpo_conditioning(self, feature, T, length, num_generations, cfg_scale):
        device = next(iter(feature.values())).device
        do_cfg = cfg_scale > 1
        rep = num_generations

        feature_rep = {k: v.clone() for k, v in feature.items()}
        for key in feature_rep:
            feature_rep[key] = feature_rep[key].repeat((rep,) + (1,) * (feature_rep[key].dim() - 1))
        if do_cfg:
            for key in feature_rep:
                if key == "feature":
                    feature_rep[key] = torch.cat([torch.zeros_like(feature_rep[key]), feature_rep[key]], dim=0)
                else:
                    feature_rep[key] = torch.cat([feature_rep[key], feature_rep[key]], dim=0)

        vtxt_input = torch.zeros((rep, 1, self.motion_transformer.vtxt_input_dim), device=device)
        length_tensor = torch.tensor([length], device=device)
        x_mask_temporal = length_to_mask(length_tensor, T)
        x_mask_temporal = x_mask_temporal.repeat((rep,) + (1,) * (x_mask_temporal.dim() - 1))
        if do_cfg:
            vtxt_input = torch.cat([vtxt_input] * 2, dim=0)
            x_mask_temporal = torch.cat([x_mask_temporal] * 2, dim=0)
        return feature_rep, vtxt_input, x_mask_temporal, do_cfg

    def _grpo_velocity(self, x, t_scalar, feature_rep, vtxt_input, x_mask_temporal, y0, cfg_scale, do_cfg):
        device = x.device
        t = torch.tensor(float(t_scalar), device=device, dtype=x.dtype)
        x_input = torch.cat([x] * 2, dim=0) if do_cfg else x
        x_pred = self.motion_transformer(
            x=x_input,
            ctxt_input=feature_rep,
            vtxt_input=vtxt_input,
            timesteps=t.expand(x_input.shape[0]),
            x_mask_temporal=x_mask_temporal,
            ctxt_mask_temporal=x_mask_temporal,
        )
        if self.pred_type == "velocity":
            pass
        elif self.pred_type == "x1":
            t_eps = 1 / self.validation_steps
            x_pred = (x_pred - x_input) / (1.0 - t).clamp_min(t_eps)
        elif self.pred_type == "x1raw":
            x_pred = x_pred - (torch.cat([y0] * 2, dim=0) if do_cfg else y0)
        else:
            raise NotImplementedError(f"unsupported pred_type: {self.pred_type}")
        if do_cfg:
            x_pred_basic, x_pred_text = x_pred.chunk(2, dim=0)
            x_pred = x_pred_basic + cfg_scale * (x_pred_text - x_pred_basic)
        return x_pred

    @torch.no_grad()
    def grpo_sample(
        self,
        feature: Dict[str, Tensor],
        length: int,
        num_generations: int,
        sampling_steps: int,
        eta: float,
        timesteps_train: List[int],
        init_noise: Optional[Tensor] = None,
        cfg_scale: float = 1.0,
    ) -> Dict[str, Any]:
        from ..utils.motion_sde import flow_sde_step

        feat_val = next(iter(feature.values()))
        T = feat_val.shape[1]
        device = feat_val.device
        dtype = feat_val.dtype

        feature_rep, vtxt_input, x_mask_temporal, do_cfg = self._prepare_grpo_conditioning(
            feature, T, length, num_generations, cfg_scale
        )

        if init_noise is None:
            init_noise = torch.randn((1, T, self.latent_dim), device=device, dtype=dtype)
        y0 = init_noise.repeat((num_generations,) + (1,) * (init_noise.dim() - 1))

        deterministic = [True] * sampling_steps
        for i in timesteps_train:
            if 0 <= i < sampling_steps:
                deterministic[i] = False

        t_grid = torch.linspace(0, 1, sampling_steps + 1, device=device, dtype=dtype)

        x = y0
        all_latents = [x]
        all_log_probs = []
        for i in range(sampling_steps):
            v = self._grpo_velocity(
                x, t_grid[i].item(), feature_rep, vtxt_input, x_mask_temporal, y0, cfg_scale, do_cfg
            )
            x, _pred_x1, log_prob, _mean, _std = flow_sde_step(
                v_pred=v,
                x_t=x,
                t=t_grid[i].item(),
                t_next=t_grid[i + 1].item(),
                eta=eta,
                deterministic=deterministic[i],
            )
            all_latents.append(x)
            all_log_probs.append(log_prob)

        all_latents = torch.stack(all_latents, dim=1) # (G, steps+1, T, D)
        all_log_probs = torch.stack(all_log_probs, dim=1) # (G, steps)

        output = self.decode_motion(x, fps=self.fps)

        return {
            "all_latents": all_latents,
            "all_log_probs": all_log_probs,
            "output": output,
            "conditioning": {
                "feature_rep": feature_rep,
                "vtxt_input": vtxt_input,
                "x_mask_temporal": x_mask_temporal,
                "y0": y0,
                "do_cfg": do_cfg,
                "t_grid": t_grid,
            },
        }

    def grpo_one_step(
        self,
        latents: Tensor,
        prev_latents: Tensor,
        step_index: int,
        sampling_steps: int,
        eta: float,
        conditioning: Dict[str, Any],
        cfg_scale: float = 1.0,
        return_velocity: bool = False,
    ) -> Tensor:
        from ..utils.motion_sde import flow_sde_step

        t_grid = conditioning["t_grid"]
        v = self._grpo_velocity(
            latents,
            t_grid[step_index].item(),
            conditioning["feature_rep"],
            conditioning["vtxt_input"],
            conditioning["x_mask_temporal"],
            conditioning["y0"],
            cfg_scale,
            conditioning["do_cfg"],
        )
        _x, _pred_x1, log_prob, _mean, _std = flow_sde_step(
            v_pred=v,
            x_t=latents,
            t=t_grid[step_index].item(),
            t_next=t_grid[step_index + 1].item(),
            eta=eta,
            prev_sample=prev_latents,
            deterministic=False,
        )
        if return_velocity:
            return log_prob, v
        return log_prob


    @torch.no_grad()
    def generate(
        self,
        feature: Dict[str, Tensor],
        seeds: List[int],
        length: int,
        cfg_scale: float = 1.0,
    ) -> Dict[str, Tensor]:
        feat_val = next(iter(feature.values()))
        T = feat_val.shape[1]
        device = feat_val.device
        dtype = feat_val.dtype
        repeat = len(seeds)

        do_classifier_free_guidance = cfg_scale > 1

        feature = {k: v.clone() for k, v in feature.items()}
        if cfg_scale == 1:
            for key in feature:
                feature[key] = feature[key].repeat((repeat,) + (1,) * (feature[key].dim() - 1))
        else:
            for key in feature:
                feature[key] = feature[key].repeat((repeat,) + (1,) * (feature[key].dim() - 1))
                if key == "feature":
                    feature[key] = torch.cat([torch.zeros_like(feature[key]), feature[key]], dim=0)
                else:
                    feature[key] = torch.cat([feature[key], feature[key]], dim=0)

        vtxt_input = torch.zeros((repeat, 1, self.motion_transformer.vtxt_input_dim), device=device)

        length_tensor = torch.tensor([length], device=device)
        x_mask_temporal = length_to_mask(length_tensor, T)
        x_mask_temporal = x_mask_temporal.repeat((repeat,) + (1,) * (x_mask_temporal.dim() - 1))

        if do_classifier_free_guidance:
            x_mask_temporal = torch.cat([x_mask_temporal] * 2, dim=0)
            vtxt_input = torch.cat([vtxt_input] * 2, dim=0)

        noise_shape = (1, T, self.latent_dim)
        y0_list = []
        for seed in seeds:
            generator = torch.Generator().manual_seed(seed)
            noise = randn_tensor(noise_shape, generator=generator, dtype=dtype).to(device)
            y0_list.append(noise)
        y0 = torch.cat(y0_list, dim=0)

        def fn(t: Tensor, x: Tensor) -> Tensor:
            x_input = torch.cat([x] * 2, dim=0) if do_classifier_free_guidance else x
            x_pred = self.motion_transformer(
                x=x_input,
                ctxt_input=feature,
                vtxt_input=vtxt_input,
                timesteps=t.expand(x_input.shape[0]),
                x_mask_temporal=x_mask_temporal,
                ctxt_mask_temporal=x_mask_temporal,
            )

            if self.pred_type == "velocity":
                pass
            elif self.pred_type == "x1":
                t_eps = 1 / self.validation_steps
                x_pred = (x_pred - x_input) / (1.0 - t).clamp_min(t_eps)
            elif self.pred_type == "x1raw":
                x_pred = x_pred - y0

            if do_classifier_free_guidance:
                x_pred_basic, x_pred_text = x_pred.chunk(2, dim=0)
                x_pred = x_pred_basic + cfg_scale * (x_pred_text - x_pred_basic)

            return x_pred

        t = torch.linspace(0, 1, self.validation_steps + 1, device=device, dtype=dtype)
        trajectory = odeint_custom(fn, y0, t, **self._noise_scheduler_cfg)
        sampled: Tensor = trajectory[-1]

        output = self.decode_motion(sampled, fps=self.fps)

        for key in output:
            if isinstance(output[key], Tensor) and output[key].dim() >= 2:
                output[key] = output[key][:, :length]

        return output

import json
import logging
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor

from ..core.bodymodels.smpl_skeleton import SMPLMesh, SMPLSkeleton
from ..core.bodymodels.fk_utils import get_vertices_from_smpl_params
from ..core.math.geometry import rot6d_to_rotation_matrix
from ..core.framework.loaders import load_object
from ..core.motion.postprocess import end_vel_to_static_conf
from ..evaluation.metrics import compute_global_metrics

logger = logging.getLogger(__name__)

from .ode_solvers import odeint_custom
from .utils import length_to_mask, rollout_local_transl_vel, randn_tensor
from .losses import CleanLoss, FKLoss, VertexLoss, TransRollLoss, Joint2DLoss


class SimpleV2MPipeline(torch.nn.Module):

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
        **kwargs,
    ):
        super().__init__()

        self.motion_transformer = load_object(network_module, network_module_args)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.train_frames = train_frames
        self.losses_cfg = losses_cfg
        self._noise_scheduler_cfg = noise_scheduler_cfg
        self._infer_noise_scheduler_cfg = infer_noise_scheduler_cfg
        self.pred_type = pred_type

        self.body_model = SMPLSkeleton(model_path=smpl_model_path)

        if pred_type in ("x1", "x1raw"):
            if "vertex" in losses_cfg:
                self.mesh_model = SMPLMesh(model_path=smpl_model_path)
                self.vertex_loss = VertexLoss(self.mesh_model, **losses_cfg["vertex"])
            if "transroll" in losses_cfg:
                self.transroll_loss = TransRollLoss(**losses_cfg["transroll"])
            if "joint2d" in losses_cfg:
                self.joint2d_loss = Joint2DLoss(**losses_cfg["joint2d"])
            if "clean" in losses_cfg:
                self.clean_loss = CleanLoss(**losses_cfg["clean"])
            if "fk_consistency" in losses_cfg:
                fk_cfg = {k: v for k, v in losses_cfg["fk_consistency"].items() if k != "weight"}
                self.fk_loss = FKLoss(self.body_model, **fk_cfg)

        if not hasattr(self, "mesh_model"):
            self.mesh_model = SMPLMesh(model_path=smpl_model_path)

        self.register_buffer(
            "J_regressor",
            torch.load(j_regressor_path, map_location="cpu"),
        )

        self._parse_train_cfg()
        self._parse_test_cfg()

        self.load_mean_std(mean_std)

        self.global_iteration = -1


    def _parse_train_cfg(self) -> None:
        self.cond_mask_prob = self.train_cfg.get("cond_mask_prob", 0.0)

    def _parse_test_cfg(self) -> None:
        self.validation_steps = self._infer_noise_scheduler_cfg["validation_steps"]
        self.text_guidance_scale = self.test_cfg.get("text_guidance_scale", 1)


    def load_mean_std(self, mean_std_path: str) -> None:
        with open(mean_std_path, "r") as f:
            mean_std = json.load(f)

        # root_rot6d, body_rot6d, transl_vel, shapes
        for key in ["root_rot6d", "body_rot6d", "transl_vel", "shapes"]:
            mean = torch.FloatTensor(mean_std[key]["mean"])
            std = torch.FloatTensor(mean_std[key]["std"])
            self.register_buffer(f"{key}_mean", mean[None, None])
            self.register_buffer(f"{key}_std", std[None, None])

        # end_effector_vel
        mean = torch.FloatTensor(mean_std["end_effector_vel"]["mean"])
        std = torch.FloatTensor(mean_std["end_effector_vel"]["std"])
        self.register_buffer("end_effector_vel_mean", mean[None, None])
        self.register_buffer("end_effector_vel_std", std[None, None])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    # ==================== Encode / Decode ====================

    def encode_motion(self, batch: Dict[str, Tensor]) -> Tensor:
        root_rot6d = (batch["root_rot6d"] - self.root_rot6d_mean) / self.root_rot6d_std

        body_rot6d_std = self.body_rot6d_std.clone()
        body_rot6d_std[body_rot6d_std < 1e-3] = 1.0
        body_rot6d = (batch["body_rot6d"] - self.body_rot6d_mean) / body_rot6d_std
        body_rot6d = body_rot6d.reshape(body_rot6d.shape[0], body_rot6d.shape[1], -1)

        wv_rot6d_std = torch.cat([root_rot6d, body_rot6d], dim=-1)

        transl_vel = (batch["transl_vel"] - self.transl_vel_mean) / self.transl_vel_std
        wv_rot6d_transl_std = torch.cat([wv_rot6d_std, transl_vel], dim=-1)

        body_shape = (batch["shapes"] - self.shapes_mean) / self.shapes_std
        wv_rot6d_transl_shape_std = torch.cat([wv_rot6d_transl_std, body_shape], dim=-1)

        end_effector_vel = (batch["end_effector_vel"] - self.end_effector_vel_mean) / self.end_effector_vel_std
        end_effector_vel = end_effector_vel.reshape(end_effector_vel.shape[0], end_effector_vel.shape[1], -1)

        return torch.cat([wv_rot6d_transl_shape_std, end_effector_vel], dim=-1)

    def decode_motion(self, latent: Tensor, fps: int = 30, **kwargs) -> Dict[str, Tensor]:
        root_rot6d = latent[:, :, :6] * self.root_rot6d_std + self.root_rot6d_mean

        body_rot6d = latent[:, :, 6:-37].reshape(latent.shape[0], latent.shape[1], 51, 6)
        body_rot6d_std = self.body_rot6d_std.clone()
        body_rot6d_std[body_rot6d_std < 1e-3] = 1.0
        body_rot6d = body_rot6d * body_rot6d_std + self.body_rot6d_mean

        transl_vel = latent[:, :, -37:-34] * self.transl_vel_std + self.transl_vel_mean

        root_rotmat = rot6d_to_rotation_matrix(root_rot6d)

        rot6d = torch.cat([root_rot6d[:, :, None], body_rot6d], dim=-2)

        shapes = latent[:, :, -34:-18] * self.shapes_std + self.shapes_mean
        shapes = shapes.mean(dim=-2, keepdim=True)

        end_effector_vel = latent[:, :, -18:].reshape(latent.shape[0], latent.shape[1], 6, 3)
        end_effector_vel = end_effector_vel * self.end_effector_vel_std + self.end_effector_vel_mean

        trans = rollout_local_transl_vel(transl_vel, root_rotmat, fps=fps)

        return {
            "rot6d": rot6d,
            "shapes": shapes,
            "trans": trans,
            "global_orient": root_rotmat,
            "local_transl_vel": transl_vel,
            "end_effector_vel": end_effector_vel,
        }


    @staticmethod
    def noise_from_seeds(latent: Tensor, seeds: Union[int, List[int]], seed_start: int = 0) -> Tensor:
        if isinstance(seeds, int):
            seeds = list(range(seeds))
        noise_list = []
        B = latent.shape[0]
        shape = (B, *latent.shape[1:])
        for seed in seeds:
            generator = torch.Generator().manual_seed(seed + seed_start)
            noise_sample = randn_tensor(shape, generator=generator, dtype=latent.dtype).to(latent.device)
            noise_list.append(noise_sample)
        return torch.cat(noise_list, dim=0)


    def compute_loss(
        self,
        pred: Tensor,
        gt: Tensor,
        pred_decode: Optional[Dict[str, Tensor]] = None,
        gt_decode: Optional[Dict[str, Tensor]] = None,
        data_mask_temporal: Optional[Tensor] = None,
    ) -> tuple:
        loss_fns = {
            "SmoothL1Loss": torch.nn.functional.smooth_l1_loss,
            "MSELoss": torch.nn.functional.mse_loss,
        }
        loss_dict = {}
        loss_weight = {}

        recon_type = self.losses_cfg["recons"]["name"]
        loss_fn = loss_fns[recon_type]

        # Motion rep 349-dim layout (wvrot6d_transl_shape_stationary_std):
        if "recons_decomposed" in self.losses_cfg:
            decomp_cfg = self.losses_cfg["recons_decomposed"]

            # 1. heading loss (root_rot6d): dims [0:6]
            heading_loss = loss_fn(pred[:, :, :6], gt[:, :, :6], reduction="none").mean(dim=-1)
            loss_dict["heading"] = heading_loss
            loss_weight["heading"] = decomp_cfg.get("heading_weight", 2.0)

            # 2. joint_rot loss (body_rot6d): dims [6:312]
            joint_rot_loss = loss_fn(pred[:, :, 6:312], gt[:, :, 6:312], reduction="none").mean(dim=-1)
            loss_dict["joint_rot"] = joint_rot_loss
            loss_weight["joint_rot"] = decomp_cfg.get("joint_rot_weight", 10.0)

            # 3. root_pos loss (transl_vel): dims [312:315]
            root_pos_loss = loss_fn(pred[:, :, 312:315], gt[:, :, 312:315], reduction="none").mean(dim=-1)
            loss_dict["root_pos"] = root_pos_loss
            loss_weight["root_pos"] = decomp_cfg.get("root_pos_weight", 10.0)

            # 4. shapes loss: dims [315:331]
            shapes_loss = loss_fn(pred[:, :, 315:331], gt[:, :, 315:331], reduction="none").mean(dim=-1)
            loss_dict["shapes"] = shapes_loss
            loss_weight["shapes"] = decomp_cfg.get("shapes_weight", 1.0)

            # 5. foot_contact loss (end_effector_vel): dims [331:349]
            foot_contact_loss = loss_fn(pred[:, :, 331:349], gt[:, :, 331:349], reduction="none").mean(dim=-1)
            loss_dict["foot_contact"] = foot_contact_loss
            loss_weight["foot_contact"] = decomp_cfg.get("foot_contact_weight", 4.0)

            # 6. joint_vel loss: finite difference of body_rot6d over time
            pred_body_vel = pred[:, 1:, 6:312] - pred[:, :-1, 6:312]
            gt_body_vel = gt[:, 1:, 6:312] - gt[:, :-1, 6:312]
            joint_vel_loss_raw = loss_fn(pred_body_vel, gt_body_vel, reduction="none").mean(dim=-1)
            joint_vel_loss = torch.zeros_like(loss_dict["heading"])
            joint_vel_loss[:, 1:] = joint_vel_loss_raw
            loss_dict["joint_vel"] = joint_vel_loss
            loss_weight["joint_vel"] = decomp_cfg.get("joint_vel_weight", 3.0)
        else:
            recon_loss = loss_fn(pred, gt, reduction="none").mean(dim=-1)
            loss_dict["recons"] = recon_loss
            loss_weight["recons"] = self.losses_cfg["recons"]["weight"]

        # ====== FK consistency loss ======
        if "fk_consistency" in self.losses_cfg and pred_decode is not None:
            fk_loss = self.fk_loss(pred_decode, gt_decode, self.global_iteration)
            loss_dict["fk_consistency"] = fk_loss
            loss_weight["fk_consistency"] = self.losses_cfg["fk_consistency"]["weight"]

        if "vertex" in self.losses_cfg:
            loss_dict["vertex"] = self.vertex_loss(pred_decode, gt_decode, self.global_iteration)
            loss_weight["vertex"] = self.losses_cfg["vertex"]["weight"]

        # roll-out translation loss
        if "transroll" in self.losses_cfg:
            loss_dict["transroll"] = self.transroll_loss(pred_decode, gt_decode, self.global_iteration)
            loss_weight["transroll"] = self.losses_cfg["transroll"]["weight"]

        # joint 2D loss
        if "joint2d" in self.losses_cfg:
            loss_dict["joint2d"] = self.joint2d_loss(pred_decode, gt_decode, self.global_iteration)
            loss_weight["joint2d"] = self.losses_cfg["joint2d"]["weight"]

        for key, cfg in self.losses_cfg.items():
            if key not in loss_weight and isinstance(cfg, dict) and "weight" in cfg:
                loss_weight[key] = cfg["weight"]

        loss_dict_mean = {}
        for key, val in loss_dict.items():
            if data_mask_temporal is not None:
                loss_dict_mean[key] = (val * data_mask_temporal).sum() / data_mask_temporal.sum()
            else:
                loss_dict_mean[key] = val.mean()

        return loss_dict_mean, loss_weight


    def forward_in_training(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self.global_iteration += 1

        gt_motion = self.encode_motion(batch["target"])
        device = gt_motion.device
        length = batch["length"]
        max_length = length.max().item()

        ctxt_input = batch["inputs"]["feature"]

        gt_motion = gt_motion[:, :max_length]
        if isinstance(ctxt_input, dict):
            for key in ctxt_input:
                ctxt_input[key] = ctxt_input[key][:, :max_length]
        else:
            ctxt_input = ctxt_input[:, :max_length]

        vtxt_input = torch.zeros(
            (gt_motion.shape[0], 1, self.motion_transformer.vtxt_input_dim), device=device
        )

        x0 = torch.randn(gt_motion.shape).to(device)
        x1 = gt_motion

        if "timestep_sample_method" in self.train_cfg:
            if self.train_cfg["timestep_sample_method"] == "logit_normal":
                timesteps = (
                    torch.randn(gt_motion.shape[0]) * self.train_cfg["t_sample_P_std"]
                    + self.train_cfg["t_sample_P_mean"]
                )
                timesteps = torch.sigmoid(timesteps)
            else:
                raise NotImplementedError(
                    f"unsupported timestep_sample_method: {self.train_cfg['timestep_sample_method']}"
                )
            timesteps = timesteps.to(device)
        else:
            timesteps = torch.rand(
                (gt_motion.shape[0],),
                dtype=gt_motion.dtype,
            ).to(device)

        t = timesteps.unsqueeze(-1).unsqueeze(-1)
        phi = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # temporal mask
        x_mask_temporal = length_to_mask(length, gt_motion.shape[1])

        pred = self.motion_transformer(
            x=phi,
            ctxt_input=ctxt_input,
            vtxt_input=vtxt_input,
            timesteps=timesteps,
            x_mask_temporal=x_mask_temporal,
            ctxt_mask_temporal=x_mask_temporal,
            cond_mask_prob=self.cond_mask_prob,
        )

        pred_decode = None
        gt_decode = None

        if self.pred_type == "velocity":
            pass
        elif self.pred_type == "x1":
            pred_decode = self.decode_motion(pred)
            gt_decode = self.decode_motion(x1)
            t_eps = 1 / self.validation_steps
            flow = (x1 - phi) / (1 - t).clamp_min(t_eps)
            pred = (pred - phi) / (1 - t).clamp_min(t_eps)
        elif self.pred_type == "x1raw":
            pred_decode = self.decode_motion(pred)
            gt_decode = self.decode_motion(x1)
            pred = pred - x0
        else:
            raise NotImplementedError(f"unsupported pred_type: {self.pred_type}")

        loss_dict, loss_weight = self.compute_loss(
            pred,
            flow,
            pred_decode=pred_decode,
            gt_decode=gt_decode,
            data_mask_temporal=x_mask_temporal,
        )

        loss = sum(loss_dict[k] * loss_weight[k] for k in loss_dict.keys())

        return {
            "latent": gt_motion,
            "model_output": pred,
            "loss": loss,
            "loss_dict": loss_dict,
            "tensor_results": {
                "index": batch.get("index", None),
            },
        }


    @torch.no_grad()
    def validate(
        self,
        batch: Dict[str, Any],
        seeds: List[int] = [0, 1, 2, 3],
        cfg: float = 1,
    ) -> Dict[str, Any]:
        length = batch["length"]
        gt_motion = self.encode_motion(batch["target"])
        device = gt_motion.device
        dtype = gt_motion.dtype

        feature = batch["inputs"]["feature"]
        repeat = len(seeds)

        do_classifier_free_guidance = cfg > 1

        if cfg == 1:
            if isinstance(feature, dict):
                for key in feature.keys():
                    feature[key] = feature[key].repeat((repeat,) + (1,) * (feature[key].dim() - 1))
            else:
                feature = feature.repeat((repeat,) + (1,) * (feature.dim() - 1))
        else:
            if isinstance(feature, dict):
                for key in feature.keys():
                    feature[key] = feature[key].repeat((repeat,) + (1,) * (feature[key].dim() - 1))
                    if key == "feature":
                        feature[key] = torch.cat([torch.zeros_like(feature[key]), feature[key]], dim=0)
                    else:
                        feature[key] = torch.cat([feature[key], feature[key]], dim=0)
            else:
                feature = feature.repeat((repeat,) + (1,) * (feature.dim() - 1))
                feature = torch.cat([torch.zeros_like(feature), feature], dim=0)

        vtxt_input = torch.zeros((repeat, 1, self.motion_transformer.vtxt_input_dim), device=device)

        # temporal mask
        x_mask_temporal = length_to_mask(length, gt_motion.shape[1])
        x_mask_temporal = x_mask_temporal.repeat((repeat,) + (1,) * (x_mask_temporal.dim() - 1))

        if do_classifier_free_guidance:
            x_mask_temporal = torch.cat([x_mask_temporal] * 2, dim=0)
            vtxt_input = torch.cat([vtxt_input] * 2, dim=0)

        y0 = self.noise_from_seeds(gt_motion, seeds)

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
            else:
                raise NotImplementedError(f"unsupported pred_type: {self.pred_type}")

            if do_classifier_free_guidance:
                x_pred_basic, x_pred_text = x_pred.chunk(2, dim=0)
                x_pred = x_pred_basic + cfg * (x_pred_text - x_pred_basic)

            return x_pred

        t = torch.linspace(0, 1, self.validation_steps + 1, device=device, dtype=dtype)
        trajectory = odeint_custom(fn, y0, t, **self._noise_scheduler_cfg)
        sampled: Tensor = trajectory[-1]

        # Decode
        output = self.decode_motion(sampled)
        gt_decode = self.decode_motion(gt_motion)

        bs = gt_motion.shape[0]
        metrics_list = []
        try:
            if "vertices" in batch.get("target", {}):
                gt_global_vertices = batch["target"]["vertices"]
            else:
                gt_verts = get_vertices_from_smpl_params(self.mesh_model, gt_decode)
                F_frames = gt_decode["rot6d"].shape[1]
                gt_global_vertices = gt_verts["global_vertices"].reshape(bs, F_frames, -1, 3)

            for i in range(repeat * bs):
                seed_idx = i // bs
                batch_idx = i % bs
                seq_len = length[batch_idx].item()

                pred_sample = {
                    "rot6d": output["rot6d"][i : i + 1, :seq_len],
                    "shapes": output["shapes"][i : i + 1],
                    "trans": output["trans"][i : i + 1, :seq_len],
                }
                if "end_effector_vel" in output:
                    pred_sample["end_effector_vel"] = output["end_effector_vel"][i : i + 1, :seq_len]

                gt_sample = {
                    "rot6d": gt_decode["rot6d"][batch_idx : batch_idx + 1, :seq_len],
                    "shapes": gt_decode["shapes"][batch_idx : batch_idx + 1],
                    "trans": gt_decode["trans"][batch_idx : batch_idx + 1, :seq_len],
                    "vertices": gt_global_vertices[batch_idx : batch_idx + 1, :seq_len],
                }

                metrics_batch = {"pred": pred_sample, "gt": gt_sample}
                sample_metrics = compute_global_metrics(
                    self.body_model, self.mesh_model, self.J_regressor, metrics_batch
                )
                sample_metrics["seed"] = seed_idx
                sample_metrics["batch_idx"] = batch_idx
                metrics_list.append(sample_metrics)
        except Exception as e:
            logger.warning(f"Failed to compute global metrics: {e}")

        result = {
            "pred": output,
            "gt": gt_decode,
            "length": length,
        }
        if metrics_list:
            result["metrics"] = metrics_list
        return result


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
            for key in feature.keys():
                feature[key] = feature[key].repeat((repeat,) + (1,) * (feature[key].dim() - 1))
        else:
            for key in feature.keys():
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

        noise_shape = (1, T, self.motion_transformer.motion_input_dim)
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
            else:
                raise NotImplementedError(f"unsupported pred_type: {self.pred_type}")

            if do_classifier_free_guidance:
                x_pred_basic, x_pred_text = x_pred.chunk(2, dim=0)
                x_pred = x_pred_basic + cfg_scale * (x_pred_text - x_pred_basic)

            return x_pred

        t = torch.linspace(0, 1, self.validation_steps + 1, device=device, dtype=dtype)
        trajectory = odeint_custom(fn, y0, t, **self._noise_scheduler_cfg)
        sampled: Tensor = trajectory[-1]

        # Decode
        output = self.decode_motion(sampled)

        for key in output:
            if isinstance(output[key], Tensor) and output[key].dim() >= 2:
                output[key] = output[key][:, :length]

        return output

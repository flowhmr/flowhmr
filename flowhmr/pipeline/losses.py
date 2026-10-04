import torch
from torch import Tensor

from ..core.bodymodels.smpl_skeleton import SMPLMesh, SMPLSkeleton
from ..core.bodymodels.fk_utils import get_joints_from_smpl_params
from .utils import rollout_local_transl_vel


class CleanLoss(torch.nn.Module):
    def __init__(self, name="SmoothL1Loss", default_weight=1.0):
        super().__init__()
        if name == "SmoothL1Loss":
            self.loss_func = torch.nn.functional.smooth_l1_loss
        elif name == "MSELoss":
            self.loss_func = torch.nn.functional.mse_loss
        else:
            raise ValueError(f"Unsupported loss function: {name}")
        self.default_weight = default_weight

    def forward(self, pred, gt):
        loss = self.loss_func(pred, gt, reduction="none").mean(dim=-1)
        return loss


class FKLoss(CleanLoss):
    def __init__(self, body_model: SMPLSkeleton, name="SmoothL1Loss", default_weight=1.0,
                 use_global=False, start_step=0, overlap_step=1):
        super().__init__(name=name, default_weight=default_weight)
        self.body_model = body_model
        self.use_global = use_global
        self.start_step = start_step
        self.overlap_step = overlap_step

    def forward_kinematics(self, pred, gt):
        pred_joints = get_joints_from_smpl_params(self.body_model, pred, joint_num=52)
        gt_joints = get_joints_from_smpl_params(self.body_model, gt, joint_num=52)
        return {
            "pred_joint_local": pred_joints["local_joints"],
            "pred_joint_global": pred_joints["global_joints"],
            "gt_joint_local": gt_joints["local_joints"],
            "gt_joint_global": gt_joints["global_joints"],
        }

    def forward(self, pred, gt, global_step):
        if global_step < self.start_step:
            trans = pred["trans"]
            return torch.zeros(trans.shape[0], trans.shape[1], device=trans.device, dtype=trans.dtype)
        if global_step > self.start_step + self.overlap_step:
            weight = self.default_weight
        else:
            weight = self.default_weight * (global_step - self.start_step) / self.overlap_step

        fk_results = self.forward_kinematics(pred, gt)

        if self.use_global:
            loss = self.loss_func(
                fk_results["pred_joint_global"], fk_results["gt_joint_global"], reduction="none"
            ).sum(dim=-1).mean(dim=-1)
        else:
            loss = self.loss_func(
                fk_results["pred_joint_local"], fk_results["gt_joint_local"], reduction="none"
            ).sum(dim=-1).mean(dim=-1)
        loss = loss * weight
        return loss


class VertexLoss(torch.nn.Module):
    def __init__(
        self,
        body_model: SMPLMesh,
        start_step=10000,
        overlap_step=10000,
        name="SmoothL1Loss",
        weight=1.0,
        num_sample_points=-1,
    ):
        super().__init__()
        self.body_model = body_model
        self.start_step = start_step
        self.overlap_step = overlap_step
        self.num_sample_points = num_sample_points
        if name == "SmoothL1Loss":
            self.loss_func = torch.nn.functional.smooth_l1_loss
        elif name == "MSELoss":
            self.loss_func = torch.nn.functional.mse_loss
        else:
            raise ValueError(f"Unsupported loss function: {name}")

    def calculate_smpl_mesh_vertices(self, batch, sample_indices=None):
        rot6d = batch.get("vertex_rot6d", batch["rot6d"])
        transl = batch["trans"]
        shapes = batch["shapes"]
        if shapes.dim() == 3 and shapes.shape[1] != rot6d.shape[1]:
            shapes = shapes.expand(-1, rot6d.shape[1], -1)
        rot6d_flat = rot6d.reshape(rot6d.shape[0] * rot6d.shape[1], -1, 6)
        transl_flat = torch.zeros(transl.shape[0] * transl.shape[1], 3, device=transl.device, dtype=transl.dtype)
        shapes_flat = shapes.reshape(shapes.shape[0] * shapes.shape[1], -1)

        params = {
            "rot6d": rot6d_flat,
            "trans": transl_flat,
            "shapes": shapes_flat,
        }

        out_vertices = self.body_model(params, sample_indices=sample_indices)
        out_vertices = out_vertices["vertices_wotrans"]
        out_vertices = out_vertices.reshape(rot6d.shape[0], rot6d.shape[1], -1, 3)
        return out_vertices

    def forward(self, pred, gt, global_step):
        if global_step <= self.start_step:
            trans = pred["trans"]
            return torch.zeros(trans.shape[0], trans.shape[1], device=trans.device, dtype=trans.dtype)
        if global_step > self.start_step + self.overlap_step:
            weight = 1.0
        else:
            weight = (global_step - self.start_step) / self.overlap_step

        if self.num_sample_points > 0:
            sample_indices = torch.randint(0, self.body_model.v_template.shape[0], (self.num_sample_points,))
        else:
            sample_indices = None
        pred_vertices = self.calculate_smpl_mesh_vertices(pred, sample_indices)
        gt_vertices = self.calculate_smpl_mesh_vertices(gt, sample_indices)

        loss = self.loss_func(pred_vertices, gt_vertices, reduction="none").sum(dim=-1).mean(dim=-1)
        loss = loss * weight
        return loss


class TransRollLoss(torch.nn.Module):
    def __init__(self, start_step=10000, overlap_step=10000, name="SmoothL1Loss", weight=1.0):
        super().__init__()
        self.start_step = start_step
        self.overlap_step = overlap_step
        if name == "SmoothL1Loss":
            self.loss_func = torch.nn.functional.smooth_l1_loss
        elif name == "MSELoss":
            self.loss_func = torch.nn.functional.mse_loss
        else:
            raise ValueError(f"Unsupported loss function: {name}")

    def forward(self, pred, gt, global_step):
        if global_step < self.start_step:
            trans = pred["trans"]
            return torch.zeros(trans.shape[0], trans.shape[1], device=trans.device, dtype=trans.dtype)
        if global_step > self.start_step + self.overlap_step:
            weight = 1.0
        else:
            weight = (global_step - self.start_step) / self.overlap_step

        gt_transl_w = gt["trans"]
        gt_global_orient_w = gt["global_orient"]
        local_transl_vel = pred["local_transl_vel"]

        pred_transl_w = rollout_local_transl_vel(local_transl_vel, gt_global_orient_w, fps=30)

        trans_w_loss = self.loss_func(pred_transl_w, gt_transl_w, reduction="none").mean(dim=-1)
        return trans_w_loss * weight


class Joint2DLoss(FKLoss):
    def __init__(
        self, body_model: SMPLMesh, start_step=10000, overlap_step=10000, name="SmoothL1Loss", default_weight=1.0
    ):
        super().__init__(body_model, name=name, default_weight=default_weight)
        self.start_step = start_step
        self.overlap_step = overlap_step

    def forward(self, pred, gt, global_step):
        if global_step < self.start_step:
            trans = pred["trans"]
            return torch.zeros(trans.shape[0], trans.shape[1], device=trans.device, dtype=trans.dtype)
        if global_step > self.start_step + self.overlap_step:
            weight = 1.0
        else:
            weight = (global_step - self.start_step) / self.overlap_step

        fk_results = self.forward_kinematics(pred, gt)
        # TODO: gt_j2d = normalize(fk_results["gt_joint_global"])
        pred_j2d = torch.zeros(1)
        gt_j2d = torch.zeros(1)

        loss = self.loss_func(pred_j2d, gt_j2d, reduction="none").sum(dim=-1).mean(dim=-1)
        loss = loss * weight
        return loss

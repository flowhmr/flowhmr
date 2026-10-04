import os
import sys
import time
import traceback
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from .v2m_trainer import V2MTrainer
from ..utils.grpo_states import GRPOTrainingStates
from ..core.math.geometry import (
    rot6d_to_rotation_matrix,
    rotation_matrix_to_angle_axis,
)

def _load_module_from_file(mod_name: str, rel_path: str):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, rel_path)
    import importlib.util
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mpjpe = _load_module_from_file("v2m_mpjpe_reward", "reward_model/mpjpe/mpjpe_reward.py")


class V2MGRPOTrainer(V2MTrainer):

    def __init__(self, config):
        super().__init__(config)
        g = config.get("grpo", {})
        self.num_generations = int(g.get("num_generations", 8))
        self.grpo_sampling_steps = int(g.get("sampling_steps", self.model.validation_steps))
        self.eta = float(g.get("eta", 0.7))
        self.cfg_scale = float(g.get("cfg_scale", 1.0))
        self.clip_range = float(g.get("clip_range", 1e-4))
        self.adv_clip_max = float(g.get("adv_clip_max", 5.0))
        self.kl_coeff = float(g.get("kl_coeff", 0.0))
        self.ref_kl_coeff = float(g.get("ref_kl_coeff", 0.0))
        self.init_same_noise = bool(g.get("init_same_noise", True))
        self.inner_updates = int(g.get("inner_updates", 1))
        self.adv_std_floor = float(g.get("adv_std_floor", 1e-4))
        self.manual_grad_sync = bool(g.get("manual_grad_sync", True))
        self.grad_bucket_numel = int(g.get("grad_bucket_numel", 128 * 1024 * 1024))
        self.grad_sync_dtype = str(g.get("grad_sync_dtype", "bf16")).lower()
        self.mpjpe_weight = float(g.get("mpjpe_weight", 1.0))
        self.tracking_weight = float(g.get("tracking_weight", 1.0))
        self.kinematics_weight = float(g.get("kinematics_weight", 0.0))
        self.kin_acc_scale = float(g.get("kin_acc_scale", 0.01))
        self.kin_skate_scale = float(g.get("kin_skate_scale", 0.01))
        self.kin_foot_ids = tuple(g.get("kin_foot_ids", [7, 8, 10, 11]))
        self.score_scale = float(g.get("score_scale", 0.15))
        self.joint_range = g.get("joint_range", None) # None = all 52 joints; [0,22] = body joints only
        if self.joint_range is not None:
            self.joint_range = tuple(self.joint_range)
        sim = g.get("sim", {})
        self.sim_enabled = bool(sim.get("enabled", True))
        self._phc_reward = None
        if self.sim_enabled:
            from ..reward_model.simulation.phc_tracking_reward import PHCTrackingReward
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            _default_wd = ("/dev/shm/grpo_phc_reward_cache" if os.path.isdir("/dev/shm")
                           else "./grpo_phc_reward_cache")
            _did = sim.get("device_id", -1)
            _device_id = local_rank if (int(_did) < 0) else int(_did)
            _run_tag = os.path.basename(str(self.exp).rstrip(os.sep))[:48]
            self._phc_reward = PHCTrackingReward(
                phc_root=sim.get("phc_root") or None,  # None -> repo default (PHC+ submodule)
                phc_python=sim.get("phc_python") or sys.executable,
                max_frames=int(sim.get("max_frames", 180)),
                num_envs=int(sim.get("num_envs", 64)),
                device_id=_device_id,
                work_dir=sim.get("work_dir", None) or f"{_default_wd}_{_run_tag}_rank{local_rank}",
                timeout_s=int(sim.get("batch_timeout", 600)),
                idle_timeout_s=int(sim.get("idle_timeout", 1800)),
                start_stagger_s=min(local_rank, 7) * 25,
                verbose=bool(sim.get("verbose", False)),
            )
        max_timesteps = int(g.get("max_timesteps", self.grpo_sampling_steps - 2))
        self.grpo_states = GRPOTrainingStates(
            iters_per_group=int(g.get("iters_per_group", 25)),
            group_size=int(g.get("group_size", 4)),
            max_timesteps=max_timesteps,
            cur_timestep=0,
            cur_iter_in_group=0,
            sample_strategy=g.get("sample_strategy", "progressive"),
            prog_overlap=bool(g.get("prog_overlap", True)),
            prog_overlap_step=int(g.get("prog_overlap_step", 1)),
            max_iters_per_group=g.get("max_iters_per_group", None),
            min_iters_per_group=g.get("min_iters_per_group", None),
            roll_back=bool(g.get("roll_back", True)),
        )
        self.logger.info(
            f"[V2MGRPOTrainer] num_generations={self.num_generations}, "
            f"sampling_steps={self.grpo_sampling_steps}, group_size={self.grpo_states.group_size}, "
            f"eta={self.eta}, mpjpe_w={self.mpjpe_weight}, tracking_w={self.tracking_weight}, "
            f"kin_w={self.kinematics_weight}, "
            f"sim={'on(max_frames=%d)' % self._phc_reward.max_frames if self.sim_enabled else 'off'}, "
            f"ref_kl={self.ref_kl_coeff}, inner_updates={self.inner_updates}, "
            f"clip_range={self.clip_range}, adv_std_floor={self.adv_std_floor}, "
            f"manual_grad_sync={self.manual_grad_sync}"
        )

        self.save_every_steps = int(g.get("save_every_steps", 250))
        self.ckpt_keep_recent = int(g.get("ckpt_keep_recent", 2))
        self.ckpt_keep_every = int(g.get("ckpt_keep_every", 500))
        self.probe_every_steps = int(g.get("probe_every_steps", 250))
        self.probe_samples = int(g.get("probe_samples", 32))
        self.timing_every = int(g.get("timing_every", 50))
        self._timing_on = False
        self._timings = {}
        self._grpo_state_synced = False

        self._ref_model = None

    def _get_ref_model(self, device):
        if self._ref_model is not None:
            return self._ref_model
        from ..core.framework.loaders import load_object

        ckpt_path = self.config.get("load_from_checkpoint", None)
        assert ckpt_path is not None, "ref_kl_coeff>0 requires load_from_checkpoint (used as the reference model)"

        ref = load_object(
            self.config["train_pipeline"],
            self.config["train_pipeline_args"],
            network_module=self.config["network_module"],
            network_module_args=self.config["network_module_args"],
        )
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        clean = {}
        for k, v in state_dict.items():
            nk = k
            for prefix in ("module.", "model.", "_forward_module."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            clean[nk] = v
        for key in ("mean", "std"):
            if key in clean and clean[key].ndim == 2:
                clean[key] = clean[key].squeeze(0)
        current = ref.state_dict()
        filtered = {k: v for k, v in clean.items() if k in current and current[k].shape == v.shape}
        ref.load_state_dict(filtered, strict=False)
        ref.to(device)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        self._ref_model = ref
        self.logger.info(
            f"[V2MGRPOTrainer] reference model built for ref-KL "
            f"(ref_kl_coeff={self.ref_kl_coeff}, from {ckpt_path})"
        )
        return self._ref_model

    def train_dataloader(self):
        from ..datasets.v2m_generation.grpo_gt_dataset import grpo_gt_collate

        train_iterations = self.config["train"].get("train_iterations", None)
        if train_iterations is not None and hasattr(self.dataset, "set_train_iterations"):
            num_processes = self.accelerator.num_processes
            self.dataset.set_train_iterations(
                int(train_iterations) * num_processes * self.batch_size
            )
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config["train"].get("num_workers", 2),
            drop_last=True,
            collate_fn=grpo_gt_collate,
            pin_memory=self.config["train"].get("pin_memory", False),
        )

    def val_dataloader(self, use_distributed=None):
        from ..datasets.v2m_generation.grpo_gt_dataset import grpo_gt_collate

        return DataLoader(
            self.val_dataset,
            batch_size=self.config["train"].get("batch_size_val", 1),
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn=grpo_gt_collate,
            pin_memory=False,
        )

    def _mpjpe_rewards(self, output: Dict[str, torch.Tensor], gt_joints, length: int):
        rot6d = output["rot6d"]          # (G,T,52,6)
        trans = output["trans"]          # (G,T,3)
        shapes = output["shapes"]        # (G,1,16)
        G = rot6d.shape[0]
        device = rot6d.device
        body_model = self.model.body_model
        try:
            return self._mpjpe_rewards_batched(body_model, rot6d, trans, shapes,
                                               gt_joints, length)
        except Exception as e:
            if not getattr(self, "_mpjpe_batch_err_logged", False):
                self.logger.warning(
                    f"[V2MGRPOTrainer] batched MPJPE failed ({e!r}), falling back to per-generation loop")
                self._mpjpe_batch_err_logged = True

        gt_np = gt_joints[:length].detach().cpu().numpy()
        rew_list, mpjpe_list = [], []
        for gi in range(G):
            single = {"rot6d": rot6d[gi, :length], "shapes": shapes[gi], "trans": trans[gi, :length]}
            try:
                m = _mpjpe.compute_mpjpe_reward_from_params(
                    body_model, single, gt_np, length=length,
                    joint_range=self.joint_range, score_scale=self.score_scale,
                )
                rew_list.append(m["reward"])
                mpjpe_list.append(m["mpjpe_m"])
            except Exception as e:
                if not getattr(self, "_mpjpe_err_logged", False):
                    self.logger.warning(
                        f"[V2MGRPOTrainer] MPJPE reward failed (gen {gi}): {repr(e)}\n"
                        + traceback.format_exc()
                    )
                    self._mpjpe_err_logged = True
                rew_list.append(0.0)
                mpjpe_list.append(1.0)
        return (torch.tensor(rew_list, device=device, dtype=torch.float32),
                np.array(mpjpe_list))

    def _mpjpe_rewards_batched(self, body_model, rot6d, trans, shapes, gt_joints,
                               length: int):
        G = rot6d.shape[0]
        T = min(rot6d.shape[1], length)
        dev = rot6d.device
        p = next(body_model.parameters(), None) if hasattr(body_model, "parameters") else None
        if p is not None:
            dev = p.device
        r6 = rot6d[:, :T].reshape(G * T, 52, 6).to(device=dev, dtype=torch.float32)
        tr = trans[:, :T].reshape(G * T, 3).to(device=dev, dtype=torch.float32)
        sh = shapes.to(device=dev, dtype=torch.float32)
        if sh.ndim == 3:
            sh = sh.expand(-1, T, -1) if sh.shape[1] == 1 else sh[:, :T]
        else:
            sh = sh.unsqueeze(1).expand(-1, T, -1)
        sh = sh.reshape(G * T, -1)
        with torch.no_grad():
            out = body_model({"rot6d": r6, "shapes": sh, "trans": tr})
        pred = out["keypoints3d"].reshape(G, T, -1, 3)[:, :, :52]
        gt = gt_joints[:T].to(device=dev, dtype=torch.float32).unsqueeze(0)
        if self.joint_range is not None:
            lo, hi = self.joint_range
            pred, gt = pred[:, :, lo:hi], gt[:, :, lo:hi]
        mpjpe = torch.linalg.norm(pred - gt, dim=-1).reshape(G, -1).mean(dim=1)
        reward = torch.exp(-mpjpe / self.score_scale)
        return (reward.to(device=rot6d.device, dtype=torch.float32),
                mpjpe.detach().float().cpu().numpy())

    def _kinematic_rewards(self, output: Dict[str, torch.Tensor], length: int):
        rot6d = output["rot6d"]          # (G,T,52,6)
        trans = output["trans"]
        shapes = output["shapes"]
        G = rot6d.shape[0]
        T = min(rot6d.shape[1], length)
        dev = rot6d.device
        body_model = self.model.body_model
        p = next(body_model.parameters(), None) if hasattr(body_model, "parameters") else None
        if p is not None:
            dev = p.device
        r6 = rot6d[:, :T].reshape(G * T, 52, 6).to(device=dev, dtype=torch.float32)
        tr = trans[:, :T].reshape(G * T, 3).to(device=dev, dtype=torch.float32)
        sh = shapes.to(device=dev, dtype=torch.float32)
        if sh.ndim == 3:
            sh = sh.expand(-1, T, -1) if sh.shape[1] == 1 else sh[:, :T]
        else:
            sh = sh.unsqueeze(1).expand(-1, T, -1)
        sh = sh.reshape(G * T, -1)
        with torch.no_grad():
            out = body_model({"rot6d": r6, "shapes": sh, "trans": tr})
        j3d = out["keypoints3d"].reshape(G, T, -1, 3)[:, :, :52]   # y-up

        if T >= 3:
            acc = (j3d[:, 2:] - 2 * j3d[:, 1:-1] + j3d[:, :-2]).norm(dim=-1)
            accel_m = acc.mean(dim=(1, 2))
        else:
            accel_m = torch.zeros(G, device=dev)

        feet = j3d[:, :, list(self.kin_foot_ids), :]
        if T >= 2:
            foot_y = feet[..., 1]
            ground = foot_y.flatten(1).quantile(0.05, dim=1)
            contact = foot_y < (ground.unsqueeze(-1).unsqueeze(-1) + 0.05)
            disp = (feet[:, 1:, :, :2] - feet[:, :-1, :, :2]).norm(dim=-1)
            contact_f = contact[:, 1:] & contact[:, :-1]
            denom = contact_f.sum(dim=(1, 2)).clamp_min(1.0)
            skate_m = (disp * contact_f.float()).sum(dim=(1, 2)) / denom
        else:
            skate_m = torch.zeros(G, device=dev)

        reward = (0.5 * torch.exp(-accel_m / self.kin_acc_scale)
                  + 0.5 * torch.exp(-skate_m / self.kin_skate_scale))
        return (reward.to(device=rot6d.device, dtype=torch.float32),
                accel_m.detach().float().cpu().numpy(),
                skate_m.detach().float().cpu().numpy())

    def _rollout_motions(self, output: Dict[str, torch.Tensor], length: int) -> List[dict]:
        rot6d = output["rot6d"] # (G,T_pad,52,6), T_pad = max_len
        trans = output["trans"]      # (G,T_pad,3)
        shapes = output["shapes"]    # (G,1,16)
        G = rot6d.shape[0]
        T = min(rot6d.shape[1], length)
        r6 = rot6d[:, :T]
        rotmats = rot6d_to_rotation_matrix(r6.reshape(G * T * 52, 6))
        poses_aa = rotation_matrix_to_angle_axis(rotmats).reshape(G, T, 52, 3)
        motions = []
        for gi in range(G):
            motions.append({
                "poses": poses_aa[gi].detach().float().cpu().numpy(),
                "trans": trans[gi, :T].detach().float().cpu().numpy(),
                "betas": shapes[gi].detach().float().cpu().numpy().reshape(-1),
            })
        return motions

    def _probe_validation(self, accelerator, model):
        pipeline = accelerator.unwrap_model(model)
        was_training = model.training
        model.eval()
        device = accelerator.device

        ds = self.val_dataset
        n_ds = max(len(ds), 1)
        n_total = min(self.probe_samples, n_ds)
        world = accelerator.num_processes
        rank = accelerator.process_index
        my_idxs = [(rank + world * k) % n_total
                   for k in range((n_total + world - 1) // world)]

        rs, ms, ts, ks = [], [], [], []
        for idx in my_idxs:
            s = ds[idx]
            feat = {k: v.unsqueeze(0).to(device) for k, v in s["inputs"]["feature"].items()}
            L = int(s["length"])
            gen = torch.Generator(device="cpu").manual_seed(1234 + idx)
            init_noise = torch.randn(
                (1, feat["feature"].shape[1], pipeline.latent_dim),
                generator=gen, dtype=torch.float32).to(device=device,
                                                       dtype=feat["feature"].dtype)
            mpjpe_r, mpjpe_m, track_r, kin_r = 0.0, 1.0, 0.0, 0.0
            try:
                with torch.no_grad():
                    sample = pipeline.grpo_sample(
                        feature=feat, length=L, num_generations=1,
                        sampling_steps=self.grpo_sampling_steps, eta=self.eta,
                        timesteps_train=[],
                        init_noise=init_noise, cfg_scale=self.cfg_scale,
                    )
                r, mv = self._mpjpe_rewards(
                    sample["output"], s["reward_gt"]["gt_joints"], L)
                mpjpe_r, mpjpe_m = float(r.mean()), float(np.mean(mv))
                if self.sim_enabled:
                    res = self._phc_reward.score_batch(
                        self._rollout_motions(sample["output"], L))
                    track_r = float(np.mean([x["tracking_score"] for x in res]))
                if self.kinematics_weight > 0:
                    kr, _, _ = self._kinematic_rewards(sample["output"], L)
                    kin_r = float(kr.mean())
            except Exception as e:
                if not getattr(self, "_probe_err_logged", False):
                    self.logger.warning(f"[V2MGRPOTrainer] probe failed: {e!r}")
                    self._probe_err_logged = True
            rs.append(mpjpe_r)
            ms.append(mpjpe_m)
            ts.append(track_r)
            ks.append(kin_r)

        mpjpe_r = float(np.mean(rs)) if rs else 0.0
        mpjpe_m = float(np.mean(ms)) if ms else 1.0
        track_r = float(np.mean(ts)) if ts else 0.0
        kin_r = float(np.mean(ks)) if ks else 0.0

        with torch.no_grad():
            ss = torch.zeros((), device=device, dtype=torch.float64)
            for p in pipeline.parameters():
                ss += p.detach().double().pow(2).sum()

        stat = torch.tensor([[mpjpe_r, track_r, mpjpe_m, float(ss.item()), kin_r]],
                            device=device, dtype=torch.float64)
        gathered = accelerator.gather(stat)

        if was_training:
            model.train()

        if accelerator.is_main_process:
            m = gathered.mean(0)
            chk = gathered[:, 3]
            spread = float((chk.max() - chk.min()) / chk.abs().mean().clamp_min(1e-12))
            metrics = {"mpjpe_reward": float(m[0]), "tracking": float(m[1]),
                       "mpjpe_cm": float(m[2]) * 100.0,
                       "param_spread": spread,
                       "kin_reward": float(m[4])}
            for k, v in metrics.items():
                self.writer.add_scalar(f"probe/{k}", v, self.global_step)
            self.logger.info(
                f"[V2MGRPOTrainer] [probe@{self.global_step}] "
                f"mpjpe_reward={metrics['mpjpe_reward']:.4f} "
                f"({metrics['mpjpe_cm']:.2f}cm)  "
                f"tracking={metrics['tracking']:.4f}  "
                f"kin={metrics['kin_reward']:.4f}  "
                f"param_spread={spread:.2e}"
                f"{' <== params differ across ranks!' if spread > 1e-6 else ''} "
                f"(n={gathered.shape[0]} fixed samples)")

    def _group_advantage(self, rewards: torch.Tensor) -> torch.Tensor:
        if rewards.numel() < 2:
            return torch.zeros_like(rewards)
        std = rewards.std()
        if (not torch.isfinite(std)) or float(std) < self.adv_std_floor:
            return torch.zeros_like(rewards)
        return (rewards - rewards.mean()) / (std + 1e-8)

    def _sync_grads_manually(self, accelerator, model):
        if not self.manual_grad_sync:
            return
        if accelerator.num_processes <= 1:
            return
        if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
            return
        import torch.distributed as dist
        if not (dist.is_available() and dist.is_initialized()):
            return
        world = float(accelerator.num_processes)
        params = [p for p in accelerator.unwrap_model(model).parameters() if p.requires_grad]
        mask = torch.tensor([1.0 if p.grad is not None else 0.0 for p in params],
                            device=accelerator.device)
        dist.all_reduce(mask, op=dist.ReduceOp.MAX)
        if not getattr(self, "_grad_sync_logged", False):
            self._grad_sync_logged = True
            self.logger.info(
                f"[V2MGRPOTrainer] manual grad all-reduce active: "
                f"world={int(world)}, params_with_grad={int(mask.sum().item())}/{len(params)}")
        grads = []
        for p, has_grad in zip(params, mask.tolist()):
            if has_grad <= 0:
                continue
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            grads.append(p.grad)

        bucket, cnt = [], 0
        for g in grads:
            if bucket and (g.dtype != bucket[0].dtype
                           or cnt + g.numel() > self.grad_bucket_numel):
                self._allreduce_bucket(bucket, world, self.grad_sync_dtype)
                bucket, cnt = [], 0
            bucket.append(g)
            cnt += g.numel()
        if bucket:
            self._allreduce_bucket(bucket, world, self.grad_sync_dtype)

    @staticmethod
    def _allreduce_bucket(grads: List[torch.Tensor], world: float,
                          comm_dtype: str = "fp32"):
        import torch.distributed as dist

        if len(grads) == 1 and comm_dtype != "bf16":
            dist.all_reduce(grads[0], op=dist.ReduceOp.SUM)
            grads[0].div_(world)
            return
        flat = torch.cat([g.reshape(-1) for g in grads])
        if comm_dtype == "bf16" and flat.dtype == torch.float32:
            payload = (flat / world).to(torch.bfloat16)
            dist.all_reduce(payload, op=dist.ReduceOp.SUM)
            flat = payload.to(flat.dtype)
        else:
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat.div_(world)
        off = 0
        for g in grads:
            n = g.numel()
            g.copy_(flat[off:off + n].view_as(g))
            off += n

    def _save_step_snapshot(self, accelerator, model, step: int):
        import shutil
        import threading

        t0 = time.time()
        model_copy = accelerator.unwrap_model(model)
        cpu_sd = {k.replace("._orig_mod.", "."): v.detach().to("cpu", copy=True)
                  for k, v in model_copy.state_dict().items()}
        payload = {"model_state_dict": cpu_sd, "epoch": 0, "global_step": step}

        stage_dir = "/dev/shm/grpo_ckpt_stage"
        try:
            os.makedirs(stage_dir, exist_ok=True)
            stage_path = os.path.join(stage_dir, f"step{step}.ckpt")
            torch.save(payload, stage_path)
        except OSError as e:
            self.logger.warning(f"[V2MGRPOTrainer] /dev/shm staging FAILED ({e}), saving directly to the output dir")
            stage_path = None
        dt_stage = time.time() - t0

        dst = os.path.join(self.exp, f"step{step}.ckpt")
        exp_dir, logger = self.exp, self.logger
        keep_recent, keep_every = self.ckpt_keep_recent, self.ckpt_keep_every

        def _flush_to_output_dir():
            try:
                tmp = dst + ".tmp"
                if stage_path:
                    shutil.move(stage_path, tmp)
                else:
                    torch.save(payload, tmp)
                os.replace(tmp, dst)
                latest = os.path.join(exp_dir, "latest_step.ckpt")
                if os.path.lexists(latest):
                    os.remove(latest)
                os.symlink(f"step{step}.ckpt", latest)
                import glob as _glob
                olds = sorted(_glob.glob(os.path.join(exp_dir, "step*.ckpt")),
                              key=lambda p: int(p.split("step")[-1].split(".")[0]))
                for p in olds[:-keep_recent]:
                    s = int(p.split("step")[-1].split(".")[0])
                    if keep_every > 0 and s % keep_every == 0:
                        continue
                    os.remove(p)
                logger.info(f"[V2MGRPOTrainer] step ckpt flushed to output dir: {dst}")
            except Exception as e:
                logger.warning(f"[V2MGRPOTrainer] FAILED to flush step{step} checkpoint to the output dir: {e!r}")

        prev = getattr(self, "_flush_thread", None)
        if prev is not None and prev.is_alive():
            prev.join(timeout=300)
        self._flush_thread = threading.Thread(target=_flush_to_output_dir, daemon=True)
        self._flush_thread.start()
        self.logger.info(
            f"[V2MGRPOTrainer] step{step} checkpoint staged ({dt_stage:.1f}s, "
            f"training resumed), flushing to the output dir in background")

    def _tick(self, name: str):
        if not getattr(self, "_timing_on", False):
            return
        torch.cuda.synchronize()
        now = time.time()
        self._timings[name] = now - self._tick_t0
        self._tick_t0 = now

    def training_step(self, accelerator, model, optimizer, batch, scheduler):
        pipeline = accelerator.unwrap_model(model)
        device = accelerator.device
        batch = self.batch_to_device(batch, accelerator)

        self._timing_on = (self.timing_every > 0
                           and self.global_step % self.timing_every == 0)
        if self._timing_on:
            self._timings = {}
            torch.cuda.synchronize()
            self._tick_t0 = time.time()

        if not self._grpo_state_synced:
            for _ in range(self.global_step):
                self.grpo_states.update_iteration()
            self._grpo_state_synced = True
            if self.global_step > 0:
                self.logger.info(
                    f"[V2MGRPOTrainer] grpo window state re-synced to "
                    f"cur_timestep={self.grpo_states.cur_timestep} "
                    f"(from {self.global_step} steps)"
                )

        timesteps_train = self.grpo_states.get_current_timesteps()

        B = batch["length"].shape[0]
        feature_all = batch["inputs"]["feature"]
        gt_joints_all = batch["reward_gt"]["gt_joints"]   # (B,T,52,3)
        lengths = batch["length"]

        valid_steps_all = [i for i in timesteps_train if 0 <= i < self.grpo_sampling_steps]

        rollouts = []
        all_sim_motions = []
        model.eval()
        with torch.no_grad():
            for bi in range(B):
                length = int(lengths[bi].item())
                feat_single = {k: v[bi:bi + 1] for k, v in feature_all.items()}

                init_noise = None
                if self.init_same_noise:
                    init_noise = torch.randn(
                        (1, feat_single["feature"].shape[1], pipeline.latent_dim),
                        device=device, dtype=feat_single["feature"].dtype,
                    )
                sample = pipeline.grpo_sample(
                    feature=feat_single,
                    length=length,
                    num_generations=self.num_generations,
                    sampling_steps=self.grpo_sampling_steps,
                    eta=self.eta,
                    timesteps_train=timesteps_train,
                    init_noise=init_noise,
                    cfg_scale=self.cfg_scale,
                )
                if not valid_steps_all:
                    continue
                n_lat = sample["all_latents"].shape[1]
                keep = sorted(set(valid_steps_all + [max(valid_steps_all) + 1]))
                latents_w = {i: sample["all_latents"][:, i].detach()
                             for i in keep if i < n_lat}
                old_lp_w = {i: sample["all_log_probs"][:, i].detach() for i in valid_steps_all}
                rollouts.append(dict(
                    bi=bi, length=length, output=sample["output"],
                    conditioning=sample["conditioning"],
                    latents_w=latents_w, old_lp_w=old_lp_w,
                ))
                if self.sim_enabled:
                    all_sim_motions.extend(self._rollout_motions(sample["output"], length))
        self._tick("rollout")

        if not rollouts:
            self.grpo_states.update_iteration()
            return {
                "loss": 0.0, "loss_dict": {"reward_mpjpe": 0.0, "reward_tracking": 0.0,
                                           "cur_timestep": float(self.grpo_states.cur_timestep)},
                "loss_dict_nosync": {}, "avg_loss": 0.0,
                "lr": scheduler.get_last_lr()[0], "batch_device": batch,
                "tensor_results": {"index": batch.get("index", None)}, "grad_norm": None,
            }

        # ================= Reward =================
        mpjpe_rews, mpjpe_vals = [], []
        for ro in rollouts:
            r, v = self._mpjpe_rewards(ro["output"], gt_joints_all[ro["bi"]], ro["length"])
            mpjpe_rews.append(r)
            mpjpe_vals.append(v)
        self._tick("mpjpe")

        kin_rews = None
        if self.kinematics_weight > 0:
            kin_rews = []
            for ro in rollouts:
                r, _, _ = self._kinematic_rewards(ro["output"], ro["length"])
                kin_rews.append(r)
        self._tick("kin")

        track_rews = []
        if self.sim_enabled:
            phc_results = self._phc_reward.score_batch(all_sim_motions)
            idx = 0
            for ro in rollouts:
                G = ro["output"]["rot6d"].shape[0]
                scores = torch.tensor(
                    [phc_results[idx + g]["tracking_score"] for g in range(G)],
                    device=device, dtype=torch.float32)
                track_rews.append(scores)
                idx += G
        else:
            for ro in rollouts:
                track_rews.append(torch.zeros(ro["output"]["rot6d"].shape[0], device=device))
        self._tick("phc")

        log_mpjpe, log_track, log_kin, log_clipfrac, log_kl, log_refkl = [], [], [], [], [], []
        log_ratio_dev, log_loss = [], []

        model.train()
        ref_model = self._get_ref_model(device) if self.ref_kl_coeff > 0 else None

        advs = []
        for i, (ro, mpjpe_r, track_r) in enumerate(zip(rollouts, mpjpe_rews, track_rews)):
            adv = (
                self.mpjpe_weight * self._group_advantage(mpjpe_r)
                + self.tracking_weight * self._group_advantage(track_r)
            )
            if kin_rews is not None:
                adv = adv + self.kinematics_weight * self._group_advantage(kin_rews[i])
                log_kin.append(float(kin_rews[i].mean()))
            advs.append(torch.clamp(adv, -self.adv_clip_max, self.adv_clip_max))
            log_mpjpe.append(float(mpjpe_r.mean()))
            log_track.append(float(track_r.mean()))

        opt_steps = [i for i in valid_steps_all if (i + 1) in rollouts[0]["latents_w"]]
        K = max(1, min(self.inner_updates, len(opt_steps)))
        if K > 1:
            rng = np.random.default_rng(int(self.global_step))
            order = list(rng.permutation(np.array(opt_steps)))
            chunks = [list(map(int, c)) for c in np.array_split(np.array(order), K)]
        else:
            chunks = [list(opt_steps)]

        grad_norm = None
        for ci, chunk in enumerate(chunks):
            if not chunk:
                continue
            optimizer.zero_grad(set_to_none=True)
            total_loss = torch.zeros((), device=device)
            for ro, adv in zip(rollouts, advs):
                sample_loss = torch.zeros((), device=device)
                for i in chunk:
                    latents_i = ro["latents_w"][i]
                    prev_i = ro["latents_w"][i + 1]
                    if ref_model is not None:
                        new_log_prob, v_cur = pipeline.grpo_one_step(
                            latents=latents_i, prev_latents=prev_i, step_index=i,
                            sampling_steps=self.grpo_sampling_steps, eta=self.eta,
                            conditioning=ro["conditioning"], cfg_scale=self.cfg_scale,
                            return_velocity=True,
                        )
                    else:
                        new_log_prob = pipeline.grpo_one_step(
                            latents=latents_i, prev_latents=prev_i, step_index=i,
                            sampling_steps=self.grpo_sampling_steps, eta=self.eta,
                            conditioning=ro["conditioning"], cfg_scale=self.cfg_scale,
                        )
                    old_lp = ro["old_lp_w"][i]
                    ratio = torch.exp(new_log_prob - old_lp)
                    unclipped = -adv * ratio
                    clipped = -adv * torch.clamp(
                        ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                    policy_loss = torch.mean(torch.maximum(unclipped, clipped))
                    kl_loss = 0.5 * torch.mean((new_log_prob - old_lp) ** 2)
                    sample_loss = sample_loss + policy_loss + self.kl_coeff * kl_loss

                    if ref_model is not None:
                        with torch.no_grad():
                            _ref_lp, v_ref = ref_model.grpo_one_step(
                                latents=latents_i, prev_latents=prev_i, step_index=i,
                                sampling_steps=self.grpo_sampling_steps, eta=self.eta,
                                conditioning=ro["conditioning"], cfg_scale=self.cfg_scale,
                                return_velocity=True,
                            )
                        ref_reg = torch.mean((v_cur - v_ref) ** 2)
                        sample_loss = sample_loss + self.ref_kl_coeff * ref_reg
                        log_refkl.append(ref_reg.item())

                    if ci > 0:
                        log_clipfrac.append(
                            torch.mean((torch.abs(ratio - 1.0) > self.clip_range).float()).item()
                        )
                        log_ratio_dev.append(torch.mean(torch.abs(ratio - 1.0)).item())
                    log_kl.append(kl_loss.item())
                sample_loss = sample_loss / len(chunk)
                total_loss = total_loss + sample_loss

            total_loss = total_loss / max(len(rollouts), 1)
            accelerator.backward(total_loss)
            self._tick(f"fwd_bwd{ci}")
            self._sync_grads_manually(accelerator, model)
            self._tick(f"allreduce{ci}")
            if accelerator.sync_gradients:
                is_ds = getattr(accelerator.state, "deepspeed_plugin", None) is not None
                if is_ds and hasattr(model, "get_global_grad_norm"):
                    gn = model.get_global_grad_norm()
                    grad_norm = float(gn) if gn is not None else None
                else:
                    gn = accelerator.clip_grad_norm_(
                        model.parameters(),
                        max_norm=self.config["train"]["grad_clip"]["max_norm"],
                        norm_type=self.config["train"]["grad_clip"]["norm_type"],
                    )
                    grad_norm = float(gn) if gn is not None else None
            optimizer.step()
            log_loss.append(total_loss.item())
        self._tick("optim")

        if self._timing_on:
            fb = sum(v for k, v in self._timings.items() if k.startswith("fwd_bwd"))
            ar = sum(v for k, v in self._timings.items() if k.startswith("allreduce"))
            self.logger.info(
                "[V2MGRPOTrainer] [timing@%d] rollout=%.1fs mpjpe=%.1fs phc=%.1fs "
                "fwd_bwd=%.1fs allreduce=%.1fs optim=%.1fs | total=%.1fs" % (
                    self.global_step, self._timings.get("rollout", 0),
                    self._timings.get("mpjpe", 0), self._timings.get("phc", 0),
                    fb, ar, self._timings.get("optim", 0),
                    sum(self._timings.values())))
            if accelerator.is_main_process:
                for k, v in (("rollout", self._timings.get("rollout", 0)),
                             ("phc", self._timings.get("phc", 0)),
                             ("fwd_bwd", fb), ("allreduce", ar)):
                    self.writer.add_scalar(f"timing/{k}", v, self.global_step)

        if self.ema is not None:
            self.ema.step(accelerator.unwrap_model(model).parameters())
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        mean_loss = float(np.mean(log_loss)) if log_loss else 0.0

        self.grpo_states.update_iteration()

        if (self.probe_every_steps > 0 and accelerator.sync_gradients
                and self.global_step % self.probe_every_steps == 0):
            self._probe_validation(accelerator, model)

        if (self.save_every_steps > 0 and accelerator.sync_gradients
                and self.global_step > 0
                and (self.global_step + 1) % self.save_every_steps == 0):
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                self._save_step_snapshot(accelerator, model, self.global_step + 1)
            accelerator.wait_for_everyone()

        loss_dict = {
            "reward_mpjpe": float(np.mean(log_mpjpe)) if log_mpjpe else 0.0,
            "reward_tracking": float(np.mean(log_track)) if log_track else 0.0,
            "reward_kinematic": float(np.mean(log_kin)) if log_kin else 0.0,
            "clip_frac": float(np.mean(log_clipfrac)) if log_clipfrac else 0.0,
            "ratio_dev_ppm": float(np.mean(log_ratio_dev)) * 1e6 if log_ratio_dev else 0.0,
            "kl": float(np.mean(log_kl)) if log_kl else 0.0,
            "ref_reg": float(np.mean(log_refkl)) if log_refkl else 0.0,
            "n_updates": float(len([c for c in chunks if c])),
            "cur_timestep": float(self.grpo_states.cur_timestep),
        }
        if self.sim_enabled and accelerator.is_main_process:
            for k, v in self._phc_reward.stats().items():
                loss_dict[f"phc_{k}"] = v
        elif self.sim_enabled and self._timing_on:
            st = self._phc_reward.stats()
            self.logger.info(
                f"[V2MGRPOTrainer] [phc@{self.global_step}] "
                f"motions_per_s={st.get('motions_per_s', 0)} "
                f"total_s={st.get('total_s', 0)} calls={st.get('calls', 0)}")

        if accelerator.is_main_process:
            for k, v in loss_dict.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"grpo/{k}", v, self.global_step)

        return {
            "loss": mean_loss,
            "loss_dict": loss_dict,
            "loss_dict_nosync": loss_dict,
            "avg_loss": mean_loss,
            "lr": scheduler.get_last_lr()[0],
            "batch_device": batch,
            "tensor_results": {"index": batch.get("index", None)},
            "grad_norm": grad_norm,
        }

    def validation(self, accelerator, model, output_dir="vis_train", vis=True, seeds=None):
        if not self.config.get("grpo", {}).get("do_validation", False):
            if accelerator.is_main_process:
                self.logger.info("[V2MGRPOTrainer] validation skipped (grpo.do_validation=False)")
            return {}
        pipeline = accelerator.unwrap_model(model)
        model.eval()
        loader = self.val_dataloader()
        mpjpe_scores, track_scores = [], []
        sim_motions, sim_meta = [], []
        for batch in loader:
            batch = self.batch_to_device(batch, accelerator)
            B = batch["length"].shape[0]
            for bi in range(B):
                length = int(batch["length"][bi].item())
                feat_single = {k: v[bi:bi + 1] for k, v in batch["inputs"]["feature"].items()}
                with torch.no_grad():
                    sample = pipeline.grpo_sample(
                        feature=feat_single, length=length,
                        num_generations=1, sampling_steps=self.grpo_sampling_steps,
                        eta=self.eta, timesteps_train=[], cfg_scale=self.cfg_scale,
                    )
                r, _ = self._mpjpe_rewards(
                    sample["output"], batch["reward_gt"]["gt_joints"][bi], length)
                mpjpe_scores.append(float(r.mean()))
                if self.sim_enabled:
                    sim_motions.extend(self._rollout_motions(sample["output"], length))
                    sim_meta.append(len(mpjpe_scores) - 1)
        if self.sim_enabled and sim_motions:
            phc = self._phc_reward.score_batch(sim_motions)
            for i, res in zip(sim_meta, phc):
                track_scores.append((i, res["tracking_score"]))
            track_scores = [s for _, s in sorted(track_scores)]
        if accelerator.is_main_process and mpjpe_scores:
            metrics = {
                "mpjpe_reward": float(np.mean(mpjpe_scores)),
                "tracking_score": float(np.mean(track_scores)) if track_scores else 0.0,
            }
            self.logger.info(
                f"[V2MGRPOTrainer] [{self.global_step}] "
                f"val/mpjpe_reward: {metrics['mpjpe_reward']:.4f}, "
                f"val/tracking_score: {metrics['tracking_score']:.4f}"
            )
            for k, v in metrics.items():
                self.writer.add_scalar(f"val/{k}", v, self.global_step)
            return metrics
        return {"mpjpe_reward": float(np.mean(mpjpe_scores)) if mpjpe_scores else 0.0}

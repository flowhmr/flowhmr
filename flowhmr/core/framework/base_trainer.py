import logging
import math
import os
import random
import time
from collections import defaultdict, deque
from fnmatch import fnmatch
from typing import Optional

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from torch.utils.data import DistributedSampler
from tqdm import tqdm

try:
    from diffusers.training_utils import EMAModel
except ImportError:
    EMAModel = None

from .loaders import load_object


def is_webdataset(dataset) -> bool:
    return hasattr(dataset, "is_webdataset") and dataset.is_webdataset


def seed_everything(seed: int = 42, rank: int = 0) -> None:
    final_seed = seed + rank
    torch.manual_seed(final_seed)
    # torch.cuda.manual_seed_all(final_seed)
    np.random.seed(final_seed)
    random.seed(final_seed)


def setup_logger(log_dir: str, local_rank: int, global_size: int) -> logging.Logger:
    """
    Setup logger for each local rank
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger(f"rank_{local_rank}")
    logger.setLevel(logging.INFO)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create file handler
    log_file = os.path.join(log_dir, f"rank_{local_rank}_of_{global_size}.log")
    if os.path.exists(log_file):
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    else:
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    # Only add console handler for rank 0
    if local_rank == 0:
        logger.addHandler(console_handler)
    # Prevent logging messages from being propagated to the root logger
    logger.propagate = False
    return logger


def silence_libraries():
    for lib in ["diffusers", "transformers", "urllib3", "PIL"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


class AccelerateTrainer:
    def set_exp(self, config):
        # Check if we're running in a distributed environment
        assert "exp" in config, "exp is not in config"
        self.accelerator.print(
            f">>> {config['exp']} rank: {self.rank}, num_processes: {self.accelerator.num_processes}"
        )

        if self.accelerator.num_processes > 1:
            # Get environment variables for distributed training
            num_processes = self.accelerator.num_processes

            # If running on multiple machines, modify the experiment directory name
            if num_processes > 1:
                # Append the total GPU count to the experiment name
                self.accelerator.print(
                    f"[{self.__class__.__name__}] Rank {self.rank} Running on {num_processes} processes"
                )
                config["exp"] = f"{config['exp'].strip(os.sep)}_gpus{num_processes}"
                self.accelerator.print(
                    f"[{self.__class__.__name__}] Rank {self.rank} Modified experiment directory to: {config['exp']}"
                )
        else:
            self.accelerator.print(config["exp"])

        self.exp = config["exp"]

    def copy_code(self):
        import shutil
        from datetime import datetime

        # Create timestamp for backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(self.exp, f"flowhmr_backup_{timestamp}")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        flowhmr_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

        try:
            if os.path.exists(flowhmr_dir):
                shutil.copytree(
                    flowhmr_dir, backup_dir, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
                )
                self.logger.info(f"[{self.__class__.__name__}] Backed up flowhmr directory to {backup_dir}")
            else:
                self.logger.warning(f"[{self.__class__.__name__}] flowhmr directory not found at {flowhmr_dir}")
        except Exception as e:
            self.logger.error(f"[{self.__class__.__name__}] Failed to backup flowhmr directory: {e}")

    def dump_config(self, config):
        if not os.path.exists(self.exp):
            os.makedirs(self.exp, exist_ok=True)
        with open(os.path.join(self.exp, "config.yml"), "w") as f:
            if hasattr(config, "to_dict"):
                config_dict = config.to_dict()
            elif hasattr(config, "_cfg_dict"):
                config_dict = dict(config._cfg_dict)
            else:
                config_dict = config
            try:
                yaml.safe_dump(config_dict, f, default_flow_style=False, indent=2)
            except Exception as e:
                self.logger.error(f"[{self.__class__.__name__}] Error dumping config: {e}")
                self.logger.error(f"[{self.__class__.__name__}] Config: {config}")
        return 0

    def set_tensorboard(self):
        if self.accelerator.is_main_process:
            from torch.utils.tensorboard.writer import SummaryWriter

            self.writer = SummaryWriter(log_dir=os.path.join(self.exp, "logs"))

    def __init__(self, config):
        self.phase = "train"
        self.config = config
        _acc_kwargs = {}
        try:
            from datetime import timedelta

            from accelerate import InitProcessGroupKwargs

            _timeout_s = int(os.environ.get("NCCL_COLLECTIVE_TIMEOUT_S", 1800))
            _acc_kwargs["kwargs_handlers"] = [
                InitProcessGroupKwargs(timeout=timedelta(seconds=_timeout_s))
            ]
        except Exception:
            pass
        self.accelerator = Accelerator(split_batches=False,
                                       step_scheduler_with_optimizer=False,
                                       **_acc_kwargs)
        if "use_same_seed_across_processes" in config and config["use_same_seed_across_processes"]:
            self.seed_everything(42)
        else:
            self.seed_everything(42 + self.accelerator.process_index)
        self.rank = self.accelerator.process_index

        if self.accelerator.is_main_process:
            silence_libraries()

        self.set_exp(config)
        self.logger = setup_logger(os.path.join(self.exp, "logger"), self.rank, self.accelerator.num_processes)
        if self.accelerator.is_main_process:
            self.dump_config(config)
            self.copy_code()
        self.make_train_dataset(config)
        self.make_val_dataset(config)
        self.model = load_object(
            config["train_pipeline"],
            config["train_pipeline_args"],
            network_module=config["network_module"],
            network_module_args=config["network_module_args"],
        )
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"[{self.__class__.__name__}] Model has {num_params/1e9:.2f}B parameters")
        self.global_step = 0
        self.skip_log_zero_epoch = config["train"].get("skip_log_zero_epoch", False)
        self.max_epoch = config["train"]["max_epoch"]
        self.batch_size = config["train"]["batch_size"]
        self.set_tensorboard()
        # abnormal loss tracking (main process only for logging)
        self._abn_window = deque(maxlen=self.config["train"].get("abnormal_loss_window", 2048))
        self._abn_sigma = float(self.config["train"].get("abnormal_loss_sigma", 3.0))
        self._abn_topk = int(self.config["train"].get("abnormal_topk", 0))
        self._abn_recent = deque(maxlen=self.config["train"].get("abnormal_record_size", 500))
        self._abn_log_fpath = os.path.join(self.exp, "abnormal_samples.log")

        # EMA model setting
        self.use_ema = bool(self.config["train"].get("use_ema", False))
        self.ema = None
        self.ema_validate = bool(self.config["train"].get("ema_validate", True))

    def seed_everything(self, seed=42):
        self.accelerator.print(f"[{self.__class__.__name__}] set seed to {seed}")
        # seed anything
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch_seed = torch.initial_seed()
        self.accelerator.print(f"[{self.__class__.__name__}] after set seed: {torch_seed}")

    def make_train_dataset(self, config):
        self.dataset = load_object(config["train_dataset"], config["train_dataset_args"])
        if self.accelerator.is_main_process:
            if is_webdataset(self.dataset):
                self.logger.info(f"[{self.__class__.__name__}] Loading WebDataset train dataset (streaming mode)")
            else:
                self.logger.info(f"[{self.__class__.__name__}] Loading train dataset with {len(self.dataset)} samples")
        return 0

    def make_val_dataset(self, config):
        is_main_process = self.accelerator.is_main_process
        if not config.get("val_dataset"):
            self.val_dataset = None
            if is_main_process:
                self.logger.info(f"[{self.__class__.__name__}] No val_dataset configured; validation disabled")
            return 0
        self.val_dataset = load_object(config["val_dataset"], config.get("val_dataset_args", {}))
        if is_main_process:
            self.logger.info(f"[{self.__class__.__name__}] Loading val dataset with {len(self.val_dataset)} samples")
        return 0

    def train_dataloader(self):
        self.logger.info(
            f"[{self.__class__.__name__}] Loading train dataloader with {len(self.dataset)} samples, batch_size={self.batch_size}, num_workers={self.config['train']['num_workers']}"
        )
        train_iterations = self.config["train"]["train_iterations"]
        num_processes = self.accelerator.num_processes
        self.dataset.set_train_iterations(train_iterations * num_processes * self.batch_size)
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config["train"]["num_workers"],
            drop_last=True,
            pin_memory=self.config["train"].get("pin_memory", False),
        )

    def val_dataloader(self, use_distributed=None):
        batch_size_val = (
            self.config["train"].get("batch_size_val", 1)
            if self.phase == "train"
            else self.config["inference"].get("batch_size", 1)
        )
        self.logger.info(
            f"[{self.__class__.__name__}] Loading val dataloader with {len(self.val_dataset)} samples, batch_size={batch_size_val}, num_workers={0}"
        )
        if is_webdataset(self.val_dataset) and hasattr(self.val_dataset, "get_dataloader"):
            return self.val_dataset.get_dataloader(
                batch_size=batch_size_val,
                num_workers=0,
                pin_memory=False,
                steps_per_epoch=None,
            )
        sampler = None
        if use_distributed is None:
            use_distributed = self.phase == "train"
        if use_distributed and self.accelerator.num_processes > 1:
            sampler = DistributedSampler(self.val_dataset, shuffle=False, drop_last=False)
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=batch_size_val,
            shuffle=False,
            sampler=sampler,
            num_workers=0,
            drop_last=False,
            pin_memory=False,
        )

    def get_optimizer(self):
        lr = self.config["train"]["lr"]
        self.logger.info(f"[{self.__class__.__name__}] lr: {lr}")

        opt_cfg = self.config["train"].get("optimizer", {})
        opt_type = opt_cfg.get("type", "AdamW")
        opt_params = {k: v for k, v in opt_cfg.items() if k != "type"}

        trainable = filter(lambda p: p.requires_grad, self.model.parameters())
        if opt_type == "AdamW":
            optimizer = torch.optim.AdamW(trainable, lr=lr, **opt_params)
        elif opt_type == "Adam":
            optimizer = torch.optim.Adam(trainable, lr=lr, **opt_params)
        elif opt_type == "SGD":
            optimizer = torch.optim.SGD(trainable, lr=lr, **opt_params)
        else:
            raise ValueError(f"Unsupported optimizer type: {opt_type}")
        self.logger.info(f"[{self.__class__.__name__}] optimizer: {opt_type}, params: {opt_params}")

        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=1)
        return optimizer, scheduler

    def set_sampler_epoch(self, train_dataloader, epoch):
        if is_webdataset(self.dataset):
            return

        if hasattr(train_dataloader, "sampler") and hasattr(train_dataloader.sampler, "set_epoch"):
            self.accelerator.print(f"[{self.__class__.__name__}] set sampler epoch")
            train_dataloader.sampler.set_epoch(epoch)
        elif hasattr(train_dataloader, "set_epoch"):
            # accelerate-wrapped dataloader: set epoch on the underlying sampler
            if self.accelerator.num_processes == 1:
                train_dataloader.set_epoch(epoch)
            else:
                sampler = train_dataloader.batch_sampler.batch_sampler.sampler
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)
                else:
                    self.accelerator.print(
                        f"[{self.__class__.__name__}] sampler {type(sampler).__name__} has no set_epoch, skipping"
                    )
        else:
            self.accelerator.print(f"[{self.__class__.__name__}] WARNING: cannot set sampler epoch")

    def _autodetect_resume(self) -> Optional[str]:
        import glob as _glob
        import re as _re

        def _scan(d):
            out = []
            for pat in ("step*.ckpt", "epoch*.ckpt"):
                for p in _glob.glob(os.path.join(d, pat)):
                    if p.endswith(".tmp") or os.path.islink(p):
                        continue
                    try:
                        if os.path.getsize(p) > (1 << 30):
                            out.append((os.path.getmtime(p), p))
                    except OSError:
                        continue
            return out

        cands = _scan(self.exp)
        if not cands:
            sibling = _re.sub(r"_gpus\d+$", "", self.exp) + "_gpus*"
            for d in _glob.glob(sibling):
                if os.path.abspath(d) != os.path.abspath(self.exp):
                    cands += _scan(d)
            if cands:
                self.logger.info(
                    f"[resume=auto] no checkpoint in {self.exp}, using sibling _gpus* dir")
        if not cands:
            self.logger.info(
                f"[resume=auto] no checkpoint found, training from scratch (exp={self.exp})")
            return None
        path = max(cands)[1]
        self.logger.info(f"[resume=auto] resuming from latest checkpoint: {path}")
        return path

    def fit(self, log_loss_interval: int = 100, resume: Optional[str] = None):
        train_dataloader = self.train_dataloader()
        trainables = self.config["train"].get("trainable_modules", None)
        if trainables:
            for _, p in self.model.named_parameters():
                p.requires_grad = False
            for name, p in self.model.named_parameters():
                parts = name.split(".")
                if any(
                    name.startswith(k) or fnmatch(name, k) or k in parts or any(fnmatch(seg, k) for seg in parts)
                    for k in trainables
                ):
                    p.requires_grad = True
            num_all = sum(1 for _ in self.model.parameters())
            num_train = sum(p.requires_grad for p in self.model.parameters())
            self.logger.info(f"Trainable params: {num_train}/{num_all} ({trainables})")

        optimizer, scheduler = self.get_optimizer()

        model = self.model.to(self.accelerator.device)
        resume_step = 0
        resume_epoch = 0
        if resume == "auto":
            resume = self._autodetect_resume()
        if resume is not None:
            if os.path.isdir(resume):
                ckpt_path = None
                base_dir = os.path.dirname(resume.rstrip(os.sep))
                bn = os.path.basename(resume.rstrip(os.sep))
                if bn.startswith("state_epoch"):
                    try:
                        ep = int(bn.replace("state_epoch", ""))
                        cand = os.path.join(base_dir, f"epoch{ep}.ckpt")
                        if os.path.isfile(cand):
                            ckpt_path = cand
                    except Exception:
                        pass
                elif bn.startswith("state_step"):
                    try:
                        sp = int(bn.replace("state_step", ""))
                        cand = os.path.join(base_dir, f"step{sp}.ckpt")
                        if os.path.isfile(cand):
                            ckpt_path = cand
                    except Exception:
                        pass
                # fallback: latest.ckpt
                if ckpt_path is None:
                    latest_ckpt = os.path.join(base_dir, "latest.ckpt")
                    if os.path.isfile(latest_ckpt):
                        ckpt_path = latest_ckpt
                if ckpt_path is None:
                    raise ValueError(f"Cannot find checkpoint file in {resume}")
                checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                try:
                    self.accelerator.load_state(resume)
                    self.logger.info(f"Loaded accelerator state from {resume}")
                except KeyError as exc:
                    # accelerate<=0.30 assumes every current rank has a saved RNG
                    # file and unconditionally reads override_attributes["step"].
                    # optimizer and scheduler have already been restored before this
                    # late KeyError; new ranks should simply keep their fresh RNG.
                    if exc.args == ("step",):
                        self.logger.warning(
                            "Accelerator state restored without per-rank RNG step "
                            f"(world-size change): {exc}. Fresh RNG will be used."
                        )
                    else:
                        raise
            elif os.path.isfile(resume):
                checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
                if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
                    try:
                        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                        self.logger.info("Successfully loaded optimizer state")
                    except Exception as e:
                        self.logger.warning(f"⚠️  Failed to load optimizer state: {e}")
                        self.logger.warning("⚠️  Will start with fresh optimizer state")
                if (
                    scheduler is not None
                    and "scheduler_state_dict" in checkpoint
                    and checkpoint["scheduler_state_dict"] is not None
                ):
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            else:
                raise ValueError(f"Invalid resume path: {resume}")

            assert isinstance(checkpoint, dict) and "model_state_dict" in checkpoint, f"Invalid checkpoint: {resume}"
            self.accelerator.unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
            if self.ema is not None and "ema_state_dict" in checkpoint and checkpoint["ema_state_dict"] is not None:
                self.ema.load_state_dict(checkpoint["ema_state_dict"])
            resume_epoch = checkpoint.get("epoch", 0)
            _iters = int(self.config["train"].get("train_iterations", 0) or 0)
            _gs = int(checkpoint.get("global_step", resume_epoch * _iters))
            if _iters > 0:
                resume_epoch = _gs // _iters
                if _gs % _iters != 0:
                    resume_step = _gs % _iters
            self.global_step = _gs
            model.global_iteration = self.global_step
            if scheduler is not None and _gs > 0:
                _sched_state = checkpoint.get("scheduler_state_dict", None)
                if _sched_state is not None:
                    scheduler.load_state_dict(_sched_state)
                    self.logger.info("Restored scheduler state from checkpoint")
                else:
                    for _ in range(_gs):
                        scheduler.step()
                    self.logger.info(
                        f"Scheduler fast-forwarded to step {_gs}, "
                        f"lr={scheduler.get_last_lr()[0]:.3e}")
            self.logger.info(f"Loaded complete checkpoint from {resume}, resuming from epoch {resume_epoch}, global_step {self.global_step}")
        elif self.config.get("load_from_checkpoint", None) is not None:
            assert os.path.isfile(
                self.config["load_from_checkpoint"]
            ), f"Checkpoint not found: {self.config['load_from_checkpoint']}, which should be a file"
            checkpoint = torch.load(self.config["load_from_checkpoint"], map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            clean_state_dict = {}
            for k, v in state_dict.items():
                nk = k
                for prefix in ("module.", "model.", "_forward_module."):
                    if nk.startswith(prefix):
                        nk = nk[len(prefix) :]
                clean_state_dict[nk] = v

            if "mean" in clean_state_dict and clean_state_dict["mean"].ndim == 2:
                clean_state_dict["mean"] = clean_state_dict["mean"].squeeze(0)
            if "std" in clean_state_dict and clean_state_dict["std"].ndim == 2:
                clean_state_dict["std"] = clean_state_dict["std"].squeeze(0)

            current = self.accelerator.unwrap_model(model).state_dict()
            filtered = {k: v for k, v in clean_state_dict.items() if k in current and current[k].shape == v.shape}

            missing = [k for k in current.keys() if k not in filtered]
            unexpected = [k for k in clean_state_dict.keys() if k not in current]
            self.logger.info(
                f"Loading weights: matched={len(filtered)}, missing={len(missing)}, unexpected={len(unexpected)}"
            )

            self.accelerator.unwrap_model(model).load_state_dict(filtered, strict=False)

            if self.ema is not None and isinstance(checkpoint, dict) and checkpoint.get("ema_state_dict") is not None:
                try:
                    self.ema.load_state_dict(checkpoint["ema_state_dict"])
                    self.logger.info("Loaded EMA weights")
                except Exception as e:
                    self.logger.warning(f"Failed to load EMA: {e}")

            self.logger.info(f"Loaded model weights from {self.config['load_from_checkpoint']}")

        if self.config.get("torch_compile", False):
            _net = getattr(model, "motion_transformer", None)
            if _net is not None:
                import torch._dynamo as torch_dynamo
                torch_dynamo.config.cache_size_limit = 64
                model.motion_transformer = torch.compile(_net, dynamic=True)
                self.logger.info(
                    f"[{self.__class__.__name__}] torch.compile enabled on motion_transformer (dynamic=True)"
                )

        # train_dataloader.batch_size = self.batch_size
        if is_webdataset(self.dataset) and self.accelerator.state.deepspeed_plugin is not None:
            self.accelerator.print(
                f"[{self.__class__.__name__}] set train_micro_batch_size_per_gpu to {self.batch_size}"
            )
            self.accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = int(
                self.batch_size
            )

        model, optimizer, train_dataloader, scheduler = self.accelerator.prepare(
            model, optimizer, train_dataloader, scheduler
        )
        if self.use_ema and self.ema is None:
            if EMAModel is None:
                raise ImportError("use_ema requires diffusers>=0.24")
            self.ema = EMAModel(
                self.accelerator.unwrap_model(model).parameters(),
                decay=self.config["train"]["ema_decay"],
                device=self.accelerator.device,
            )

        self.validation(self.accelerator, model)
        for epoch in range(resume_epoch, self.max_epoch):
            epoch_start_time = time.time()

            # Access the underlying model when using DDP
            if hasattr(model, "module"):
                model.module.set_epoch(epoch)
            else:
                model.set_epoch(epoch)

            self.set_sampler_epoch(train_dataloader, epoch)
            model.train()

            total_steps = self.config["train"].get("train_iterations", None)
            if total_steps is None:
                try:
                    total_steps = len(train_dataloader)
                except:
                    total_steps = None

            if resume and epoch == resume_epoch and resume_step != 0:
                activate_dataloader = self.accelerator.skip_first_batches(
                    train_dataloader,
                    resume_step * self.accelerator.gradient_accumulation_steps,
                )
                if total_steps is not None:
                    total_steps = max(1, total_steps - resume_step)
            else:
                activate_dataloader = train_dataloader

            end_data_time = time.time()
            get_data_time_list = []
            for batch_idx, batch in enumerate(activate_dataloader):
                get_data_time = time.time() - end_data_time
                get_data_time_list.append(get_data_time)
                ret = self.training_step(self.accelerator, model, optimizer, batch, scheduler)
                if batch_idx == 0:
                    batch_index = batch.get("index", None)
                    print(
                        f"[{self.__class__.__name__}] rank {self.accelerator.process_index}, "
                        f"epoch {epoch}, first batch indices: {batch_index}"
                    )
                if batch_idx % 100 == 0:
                    epoch_time = time.time() - epoch_start_time
                    log_string = (
                        f"[{self.__class__.__name__}] {epoch_time:.2f}s epoch: {epoch}, batch_idx: {batch_idx}/{total_steps}, loss: {ret['loss']:.4f}"
                    )
                    log_string += f", get_data_time: {sum(get_data_time_list) / len(get_data_time_list):.2f}s"
                    if "loss_dict_nosync" in ret:
                        for key, value in ret["loss_dict_nosync"].items():
                            log_string += f", {key}: {value:.4f}"
                    self.logger.info(log_string)
                if self.accelerator.is_main_process and (batch_idx + 1) % log_loss_interval == 0:
                    self.writer.add_scalar("train/loss", ret["loss"], self.global_step)
                    self.writer.add_scalar("train/lr", ret["lr"], self.global_step)
                    self.writer.add_scalar(
                        "train/get_data_time", sum(get_data_time_list) / len(get_data_time_list), self.global_step
                    )
                    get_data_time_list = []
                    if ret.get("grad_norm") is not None:
                        self.writer.add_scalar("train/grad_norm", ret["grad_norm"], self.global_step)
                    for key, values in ret["loss_dict"].items():
                        self.writer.add_scalar(f"train/loss_{key}", values, self.global_step)
                self.global_step += 1
                end_data_time = time.time()

            self.validation(self.accelerator, model)

            # Log epoch time
            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f"[{self.__class__.__name__}] Epoch {epoch} finished, epoch_time: {epoch_time:.2f}s ({epoch_time/60:.2f}min)"
            )
            if self.accelerator.is_main_process:
                self.writer.add_scalar("train/epoch_time", epoch_time, epoch)

            if (epoch + 1) % self.config.get("save_interval", 1) == 0:
                state_dir = os.path.join(self.exp, f"state_epoch{epoch+1}")
                self.accelerator.save_state(state_dir)
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    self.accelerator.print(f"saved accelerator state to {state_dir}")

                    epoch_save_path = os.path.join(self.exp, f"epoch{epoch+1}.ckpt")
                    model_copy = self.accelerator.unwrap_model(model)
                    _sd = {k.replace("._orig_mod.", "."): v for k, v in model_copy.state_dict().items()}
                    checkpoint = {
                        "model_state_dict": _sd,
                        "epoch": epoch + 1,
                        "global_step": self.global_step,
                    }
                    if self.ema is not None:
                        checkpoint["ema_state_dict"] = self.ema.state_dict()
                    torch.save(checkpoint, epoch_save_path)
                    self.accelerator.print(f"saved checkpoint to {epoch_save_path}")

                    latest_path = os.path.join(self.exp, "latest.ckpt")
                    if os.path.lexists(latest_path):
                        os.remove(latest_path)
                    os.symlink(f"epoch{epoch+1}.ckpt", latest_path)
                    self.accelerator.print(f"updated latest.ckpt symlink to epoch{epoch+1}.ckpt")

        last_epoch = self.max_epoch
        if last_epoch % self.config.get("save_interval", 1) != 0:
            if self.accelerator.is_main_process:
                final_save_path = os.path.join(self.exp, f"epoch{last_epoch}.ckpt")
                model_copy = self.accelerator.unwrap_model(model)
                _sd = {k.replace("._orig_mod.", "."): v for k, v in model_copy.state_dict().items()}
                checkpoint = {
                    "model_state_dict": _sd,
                    "epoch": last_epoch,
                    "global_step": self.global_step,
                }
                if self.ema is not None:
                    checkpoint["ema_state_dict"] = self.ema.state_dict()
                torch.save(checkpoint, final_save_path)
                self.accelerator.print(f"saved final checkpoint to {final_save_path}")

                latest_path = os.path.join(self.exp, "latest.ckpt")
                if os.path.lexists(latest_path):
                    os.remove(latest_path)
                os.symlink(f"epoch{last_epoch}.ckpt", latest_path)
                self.accelerator.print(f"updated latest.ckpt symlink to epoch{last_epoch}.ckpt")

        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()

        self.logger.info(f"[{self.__class__.__name__}] Training finished")

    def val(self, output_dir="vis_val", vis=True):
        model = self.model.to(self.accelerator.device)
        accelerator = self.accelerator
        self.validation(accelerator, model, output_dir=output_dir, vis=vis)

    def batch_to_device(self, batch, torch_device):
        batch_device = {}
        for key, val in batch.items():
            if torch.is_tensor(val):
                if hasattr(torch_device, "device"):
                    batch_device[key] = val.to(torch_device.device, non_blocking=True)
                else:
                    batch_device[key] = val.to(torch_device, non_blocking=True)
            elif isinstance(val, dict):
                batch_device[key] = self.batch_to_device(val, torch_device)
            else:
                batch_device[key] = val
        return batch_device

    def _track_abnormal_loss(self, outputs, batch):
        """Track abnormally high per-sample losses using a sliding window.

        Called on main process only. Writes to ``self._abn_log_fpath``.
        """
        tsr = outputs.get("tensor_results", {})
        per_sample = tsr.get("per_sample_loss", None)
        if per_sample is None:
            return

        idxs = batch.get("index", None)
        names_cpu = None
        meta = batch.get("data_meta", None)
        if isinstance(meta, dict):
            fnames = meta.get("caption_filename", None) or meta.get("input_filename", None)
            if isinstance(fnames, (list, tuple)):
                names_cpu = list(fnames)
            elif isinstance(fnames, str):
                names_cpu = [fnames]
            elif fnames is not None:
                names_cpu = [str(fnames)]

        if torch.is_tensor(per_sample):
            per_sample_cpu = per_sample.detach().cpu().float()
        else:
            per_sample_cpu = torch.as_tensor(per_sample, dtype=torch.float)
        idxs_cpu = None
        if idxs is not None:
            idxs_cpu = idxs.detach().cpu().tolist() if torch.is_tensor(idxs) else list(idxs)

        self._abn_window.extend(per_sample_cpu.tolist())
        assert self._abn_window.maxlen is not None, "abnormal_loss_window must be set"
        if len(self._abn_window) < min(max(16 * self.batch_size, 32), self._abn_window.maxlen):
            return

        mu = float(np.mean(list(self._abn_window)))
        sigma = float(np.std(list(self._abn_window))) + 1e-8
        thr = mu + self._abn_sigma * sigma

        per_list = per_sample_cpu.tolist()
        records = []
        for j, l in enumerate(per_list):
            if l >= thr:
                idx_val = idxs_cpu[j] if idxs_cpu is not None and j < len(idxs_cpu) else None
                name_val = names_cpu[j] if names_cpu is not None and j < len(names_cpu) else None
                records.append((idx_val, name_val, float(l)))
        if self._abn_topk and len(records) > self._abn_topk:
            records = sorted(records, key=lambda x: x[2], reverse=True)[: self._abn_topk]

        if records:
            with open(self._abn_log_fpath, "a", encoding="utf-8") as f:
                for idx_val, name_val, loss_val in records:
                    self._abn_recent.append(
                        (self.global_step, idx_val if idx_val is not None else name_val, loss_val)
                    )
                    if name_val is not None and idx_val is not None:
                        f.write(f"{self.global_step}\t{idx_val}\t{name_val}\t{loss_val:.6f}\n")
                    elif name_val is not None:
                        f.write(f"{self.global_step}\t{name_val}\t{loss_val:.6f}\n")
                    else:
                        f.write(f"{self.global_step}\t{idx_val}\t{loss_val:.6f}\n")

    def training_step(self, accelerator, model, optimizer, batch, scheduler):
        batch_device = self.batch_to_device(batch, accelerator)
        # Access the underlying model when using DDP
        with accelerator.autocast():
            if hasattr(model, "module"):
                outputs = model.module.forward_in_training(batch_device)
            else:
                outputs = model.forward_in_training(batch_device)

        loss = outputs["loss"]
        avg_loss = accelerator.gather(loss.detach().unsqueeze(0)).mean()

        # Synchronize loss_dict values across GPUs
        loss_dict = outputs.get("loss_dict", {})
        synchronized_loss_dict = {}
        nosynchronized_loss_dict = {}
        for key, value in loss_dict.items():
            if not torch.is_tensor(value):
                try:
                    value = torch.tensor(value, device=accelerator.device, dtype=torch.float32)
                except Exception:
                    synchronized_loss_dict[key] = value
                    continue
            value_flat1 = value.detach().reshape(-1)[:1]
            val_mean = accelerator.gather(value_flat1).mean()
            synchronized_loss_dict[key] = val_mean.item()
            nosynchronized_loss_dict[key] = value.item()

        # abnormal sample sliding logging (main process)
        if accelerator.is_main_process:
            self._track_abnormal_loss(outputs, batch)

        accelerator.backward(loss)
        global_norm = None
        is_ds = (
            "deepspeed" in str(type(model)).lower() or getattr(accelerator.state, "deepspeed_plugin", None) is not None
        )
        if accelerator.sync_gradients:
            if is_ds and hasattr(model, "get_global_grad_norm"):
                gn = model.get_global_grad_norm()
                global_norm = float(gn) if gn is not None else None
            else:
                gn = accelerator.clip_grad_norm_(
                    model.parameters(),
                    max_norm=self.config["train"]["grad_clip"]["max_norm"],
                    norm_type=self.config["train"]["grad_clip"]["norm_type"],
                )
                global_norm = float(gn) if gn is not None else None
        optimizer.step()
        if self.ema is not None:
            self.ema.step(self.accelerator.unwrap_model(model).parameters())
        scheduler.step()
        optimizer.zero_grad()

        return {
            "loss": loss.item(),
            "loss_dict_nosync": nosynchronized_loss_dict,
            "loss_dict": synchronized_loss_dict,
            "avg_loss": avg_loss.item(),
            "lr": scheduler.get_last_lr()[0],
            "batch_device": batch_device,
            "tensor_results": outputs.get("tensor_results", {}),
            "grad_norm": global_norm,
        }

    def validation(self, accelerator, model, output_dir="vis_train", vis=True):
        if self.val_dataset is None:
            return
        swap = False
        if self.ema is not None and self.ema_validate:
            self.ema.store(self.accelerator.unwrap_model(model).parameters())
            self.ema.copy_to(self.accelerator.unwrap_model(model).parameters())
            swap = True

        model.eval()
        unwrapped_model = accelerator.unwrap_model(model)
        global_size = self.accelerator.num_processes
        rank = self.accelerator.process_index
        dataloader = self.val_dataloader()
        collection = defaultdict(list)
        if accelerator.is_main_process:
            pbar = tqdm(dataloader, desc=f"val {rank}/{global_size}")
        else:
            pbar = dataloader
        for batch_idx, batch in enumerate(pbar):
            batch_device = self.batch_to_device(batch, accelerator)
            with torch.no_grad():
                output = unwrapped_model.validate(batch_device)
            for key, value in output["metrics"].items():
                collection[key].append(value)
            if isinstance(pbar, tqdm):
                pbar.set_postfix(**{k: v for k, v in output["metrics"].items()})
        accelerator.wait_for_everyone()
        if accelerator.is_main_process and (not self.skip_log_zero_epoch or self.global_step > 0):
            global_avg = {}
            for key, value in collection.items():
                avg_value = sum(value) / len(value)
                global_avg[key] = avg_value
                self.logger.info(f"[{self.__class__.__name__}] [{self.global_step}] val/{key}: {avg_value:.3f}")
                self.writer.add_scalar(f"val/{key}", avg_value, self.global_step)
            if global_avg:
                metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in global_avg.items())
                self.logger.info(f"[{self.__class__.__name__}] Validation step {self.global_step}: {metrics_str}")
        if self.ema is not None and swap:
            self.ema.restore(self.accelerator.unwrap_model(model).parameters())
        return collection

    def performance_test(self):
        start_time = time.time()
        test_data_times = 10
        for i in range(test_data_times):
            train_data = []
            for j in range(self.batch_size):
                data = self.dataset[j]
                train_data.append(data)
        end_time = time.time()
        self.logger.info(
            f"[{self.__class__.__name__}] Performance dataset loading test finished, time: {(end_time - start_time)/test_data_times:.2f}s"
        )
        # test dataloader
        start_time = time.time()
        train_dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config["train"]["num_workers"],
            drop_last=True,
            pin_memory=self.config["train"].get("pin_memory", False),
        )
        for batch_idx, batch in enumerate(train_dataloader):
            if batch_idx >= test_data_times:
                break
            self.batch_to_device(batch, self.accelerator)
        end_time = time.time()
        self.logger.info(
            f"[{self.__class__.__name__}] Performance dataloader test finished, time: {(end_time - start_time)/test_data_times:.2f}s"
        )


class SchedulerFactory:
    @staticmethod
    def create_scheduler(scheduler_type: str, optimizer, scheduler_config: dict, total_steps: Optional[int] = None):
        if scheduler_type == "fixed":
            return torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=1.0,
                total_iters=scheduler_config.get("total_iters", 1),
            )
        elif scheduler_type == "step":
            step_size = scheduler_config.get("step_size", 30)
            gamma = scheduler_config.get("gamma", 0.1)
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == "cosine_annealing":
            if total_steps is None:
                raise ValueError("cosine_annealing scheduler requires total_steps")

            T_max = scheduler_config.get("T_max", total_steps)
            eta_min = scheduler_config.get("eta_min", 0)
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


class DistributedSamplerTrainer(AccelerateTrainer):
    def get_train_sampler(self) -> torch.utils.data.Sampler:
        # NOTE: iters_per_epoch will be divided by accelerator, so we need to multiply num_processes manually here
        self.config["train_sampler_args"].update(
            {
                "samples_per_gpu": self.batch_size,
                "dataset": self.dataset,
                "iters_per_epoch": (
                    self.config["train"].get("train_iterations", 5000) * self.accelerator.num_processes
                ),
                "dataset_already_sharded": True,
            }
        )
        return load_object(
            self.config["train_sampler"],
            self.config["train_sampler_args"],
            global_size=self.accelerator.num_processes,
            local_rank=self.accelerator.process_index,
        )

    def train_dataloader(self):
        if is_webdataset(self.dataset):
            return self._train_dataloader_webdataset()
        else:
            return self._train_dataloader_standard()

    def _train_dataloader_webdataset(self):
        train_iterations = self.config["train"].get("train_iterations", 5000)
        num_workers = self.config["train"]["num_workers"]

        self.logger.info(
            f"[{self.__class__.__name__}] Loading WebDataset train dataloader, "
            f"batch_size={self.batch_size}, num_workers={num_workers}, "
            f"steps_per_epoch={train_iterations}"
        )

        if hasattr(self.dataset, "get_dataloader"):
            loader = self.dataset.get_dataloader(
                batch_size=self.batch_size,
                num_workers=num_workers,
                pin_memory=self.config["train"].get("pin_memory", False),
                steps_per_epoch=train_iterations,
            )
        else:
            import webdataset as wds

            loader = wds.WebLoader( # type: ignore
                self.dataset.dataset,
                batch_size=self.batch_size,
                num_workers=num_workers,
                pin_memory=self.config["train"].get("pin_memory", False),
            ).with_epoch(train_iterations)

        return loader

    def _train_dataloader_standard(self):
        self.logger.info(
            f"[{self.__class__.__name__}] Loading train dataloader with {len(self.dataset)} samples, batch_size={self.batch_size}, num_workers={self.config['train']['num_workers']}"
        )
        if self.config.get("train_sampler") is None:
            self.config["train_sampler"] = "flowhmr/core/framework/fixed_length_sampler.FixedLengthSampler"
            self.config["train_sampler_args"] = dict()
        train_sampler = self.get_train_sampler()
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.config["train"]["num_workers"],
            drop_last=True,
            sampler=train_sampler,
            pin_memory=self.config["train"].get("pin_memory", False),
        )

    def get_optimizer(self):
        lr = self.config["train"]["lr"]
        self.logger.info(f"[{self.__class__.__name__}] lr: {lr}")

        opt_cfg = self.config["train"].get("optimizer", {})
        opt_type = opt_cfg.get("type", "AdamW")
        opt_params = {k: v for k, v in opt_cfg.items() if k != "type"}

        trainable = filter(lambda p: p.requires_grad, self.model.parameters())
        if opt_type == "AdamW":
            optimizer = torch.optim.AdamW(trainable, lr=lr, **opt_params)
        elif opt_type == "Adam":
            optimizer = torch.optim.Adam(trainable, lr=lr, **opt_params)
        elif opt_type == "SGD":
            optimizer = torch.optim.SGD(trainable, lr=lr, **opt_params)
        else:
            raise ValueError(f"Unsupported optimizer type: {opt_type}")
        self.logger.info(f"[{self.__class__.__name__}] optimizer: {opt_type}, params: {opt_params}")

        scheduler_type = self.config["train"].get("scheduler", "fixed")
        scheduler_cfg = self.config["train"].get("scheduler_cfg", {})

        train_iterations = self.config["train"].get("train_iterations", 5000)
        total_steps = train_iterations * self.max_epoch
        scheduler = SchedulerFactory.create_scheduler(
            scheduler_type=scheduler_type,
            optimizer=optimizer,
            scheduler_config=scheduler_cfg,
            total_steps=total_steps,
        )

        return optimizer, scheduler


if __name__ == "__main__":
    pass

import json
import os
from collections import defaultdict

import numpy as np
import torch


from ..core.framework.base_trainer import DistributedSamplerTrainer


class V2MTrainer(DistributedSamplerTrainer):


    @staticmethod
    def _extract_metrics(output):
        raw = output.get("metrics", {})
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, list) and raw:
            agg = defaultdict(list)
            for m in raw:
                for k, v in m.items():
                    if isinstance(v, (int, float)):
                        agg[k].append(v)
                    elif isinstance(v, np.ndarray) and v.size > 0:
                        agg[k].append(float(v.mean()))
            return {k: float(np.mean(vs)) for k, vs in agg.items()}
        return {}

    @staticmethod
    def _batch_sample_name(batch, sample_idx=0):
        meta = batch.get("meta", {})
        seq = meta.get("sequence_name")
        if seq is not None:
            name = seq[sample_idx] if isinstance(seq, (list, tuple)) else seq
            return str(name)
        # fallback: index
        idx = batch.get("index")
        if idx is not None:
            val = idx[sample_idx] if torch.is_tensor(idx) else idx
            return f"idx={val}"
        return "unknown"

    def validation(self, accelerator, model, output_dir="vis_train", vis=True, seeds = [0, 1, 2, 3]):
        if self.val_dataset is None:
            return
        import math

        rank = accelerator.process_index
        nproc = accelerator.num_processes
        tag = f"[V2MVal][rank={rank}/{nproc}][step={self.global_step}]"

        swap = False
        if self.ema is not None and self.ema_validate:
            print(f"{tag} Swapping to EMA weights for validation ...")
            self.ema.store(accelerator.unwrap_model(model).parameters())
            self.ema.copy_to(accelerator.unwrap_model(model).parameters())
            swap = True

        model.eval()

        dataloader = self.val_dataloader()
        total_batches = len(dataloader)

        real_total = len(self.val_dataset)
        padded_total = math.ceil(real_total / nproc) * nproc
        print(f"{tag} Validation started: {total_batches} batches, seeds=[0,1,2,3], "
              f"dataset={real_total}, padded={padded_total}, pad_count={padded_total - real_total}")

        sum_dict = defaultdict(float)
        cnt_dict = defaultdict(int)
        per_sample_metrics = []
        seen_indices = set()
        n_real = 0
        n_pad = 0
        for batch_idx, batch in enumerate(dataloader):
            sample_name = self._batch_sample_name(batch)
            bs = batch["length"].shape[0] if "length" in batch else 1

            batch_index = batch["index"]
            if torch.is_tensor(batch_index):
                idx_val = batch_index[0].item()
            else:
                idx_val = batch_index
            is_pad = idx_val in seen_indices
            seen_indices.add(idx_val)

            print(f"{tag} Batch {batch_idx+1}/{total_batches} bs={bs} "
                  f"idx={idx_val} sample={sample_name}"
                  f"{' (PAD-SKIP)' if is_pad else ''} validating ...")
            batch_device = self.batch_to_device(batch, accelerator)
            with torch.no_grad():
                unwrapped = accelerator.unwrap_model(model)
                output = unwrapped.validate(batch_device, seeds=seeds)

            if is_pad:
                n_pad += 1
            else:
                n_real += 1
                metrics = self._extract_metrics(output)
                sample_record = {"sample_id": sample_name, "index": idx_val}
                for key, value in metrics.items():
                    if torch.is_tensor(value):
                        value = value.item()
                    sum_dict[key] += float(value)
                    cnt_dict[key] += 1
                    sample_record[key] = float(value)
                per_sample_metrics.append(sample_record)

                if metrics:
                    all_metrics = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                    print(f"{tag} Batch {batch_idx+1}/{total_batches} "
                          f"idx={idx_val} sample={sample_name} {all_metrics}")

        print(f"{tag} All {total_batches} batches done (real={n_real}, pad={n_pad}), "
              f"waiting for other processes ...")
        accelerator.wait_for_everyone()

        local_keys = list(sum_dict.keys())
        if nproc > 1:
            import torch.distributed as dist
            all_keys_list = [None for _ in range(nproc)]
            dist.all_gather_object(all_keys_list, local_keys)
            all_keys = sorted({k for keys in all_keys_list if keys for k in keys})
        else:
            all_keys = local_keys

        print(f"{tag} Reducing {len(all_keys)} metrics across {nproc} processes ...")

        global_avg = {}
        for key in all_keys:
            t = torch.tensor(
                [sum_dict.get(key, 0.0), cnt_dict.get(key, 0)],
                dtype=torch.float64,
                device=accelerator.device,
            )
            t = accelerator.reduce(t, reduction="sum")
            if accelerator.is_main_process:
                total_sum, total_cnt = t[0].item(), t[1].item()
                avg = total_sum / max(total_cnt, 1.0)
                global_avg[key] = avg
                print(f"{tag} [FINAL] {key}: {avg:.3f} (sum={total_sum:.3f}, cnt={int(total_cnt)})")
                if not self.skip_log_zero_epoch or self.global_step > 0:
                    self.writer.add_scalar(f"val/{key}", avg, self.global_step)

        if nproc > 1:
            import torch.distributed as dist
            all_per_sample = [None for _ in range(nproc)]
            dist.all_gather_object(all_per_sample, per_sample_metrics)
            gathered_per_sample = []
            for rank_samples in all_per_sample:
                if rank_samples:
                    gathered_per_sample.extend(rank_samples)
        else:
            gathered_per_sample = per_sample_metrics

        if accelerator.is_main_process and gathered_per_sample:
            gathered_per_sample.sort(key=lambda x: x.get("index", 0))
            train_iters = self.config["train"].get("train_iterations", 1)
            current_epoch = self.global_step // train_iters if train_iters > 0 else 0
            eval_report = {
                "epoch": current_epoch,
                "step": self.global_step,
                "per_sample_metrics": gathered_per_sample,
                "average_metrics": {k: round(v, 6) for k, v in global_avg.items()},
            }
            eval_dir = os.path.join(self.exp, "logs")
            os.makedirs(eval_dir, exist_ok=True)
            eval_filename = f"eval_epoch{current_epoch}_step{self.global_step}.json"
            eval_path = os.path.join(eval_dir, eval_filename)
            with open(eval_path, "w", encoding="utf-8") as f:
                json.dump(eval_report, f, indent=2, ensure_ascii=False)
            print(f"{tag} Evaluation report saved to {eval_path}")

        if self.ema is not None and swap:
            print(f"{tag} Restoring original weights from EMA ...")
            self.ema.restore(accelerator.unwrap_model(model).parameters())

        print(f"{tag} Validation finished.")
        return global_avg

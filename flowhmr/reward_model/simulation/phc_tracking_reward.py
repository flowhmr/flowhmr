from __future__ import annotations

import atexit
import json
import os
import subprocess
import time
import uuid
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

DEFAULT_PHC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_PHC_PYTHON = sys.executable
DEFAULT_WORK_DIR = "/dev/shm/grpo_phc_reward_cache" # tmpfs: avoids disk IO in the training loop


class PHCTrackingReward:

    def __init__(
        self,
        phc_root: str = DEFAULT_PHC_ROOT,
        phc_python: str = DEFAULT_PHC_PYTHON,
        max_frames: int = 180,
        num_envs: int = 64,
        device_id: int = 0,
        score_threshold: float = 0.6,
        score_scale: float = 0.15,
        termination_distance: float = 0.5,
        work_dir: Optional[str] = DEFAULT_WORK_DIR,
        timeout_s: int = 1800,
        ready_timeout_s: int = 300,
        idle_timeout_s: int = 1800,
        start_stagger_s: float = 0.0,
        verbose: bool = False,
    ):
        self.phc_root = phc_root
        self.script = Path(phc_root) / "tools" / "phc_score_server.py"
        assert self.script.exists(), f"phc_score_server.py not found: {self.script}"
        self.python = phc_python
        self.max_frames = int(max_frames)
        self.num_envs = int(num_envs)
        self.device_id = int(device_id)
        self.score_threshold = float(score_threshold)
        self.score_scale = float(score_scale)
        self.termination_distance = float(termination_distance)
        self.timeout_s = timeout_s
        self.ready_timeout_s = ready_timeout_s
        self.idle_timeout_s = idle_timeout_s
        self.start_stagger_s = float(start_stagger_s)
        self.verbose = verbose

        self.work_dir = Path(work_dir) if work_dir else Path(DEFAULT_WORK_DIR)
        if not os.path.isdir("/dev/shm"): # no tmpfs (non-Linux): fall back to a local dir
            self.work_dir = Path("./grpo_phc_reward_cache")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.ctrl_dir = self.work_dir / "server"
        self.ctrl_dir.mkdir(parents=True, exist_ok=True)

        self._proc: Optional[subprocess.Popen] = None
        self._started = False
        self._call_id = 0
        self._total_motions = 0
        self._total_s = 0.0
        self._restarts = 0
        self._closed = False
        atexit.register(self.close)

    # ---------------------------------------------------------------- server
    def _server_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _start_server(self):
        ready = self.ctrl_dir / "ready"
        if ready.exists():
            ready.unlink()
        cmd = [self.python, str(self.script),
               "--ctrl-dir", str(self.ctrl_dir),
               "--num-envs", str(self.num_envs),
               "--device-id", str(self.device_id),
               "--max-frames", str(self.max_frames),
               "--idle-timeout", str(self.idle_timeout_s)]
        log = open(self.ctrl_dir / "server.log", "a")
        self._proc = subprocess.Popen(
            cmd, cwd=str(self.phc_root), stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ,
                 "PATH": os.path.dirname(os.path.abspath(self.python)) + ":" + os.environ.get("PATH", "")})
        t0 = time.time()
        while not ready.exists():
            if self._proc.poll() is not None:
                tail = ""
                try:
                    tail = "".join(open(self.ctrl_dir / "server.log",
                                        errors="ignore").readlines()[-30:])
                except Exception:
                    pass
                raise RuntimeError(
                    f"phc server exited during startup (rc={self._proc.returncode}), "
                    f"server.log tail:\n{tail}")
            if time.time() - t0 > self.ready_timeout_s:
                self._kill_server()
                raise TimeoutError("phc server ready timeout")
            time.sleep(1.0)
        if self.verbose:
            print(f"[PHCTrackingReward] server ready in {time.time()-t0:.1f}s "
                  f"(pid={self._proc.pid})")

    def _ensure_server(self):
        if self._closed:
            return
        if self._server_alive():
            return
        if not self._started and self.start_stagger_s > 0:
            time.sleep(self.start_stagger_s)
        self._started = True
        self._kill_server()
        self._start_server()
        self._restarts += 1

    def _kill_server(self):
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=10)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._kill_server()

    def score_batch(self, motions: List[Dict[str, np.ndarray]]) -> List[dict]:
        n = len(motions)
        if n == 0:
            return []
        self._call_id += 1
        cid = self._call_id
        batch_dir = self.work_dir / f"batch_{cid:06d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        bundle = batch_dir / "motions.npz"
        out_json = batch_dir / "scores.json"
        cmd_json = self.ctrl_dir / f"cmd_{uuid.uuid4().hex}.json"

        payload, keys = {}, []
        for gi, m in enumerate(motions):
            key = f"m{gi:05d}"
            keys.append(key)
            payload[f"{key}_poses"] = np.asarray(m["poses"], dtype=np.float32).reshape(len(m["poses"]), -1)
            payload[f"{key}_trans"] = np.asarray(m["trans"], dtype=np.float32)
            payload[f"{key}_betas"] = np.asarray(m["betas"], dtype=np.float32).reshape(1, -1)
        np.savez(str(bundle), **payload)

        zero = {"tracking_score": 0.0, "terminated": True, "survival": 0.0,
                "mpjpe_cm": 25.0, "num_steps": 0, "status": "sim_fail"}
        t0 = time.time()
        try:
            self._ensure_server()
            json.dump({"bundle": str(bundle), "out": str(out_json),
                       "max_frames": self.max_frames,
                       "score_scale": self.score_scale,
                       "termination_distance": self.termination_distance},
                      open(cmd_json, "w"))
            while cmd_json.exists():
                if self._proc is None or self._proc.poll() is not None:
                    raise RuntimeError("phc server died during batch")
                if time.time() - t0 > self.timeout_s:
                    raise TimeoutError("phc server batch timeout")
                time.sleep(0.2)
            scores = json.load(open(out_json))
            if "error" in scores:
                raise RuntimeError(f"server error: {scores['error']}")
        except Exception as e:
            if self.verbose:
                print(f"[PHCTrackingReward] batch#{cid} failed: {e!r} -> all scores set to 0")
            self._kill_server()
            self._rmtree(batch_dir)
            try:
                cmd_json.unlink()
            except OSError:
                pass
            return [dict(zero) for _ in range(n)]

        dt = time.time() - t0
        self._total_motions += n
        self._total_s += dt
        results = []
        for k in keys:
            results.append(scores.get(k, {
                "tracking_score": 0.0, "terminated": True, "survival": 0.0,
                "mpjpe_cm": 25.0, "num_steps": 0, "status": "missing"}))
        self._rmtree(batch_dir)
        if self.verbose:
            ts = [x["tracking_score"] for x in results]
            print(f"[PHCTrackingReward] batch#{cid}: {n} motions in {dt:.1f}s "
                  f"({n/max(dt,1e-6):.1f}/s), score mean={np.mean(ts):.4f}")
        return results

    @staticmethod
    def _rmtree(d: Path):
        import shutil
        shutil.rmtree(d, ignore_errors=True)

    def stats(self) -> dict:
        return {"calls": self._call_id, "motions": self._total_motions,
                "total_s": round(self._total_s, 1),
                "motions_per_s": round(self._total_motions / max(self._total_s, 1e-6), 2),
                "server_restarts": self._restarts,
                "server_alive": self._server_alive()}

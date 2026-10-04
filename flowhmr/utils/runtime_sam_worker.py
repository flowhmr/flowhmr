import datetime
import pickle
import socket
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from torch import Tensor


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed before the full message was received")
        buf.extend(chunk)
    return bytes(buf)


def send_msg(conn: socket.socket, data: dict):
    payload = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    header = struct.pack("!Q", len(payload)) # 8-byte length header, supports >4GB payloads
    conn.sendall(header + payload)


def recv_msg(conn: socket.socket) -> dict:
    header = _recv_exact(conn, 8)
    length = struct.unpack("!Q", header)[0]
    payload = _recv_exact(conn, length)
    return pickle.loads(payload)


def _worker_main(gpu_id: int, port: int, ready_event):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    from flowhmr.utils.runtime_sam_features import build_sam3d_extractor
    print(f"[Worker GPU:{gpu_id}] loading SAM model...")
    extractor = build_sam3d_extractor(device)
    print(f"[Worker GPU:{gpu_id}] SAM model loaded")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", port))
    server.listen(1)
    server.settimeout(1.0)

    ready_event.set()
    print(f"[Worker GPU:{gpu_id}] listening on port {port}, waiting...")

    while True:
        try:
            conn, _ = server.accept()
        except socket.timeout:
            continue

        request = recv_msg(conn)
        cmd = request.get("cmd")

        if cmd == "shutdown":
            send_msg(conn, {"status": "ok"})
            conn.close()
            print(f"[Worker GPU:{gpu_id}] shutdown received, exiting")
            break

        if cmd == "extract":
            tokens = _do_extract(extractor, request)
            send_msg(conn, {"tokens": tokens.cpu().numpy()})
            conn.close()
        else:
            send_msg(conn, {"error": f"unknown cmd: {cmd}"})
            conn.close()

    server.close()


def _do_extract(extractor, request: dict) -> Tensor:
    frames = request["frames"] # list[np.ndarray] RGB uint8
    bboxes = request["bboxes"]           # np.ndarray (N, 4)
    Ks = request["Ks"]                   # np.ndarray (N, 3, 3)
    token_dim = request.get("token_dim", 3072)

    tokens = []
    for i in range(len(frames)):
        b = bboxes[i:i + 1] # (1, 4) numpy
        cam_int = torch.from_numpy(Ks[i:i + 1]).float()
        token, _ = extractor.extract_frame(
            frames[i],
            bboxes=b,
            cam_int=cam_int,
            inference_type="full",
        )
        tokens.append(token[:token_dim].to(torch.float32))

    if len(tokens) == 0:
        return torch.zeros(0, token_dim)
    return torch.stack(tokens, dim=0)


def decode_video_frames(video_path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")

    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()

    if len(frames) == 0:
        raise ValueError(f"no frames decoded from video: {video_path}")
    return frames


def _pad_crop(tensor: Tensor, target_len: int) -> Tensor:
    src_len = int(tensor.shape[0])
    if src_len == target_len:
        return tensor
    if src_len == 0:
        return torch.zeros(target_len, *tensor.shape[1:], dtype=tensor.dtype)
    if src_len < target_len:
        pad_shape = [target_len - src_len] + [1] * (tensor.ndim - 1)
        pad = tensor[-1:].repeat(*pad_shape)
        tensor = torch.cat([tensor, pad], dim=0)
    return tensor[:target_len]


# SamWorkerPool

class SamWorkerPool:

    def __init__(self, device_ids: List[int], base_port: int = 29500):
        self.device_ids = device_ids
        self.ports = [base_port + i for i in range(len(device_ids))]
        self.processes: List[mp.Process] = []
        self._started = False

    def start(self, timeout_per_worker: float = 300.0):
        if self._started:
            print("[SamWorkerPool] already started, skipped")
            return

        mp.set_start_method("spawn", force=True)

        print(f"[SamWorkerPool] starting {len(self.device_ids)} workers...")
        t0 = time.time()

        events = []
        for i, (gpu_id, port) in enumerate(zip(self.device_ids, self.ports)):
            evt = mp.Event()
            events.append(evt)
            p = mp.Process(
                target=_worker_main,
                args=(gpu_id, port, evt),
                daemon=True,
            )
            p.start()
            self.processes.append(p)

        for i, (gpu_id, evt) in enumerate(zip(self.device_ids, events)):
            print(f"[SamWorkerPool] waiting for Worker {i} (GPU:{gpu_id}) to load the model...")
            if not evt.wait(timeout=timeout_per_worker):
                raise TimeoutError(
                    f"Worker {i} (GPU:{gpu_id}) start timed out ({timeout_per_worker}s)"
                )
            print(f"[SamWorkerPool] Worker {i} (GPU:{gpu_id}) ready")

        self._started = True
        elapsed = time.time() - t0
        print(f"[SamWorkerPool] {len(self.device_ids)} workers ready ({elapsed:.1f}s)")

    def extract_video_tokens(
        self,
        video_path: str,
        *,
        bbox_xyxy: Tensor,
        K_all: Tensor,
        token_dim: int = 3072,
        max_frames: Optional[int] = None,
        progress_cb: Optional[Callable] = None,
    ) -> Tensor:
        if not self._started:
            raise RuntimeError("SamWorkerPool is not started; call start() first")

        def _report(current: int, total: int, message: str):
            """Report progress via callback (same signature as single-GPU)."""
            if progress_cb is not None:
                progress_cb("sam", current, total, message)

        print("[SamWorkerPool] decoding video...")
        t0 = time.time()
        _report(0, 0, "sam: decoding video")
        frames = decode_video_frames(video_path, max_frames=max_frames)
        T = len(frames)
        print(f"[SamWorkerPool] decoded {T} frames ({time.time() - t0:.1f}s)")

        bbox_xyxy = _pad_crop(bbox_xyxy.detach().cpu().float(), T)
        K_all = _pad_crop(K_all.detach().cpu().float(), T)

        _report(0, T, "sam: start")

        N = len(self.device_ids)
        chunk_indices = np.array_split(range(T), N)

        results = {}
        completed_frames = 0
        t_start = time.time()

        def _send_to_worker(worker_idx: int):
            indices = chunk_indices[worker_idx]
            if len(indices) == 0:
                return worker_idx, np.zeros((0, token_dim), dtype=np.float32)

            port = self.ports[worker_idx]
            request = {
                "cmd": "extract",
                "frames": [frames[i] for i in indices],
                "bboxes": bbox_xyxy[indices].numpy(),
                "Ks": K_all[indices].numpy(),
                "token_dim": token_dim,
            }

            conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            conn.connect(("127.0.0.1", port))
            send_msg(conn, request)
            response = recv_msg(conn)
            conn.close()

            return worker_idx, response["tokens"]

        with ThreadPoolExecutor(max_workers=N) as pool:
            futures = [pool.submit(_send_to_worker, i) for i in range(N)]
            for f in futures:
                idx, tokens_np = f.result()
                results[idx] = tokens_np
                n_frames = tokens_np.shape[0] if tokens_np.ndim > 0 else 0
                completed_frames += n_frames
                elapsed = time.time() - t_start
                now_str = datetime.datetime.now().strftime("%H:%M:%S")
                print(
                    f"    [{now_str}] Worker {idx} (GPU:{self.device_ids[idx]}) "
                    f"done {n_frames} frames ({elapsed:.1f}s)"
                )
                _report(completed_frames, T, "sam: running")

        all_tokens = np.concatenate([results[i] for i in range(N)], axis=0)
        result = torch.from_numpy(all_tokens).float()
        total_elapsed = time.time() - t_start
        fps = T / total_elapsed if total_elapsed > 0 else 0
        print(
            f"[SamWorkerPool] done: shape={tuple(result.shape)}, "
            f"{total_elapsed:.1f}s, {fps:.2f} fps"
        )
        _report(T, T, "sam: done")
        return result

    def shutdown(self):
        for port in self.ports:
            try:
                conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                conn.settimeout(5.0)
                conn.connect(("127.0.0.1", port))
                send_msg(conn, {"cmd": "shutdown"})
                recv_msg(conn)
                conn.close()
            except Exception as e:
                print(f"[SamWorkerPool] shutdown of worker on port {port} failed: {e}")

        for p in self.processes:
            p.join(timeout=10)
            if p.is_alive():
                p.kill()

        self.processes.clear()
        self._started = False
        print("[SamWorkerPool] all workers stopped")

    def __del__(self):
        if self._started:
            self.shutdown()

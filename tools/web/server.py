"""FlowHMR web demo: upload a video, get SMPL-H motion, inspect it in a 3D viewer.

Launch (from the repo root):
    python app.py --ckpt checkpoints/flowhmr_latest/flowhmr_latest.ckpt --port 8080
then open http://<host>:8080

Pipeline per job (runs on one GPU, jobs are queued):
    transcode 30fps -> YOLOX detection + tracking -> VGGT-Omega camera
    -> SAM-3D-Body tokens -> FlowHMR flow-matching generation
"""
import argparse
import copy
import json
import os
import queue
import re
import shutil
import sys
import threading
import time
import traceback
import uuid

import numpy as np
import torch
from flask import Flask, abort, jsonify, request, send_file, send_from_directory

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(WEB_DIR))
sys.path.insert(0, REPO_ROOT)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
STAGES = [
    ("detection", "Detection", "YOLOX person detection & tracking"),
    ("camera", "Camera", "VGGT-Omega camera estimation"),
    ("feature_extract", "Body features", "SAM-3D-Body feature extraction"),
    ("generation", "Generation", "FlowHMR flow-matching generation"),
]
STAGE_INDEX = {name: i for i, (name, _, _) in enumerate(STAGES)}
REQUIRED_FILES = [
    "ckpts/yolox/yolox_l.pth",
    "ckpts/vggt-omega/vggt_omega_1b_512.pt",
    "ckpts/sam-3d-body-dinov3/model.ckpt",
    "ckpts/sam-3d-body-dinov3/assets/mhr_model.pt",
    "assets/body_models/smplh/neutral/model.npz",
    "assets/body_models/smpl_neutral_J_regressor.pt",
]
SUB = "input"  # every job stores its upload as input.<ext>, so intermediates live in <job>/input/

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024

ENGINE = None
ARGS = None
JOBS = {}
JOBS_LOCK = threading.Lock()
JOB_QUEUE = queue.Queue()
_POSTPROCESS = None
SELECTIONS = {}           # job_id -> {"event": Event, "choice": int | None}
CANCEL_REQUESTS = set()
SELECT_TIMEOUT_S = 600    # a job waiting for a person pick falls back to the default after this


class JobCancelled(Exception):
    pass


# --------------------------------------------------------------------------- engine
def load_engine(args):
    """Load all models once: YOLOX, SAM-3D-Body, VGGT-Omega, FlowHMR, SMPL-H."""
    from flowhmr.pipeline.v2m_inference import V2MInferenceEngine
    from flowhmr.utils.runtime_vggt_camera import DEFAULT_VGGT_OMEGA_CKPT, _load_vggt_model

    model_cfg = args.model_cfg
    if model_cfg is None:
        sibling = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "config.yml")
        model_cfg = sibling if os.path.exists(sibling) else os.path.join(
            REPO_ROOT, "configs", "base", "model_basic05B.yml")

    engine = V2MInferenceEngine.build(
        ckpt=args.ckpt, model_cfg_path=model_cfg, device=args.device,
    )
    _load_vggt_model(DEFAULT_VGGT_OMEGA_CKPT, engine.device)  # warm cache, avoids first-job delay
    return engine


def get_postprocess():
    """Post-processing pipeline, built on first use (only needed when a job asks for it)."""
    global _POSTPROCESS
    if _POSTPROCESS is None:
        from flowhmr.utils.runtime_v2m_postprocess_simple import PostprocessPipeline
        _POSTPROCESS = PostprocessPipeline.default(body_model=ENGINE.body_model,
                                                   smpl_mesh=ENGINE.smpl_mesh)
    return _POSTPROCESS


# --------------------------------------------------------------------------- parameters
# Defaults for the page's Settings card; everything is editable per run in the UI.
DEFAULT_PARAMS = {
    "seed": 0, "cfg_scale": 1.0, "steps": 20, "max_frames": 900, "vggt_interval": 1,
    "postprocess": False, "manual_select": True, "auto_frames": False,
}


def default_params():
    return dict(DEFAULT_PARAMS)


def _to_bool(value, default):
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "on", "yes")


def parse_params(src, base=None):
    """Validate per-job parameters from a form / JSON body; missing keys fall back to `base`."""
    base = {**default_params(), **(base or {})}

    def number(key, cast, lo, hi):
        raw = src.get(key)
        if raw is None or raw == "":
            return base[key]
        try:
            value = cast(raw)
        except (TypeError, ValueError):
            raise ValueError(f"invalid {key}: {raw!r}")
        if not lo <= value <= hi:
            raise ValueError(f"{key} must be between {lo} and {hi}")
        return value

    return {
        "seed": number("seed", int, 0, 2 ** 31 - 1),
        "cfg_scale": number("cfg_scale", float, 0.0, 10.0),
        "steps": number("steps", int, 1, 100),
        "max_frames": number("max_frames", int, 30, 10000),
        "auto_frames": _to_bool(src.get("auto_frames"), base["auto_frames"]),
        "vggt_interval": number("vggt_interval", int, 1, 120),
        "postprocess": _to_bool(src.get("postprocess"), base["postprocess"]),
        "manual_select": _to_bool(src.get("manual_select"), base.get("manual_select", True)),
    }


# --------------------------------------------------------------------------- jobs
def job_dir(job_id):
    return os.path.join(ARGS.work_dir, job_id)


def _persist_job_locked(job):
    path = os.path.join(job_dir(job["id"]), "job.json")
    temp = path + ".tmp"
    saved = dict(job)
    if job.get("status") != "waiting_user":
        saved["candidates"] = None
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(saved, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def _legacy_job(job_id, directory):
    """Recover pre-persistence runs from their input and completed result files."""
    inputs = [name for name in os.listdir(directory)
              if name.startswith("input.") and os.path.isfile(os.path.join(directory, name))]
    if not inputs:
        return None
    input_name = inputs[0]
    created = os.path.getmtime(os.path.join(directory, input_name))
    try:
        created = time.mktime(time.strptime(job_id[:15], "%Y%m%d_%H%M%S"))
    except ValueError:
        pass
    meta_path = os.path.join(directory, "mesh", "meta.json")
    motion_path = os.path.join(directory, "motion.npz")
    done = os.path.isfile(meta_path) and os.path.isfile(motion_path)
    meta = {}
    if done:
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError):
            done = False
    status = "done" if done else "failed"
    stages = [{"name": name, "label": label, "detail": detail,
               "status": "completed" if done else ("failed" if i == 0 else "pending"),
               "progress": 100 if done else 0,
               "start": created if done else None, "end": created if done else None}
              for i, (name, label, detail) in enumerate(STAGES)]
    return {
        "id": job_id, "filename": f"Recovered run {job_id}", "params": default_params(),
        "recovered": True,
        "ext": os.path.splitext(input_name)[1], "status": status,
        "message": "Done" if done else "Recovered from an earlier session; detailed progress is unavailable.",
        "error": None if done else "This earlier run did not have saved job details.",
        "progress": 100 if done else 0, "created": created,
        "started": None, "finished": created if done else None,
        "num_frames": meta.get("num_frames") if done else None, "fps": meta.get("fps", 30),
        "has_fbx": os.path.isfile(os.path.join(directory, "motion.fbx")),
        "candidates": None, "default_person": None, "selected_person": None,
        "selection_auto": False, "stages": stages,
    }


def restore_jobs():
    """Restore persisted jobs and migrate older run folders into the history."""
    for entry in os.scandir(ARGS.work_dir):
        if not entry.is_dir() or not re.fullmatch(r"\d{8}_\d{6}_[a-f0-9]{6}", entry.name):
            continue
        path = os.path.join(entry.path, "job.json")
        try:
            with open(path, encoding="utf-8") as f:
                job = json.load(f)
            if not isinstance(job, dict) or not isinstance(job.get("params"), dict):
                raise ValueError("invalid job metadata")
            job["id"] = entry.name
        except FileNotFoundError:
            job = _legacy_job(entry.name, entry.path)
            if job is None:
                continue
        except (OSError, ValueError):
            job = _legacy_job(entry.name, entry.path)
            if job is None:
                continue

        if job.get("status") in ("queued", "running", "waiting_user", "cancel_requested", "uploading"):
            cancelled = job["status"] == "cancel_requested"
            job.update(status="cancelled" if cancelled else "failed",
                       message="Run stopped when FlowHMR restarted." if cancelled
                       else "Run interrupted when FlowHMR restarted.",
                       error=None if cancelled else "The server restarted before this run finished.",
                       finished=time.time())
            for stage in job.get("stages", []):
                if stage.get("status") == "running":
                    stage["status"] = "cancelled" if cancelled else "failed"
        job["params"] = {**default_params(), **job.get("params", {})}
        job.setdefault("stages", [])
        with JOBS_LOCK:
            JOBS[job["id"]] = job
            _persist_job_locked(job)


def new_job(filename, params, ext):
    job_id = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    job = {
        "id": job_id, "filename": filename, "params": params, "ext": ext, "recovered": False,
        "status": "queued", "message": "Waiting in queue", "error": None, "progress": 0,
        "created": time.time(), "started": None, "finished": None,
        "num_frames": None, "fps": 30, "has_fbx": False,
        "candidates": None, "default_person": None, "selected_person": None, "selection_auto": False,
        "stages": [{"name": n, "label": l, "detail": d, "status": "pending", "progress": 0,
                    "start": None, "end": None} for n, l, d in STAGES],
    }
    os.makedirs(job_dir(job_id), exist_ok=True)
    with JOBS_LOCK:
        JOBS[job_id] = job
        _persist_job_locked(job)
    return job


def update_job(job_id, **kw):
    with JOBS_LOCK:
        JOBS[job_id].update(kw)
        _persist_job_locked(JOBS[job_id])


def input_video(job_id):
    return os.path.join(job_dir(job_id), f"input{JOBS[job_id]['ext']}")


def enqueue(job):
    JOB_QUEUE.put(job["id"])


def worker_loop():
    if ARGS.device.startswith("cuda:"):
        torch.cuda.set_device(ARGS.device)  # per-thread; YOLOX builds its grids on the current device
    while True:
        job_id = JOB_QUEUE.get()
        try:
            run_job(job_id)
        except JobCancelled:
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job is not None:
                    for stage in job["stages"]:
                        if stage["status"] == "running":
                            stage["status"] = "cancelled"
                    job.update(status="cancelled", message="Stopped by user", finished=time.time())
                    _persist_job_locked(job)
        except Exception as e:
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job is None:
                    continue
                cancelled = job_id in CANCEL_REQUESTS
                for s in job["stages"]:
                    if s["status"] == "running":
                        s["status"] = "cancelled" if cancelled else "failed"
                job.update(status="cancelled" if cancelled else "failed",
                           error=None if cancelled else f"{type(e).__name__}: {e}",
                           message="Stopped by user" if cancelled else "Failed", finished=time.time())
                _persist_job_locked(job)
            if not cancelled:
                traceback.print_exc()
        finally:
            with JOBS_LOCK:
                CANCEL_REQUESTS.discard(job_id)
            torch.cuda.empty_cache()
            JOB_QUEUE.task_done()


def _raise_if_cancelled(job_id):
    with JOBS_LOCK:
        if job_id in CANCEL_REQUESTS:
            raise JobCancelled()


def run_job(job_id):
    _raise_if_cancelled(job_id)
    with JOBS_LOCK:
        job = JOBS[job_id]
        params = job["params"]
        job.update(status="running", message="Starting", started=time.time())
        _persist_job_locked(job)

    def on_progress(stage, progress, desc):
        idx = STAGE_INDEX.get(getattr(stage, "value", str(stage)))
        if idx is None:
            return
        now = time.time()
        with JOBS_LOCK:
            if job_id in CANCEL_REQUESTS:
                raise JobCancelled()
            job = JOBS[job_id]
            before = (job["progress"], job["stages"][idx]["progress"], job["stages"][idx]["status"])
            for s in job["stages"][:idx]:  # earlier stages are finished once a later one reports
                if s["status"] != "completed":
                    s.update(status="completed", progress=100, end=now, start=s["start"] or now)
            s = job["stages"][idx]
            s["start"] = s["start"] or now
            s["progress"] = int(max(s["progress"], min(progress, 100)))
            if s["progress"] >= 100:
                s.update(status="completed", end=s["end"] or now)
            else:
                s["status"] = "running"
            job["progress"] = int(sum(x["progress"] for x in job["stages"]) / len(job["stages"]))
            job["message"] = desc
            after = (job["progress"], s["progress"], s["status"])
            if before != after:
                _persist_job_locked(job)

    def select_person(result, candidates, default_idx):
        """Pause the job until the user picks a person in the page (or the wait times out)."""
        slot = {"event": threading.Event(), "choice": None}
        try:
            save_person_thumbnails(job_id, candidates)
        except Exception:  # thumbnails are only a UI aid
            traceback.print_exc()
        with JOBS_LOCK:
            SELECTIONS[job_id] = slot
            job = JOBS[job_id]
            job["stages"][STAGE_INDEX["detection"]]["status"] = "waiting_user"
            job.update(status="waiting_user", candidates=candidates, default_person=default_idx,
                       video_size=[result.width, result.height],
                       message=f"{len(candidates)} people detected - pick the person to reconstruct")
            _persist_job_locked(job)
        picked = slot["event"].wait(SELECT_TIMEOUT_S)
        choice = slot["choice"] if picked and slot["choice"] is not None else default_idx
        with JOBS_LOCK:
            if job_id in CANCEL_REQUESTS:
                SELECTIONS.pop(job_id, None)
                raise JobCancelled()
            SELECTIONS.pop(job_id, None)
            job = JOBS[job_id]
            job["stages"][STAGE_INDEX["detection"]]["status"] = "running"
            job.update(status="running", selected_person=choice, selection_auto=not picked,
                       message=f"Person #{choice} selected" if picked else "No selection - using default person")
            _persist_job_locked(job)
        return choice

    ENGINE.pipeline.validation_steps = params["steps"]
    ENGINE.postprocess = get_postprocess() if params["postprocess"] else None
    sub_dir = ENGINE.process_video(
        input_video(job_id), job_dir(job_id), seed=params["seed"], cfg_scale=params["cfg_scale"],
        max_frames=None if params["auto_frames"] else params["max_frames"],
        vggt_frame_interval=params["vggt_interval"],
        person_selector=select_person if params.get("manual_select", True) else None,
        progress_callback=on_progress,
        cancel_check=lambda: _raise_if_cancelled(job_id),
    )
    _raise_if_cancelled(job_id)
    result = os.path.join(sub_dir, f"{SUB}_seed{params['seed']}" + ("" if params["postprocess"] else "_origin"))
    if not os.path.exists(result + ".npz"):
        raise FileNotFoundError(f"generation output missing: {result}.npz")

    update_job(job_id, message="Building 3D mesh")
    out_root = job_dir(job_id)
    _raise_if_cancelled(job_id)
    meta = export_mesh_sequence(result + ".npz", os.path.join(out_root, "mesh"))
    _raise_if_cancelled(job_id)
    start, end = 0, meta["num_frames"] - 1
    bbox_npz = os.path.join(sub_dir, f"{SUB}_bbox.npz")
    if os.path.exists(bbox_npz):
        se = np.load(bbox_npz)["start_end"]
        start, end = int(se[0]), int(se[1])
    meta.update({"track_start": start, "track_end": end})
    with open(os.path.join(out_root, "mesh", "meta.json"), "w") as f:
        json.dump(meta, f)

    shutil.copy2(result + ".npz", os.path.join(out_root, "motion.npz"))
    has_fbx = os.path.exists(result + ".fbx")
    if has_fbx:
        shutil.copy2(result + ".fbx", os.path.join(out_root, "motion.fbx"))

    now = time.time()
    with JOBS_LOCK:
        if job_id in CANCEL_REQUESTS:
            raise JobCancelled()
        job = JOBS[job_id]
        for s in job["stages"]:
            if s["status"] != "completed":
                s.update(status="completed", progress=100, end=now, start=s["start"] or now)
        job.update(status="done", progress=100, message="Done", finished=now,
                   num_frames=meta["num_frames"], has_fbx=has_fbx)
        _persist_job_locked(job)


# --------------------------------------------------------------------------- person thumbnails
def save_person_thumbnails(job_id, candidates):
    """One crop per candidate (at the frame where its box is largest), for the selection list."""
    import cv2
    video = os.path.join(job_dir(job_id), SUB, f"{SUB}_30fps.mp4")
    out_dir = os.path.join(job_dir(job_id), "persons")
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    try:
        for c in candidates:
            boxes = np.asarray(c["boxes"], dtype=np.float64)
            k = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(c["frames"][k]))
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = boxes[k]
            px, py = 0.15 * (x2 - x1), 0.08 * (y2 - y1)
            x1, y1 = int(max(0, x1 - px)), int(max(0, y1 - py))
            x2, y2 = int(min(w, x2 + px)), int(min(h, y2 + py))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            scale = 192.0 / crop.shape[0]
            crop = cv2.resize(crop, (max(1, int(round(crop.shape[1] * scale))), 192),
                              interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(out_dir, f"{c['index']}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
    finally:
        cap.release()


# --------------------------------------------------------------------------- mesh export
def export_mesh_sequence(motion_npz, out_dir):
    """Run SMPL-H forward on the generated motion and write a float16 vertex stream.

    No ground alignment: Y=0 is the floor and the model's root height is shown
    as-is (open post-processing in the page if feet should contact the floor).
    """
    os.makedirs(out_dir, exist_ok=True)
    data = np.load(motion_npz, allow_pickle=True)
    poses = torch.from_numpy(np.asarray(data["poses"], dtype=np.float32))
    trans = torch.from_numpy(np.asarray(data["trans"], dtype=np.float32))
    betas = torch.from_numpy(np.asarray(data["betas"], dtype=np.float32)).reshape(1, -1)
    T = poses.shape[0]
    poses = poses.reshape(T, -1, 3)

    mesh_model = ENGINE.smpl_mesh
    dev = mesh_model.v_template.device
    betas = betas[:, :mesh_model.shapedirs.shape[-1]].to(dev)

    verts = []
    with torch.no_grad():
        for s in range(0, T, 64):
            e = min(T, s + 64)
            out = mesh_model({"poses": poses[s:e].to(dev), "shapes": betas,
                              "trans": trans[s:e].to(dev)})
            verts.append(out["vertices"].float().cpu())
    verts = torch.cat(verts, 0).numpy()
    faces = np.asarray(mesh_model.faces, dtype=np.uint32)

    verts.astype(np.float16).tofile(os.path.join(out_dir, "vertices.bin"))
    faces.tofile(os.path.join(out_dir, "faces.bin"))

    root = trans.numpy().copy()
    return {
        "num_frames": int(T), "num_vertices": int(verts.shape[1]), "num_faces": int(faces.shape[0]),
        "fps": 30, "vertex_dtype": "float16", "face_dtype": "uint32", "up_axis": "y",
        "bbox_min": verts.reshape(-1, 3).min(0).tolist(),
        "bbox_max": verts.reshape(-1, 3).max(0).tolist(),
        "root_trajectory": root[:, [0, 2]].round(4).tolist(),
    }


# --------------------------------------------------------------------------- routes
def get_job(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            abort(404)
        return copy.deepcopy(job)


@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory(os.path.join(WEB_DIR, "static"), path)


@app.route("/api/info")
def api_info():
    return jsonify({
        "model": os.path.basename(ARGS.ckpt), "device": str(ENGINE.device),
        "steps": DEFAULT_PARAMS["steps"], "max_frames": DEFAULT_PARAMS["max_frames"],
        "fps": 30, "defaults": default_params(),
        "stages": [{"name": n, "label": l, "detail": d} for n, l, d in STAGES],
        "max_upload_mb": app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024),
    })


@app.route("/api/upload", methods=["POST"])
def api_upload():
    f = request.files.get("video")
    if f is None or not f.filename:
        return jsonify({"error": "no video file"}), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in VIDEO_EXTS:
        return jsonify({"error": f"unsupported format {ext}; use {', '.join(sorted(VIDEO_EXTS))}"}), 400
    try:
        params = parse_params(request.form)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    job = new_job(f.filename, params, ext)
    f.save(input_video(job["id"]))
    enqueue(job)
    return jsonify({"job_id": job["id"]})


@app.route("/api/jobs/<job_id>/rerun", methods=["POST"])
def api_rerun(job_id):
    """New job on the same video; cached detection / camera / SAM features are reused."""
    src = get_job(job_id)
    if src["status"] in ("queued", "running", "waiting_user", "cancel_requested"):
        return jsonify({"error": "job is still running"}), 409
    body = request.get_json(silent=True) or {}
    try:
        params = parse_params(body, base=src["params"])
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    job = new_job(src["filename"], params, src["ext"])
    shutil.copy2(input_video(job_id), input_video(job["id"]))
    src_sub = os.path.join(job_dir(job_id), SUB)
    if src["status"] in ("done", "failed", "cancelled") and os.path.isdir(src_sub):
        dst_sub = os.path.join(job_dir(job["id"]), SUB)
        os.makedirs(dst_sub, exist_ok=True)
        skip = (f"{SUB}_seed",)  # generated motion is recomputed
        if _to_bool(body.get("reselect_person"), False):
            # per-person caches are invalidated so detection runs and the user picks again
            skip += (f"{SUB}_bbox.npz", f"{SUB}_sam3d_feat.pt", f"{SUB}_vggt_camera.npz")
        for name in os.listdir(src_sub):
            if not name.startswith(skip):
                shutil.copy2(os.path.join(src_sub, name), dst_sub)
    enqueue(job)
    return jsonify({"job_id": job["id"]})


@app.route("/api/jobs/<job_id>/cancel", methods=["POST"])
def api_cancel_job(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            abort(404)
        if job["status"] not in ("queued", "running", "waiting_user"):
            return jsonify({"error": "run is no longer active"}), 409
        CANCEL_REQUESTS.add(job_id)
        job.update(status="cancel_requested", message="Stop requested; stopping at the next safe step")
        _persist_job_locked(job)
        slot = SELECTIONS.get(job_id)
        if slot is not None:
            slot["event"].set()
    return jsonify({"ok": True, "job_id": job_id})


@app.route("/api/jobs/<job_id>/select", methods=["POST"])
def api_select(job_id):
    """Resume a job waiting for person selection: body {"person": <candidate index>}."""
    get_job(job_id)
    body = request.get_json(silent=True) or {}
    with JOBS_LOCK:
        slot = SELECTIONS.get(job_id)
        n = len(JOBS[job_id]["candidates"] or [])
    if slot is None:
        return jsonify({"error": "job is not waiting for a selection"}), 409
    try:
        person = int(body.get("person"))
    except (TypeError, ValueError):
        return jsonify({"error": "person must be an integer"}), 400
    if not 0 <= person < n:
        return jsonify({"error": f"person must be in [0, {n - 1}]"}), 400
    slot["choice"] = person
    slot["event"].set()
    return jsonify({"ok": True, "person": person})


@app.route("/api/jobs/<job_id>")
def api_job(job_id):
    job = get_job(job_id)
    with JOBS_LOCK:
        job["queue_position"] = sum(1 for j in JOBS.values()
                                    if j["status"] == "queued" and j["created"] < job["created"])
    job["now"] = time.time()
    return jsonify(job)


@app.route("/api/jobs")
def api_jobs():
    keys = ("id", "filename", "status", "progress", "created", "finished", "num_frames", "params", "recovered")
    with JOBS_LOCK:
        jobs = sorted(JOBS.values(), key=lambda j: j["created"], reverse=True)[:20]
        deletable_count = sum(j["status"] not in ("queued", "running", "waiting_user", "cancel_requested", "uploading")
                              for j in JOBS.values())
        return jsonify({"now": time.time(),
                        "jobs": [{k: copy.deepcopy(j[k]) for k in keys} for j in jobs],
                        "deletable_count": deletable_count})


@app.route("/api/jobs", methods=["DELETE"])
def api_delete_jobs():
    """Delete selected jobs, or all finished jobs when no ids are supplied."""
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return jsonify({"error": "request body must be a JSON object"}), 400
    requested = body.get("ids")
    if requested is not None and (not isinstance(requested, list) or
                                  any(not isinstance(job_id, str) for job_id in requested)):
        return jsonify({"error": "ids must be a list of job IDs"}), 400

    deleted, skipped = [], []
    with JOBS_LOCK:
        job_ids = list(JOBS) if requested is None else list(dict.fromkeys(requested))
        for job_id in job_ids:
            job = JOBS.get(job_id)
            if job is None:
                if requested is not None:
                    skipped.append({"id": job_id, "reason": "not found"})
                continue
            if job["status"] in ("queued", "running", "waiting_user", "cancel_requested", "uploading"):
                skipped.append({"id": job_id, "reason": "job is still active"})
                continue
            try:
                shutil.rmtree(job_dir(job_id), ignore_errors=False)
            except FileNotFoundError:
                pass
            except OSError as e:
                skipped.append({"id": job_id, "reason": f"could not remove files: {e}"})
                continue
            JOBS.pop(job_id)
            SELECTIONS.pop(job_id, None)
            deleted.append(job_id)
    return jsonify({"deleted": deleted, "skipped": skipped})


@app.route("/api/jobs/<job_id>/person/<int:index>.jpg")
def api_person_thumb(job_id, index):
    get_job(job_id)
    path = os.path.join(job_dir(job_id), "persons", f"{index}.jpg")
    if not os.path.exists(path):
        abort(404)
    return send_file(path, mimetype="image/jpeg", conditional=True)


@app.route("/api/jobs/<job_id>/track")
def api_track(job_id):
    """Tracked person's boxes (xyxy, pixels of the 30 fps video), available after detection."""
    get_job(job_id)
    path = os.path.join(job_dir(job_id), SUB, f"{SUB}_bbox.npz")
    if not os.path.exists(path):
        abort(404)
    data = np.load(path)
    start, end = (int(x) for x in data["start_end"])
    return jsonify({"fps": 30, "start": start, "end": end,
                    "boxes": np.round(data["bbox"].astype(np.float64), 1).tolist()})


_RESULT_FILES = {
    "meta": ("mesh/meta.json", "application/json"),
    "vertices": ("mesh/vertices.bin", "application/octet-stream"),
    "faces": ("mesh/faces.bin", "application/octet-stream"),
    "npz": ("motion.npz", "application/octet-stream"),
    "fbx": ("motion.fbx", "application/octet-stream"),
}


@app.route("/api/jobs/<job_id>/<kind>")
def api_result(job_id, kind):
    job = get_job(job_id)
    if kind == "video":  # 30 fps transcode once available, the raw upload before that
        path = os.path.join(job_dir(job_id), SUB, f"{SUB}_30fps.mp4")
        if not os.path.exists(path):
            path = input_video(job_id)
        return send_file(path, conditional=True)
    if kind not in _RESULT_FILES:
        abort(404)
    rel, mime = _RESULT_FILES[kind]
    path = os.path.join(job_dir(job_id), rel)
    if not os.path.exists(path):
        abort(404)
    as_attachment = kind in ("npz", "fbx")
    name = f"{os.path.splitext(job['filename'])[0]}_flowhmr.{kind}" if as_attachment else None
    return send_file(path, mimetype=mime, as_attachment=as_attachment,
                     download_name=name, conditional=True)


# --------------------------------------------------------------------------- main
def parse_args():
    p = argparse.ArgumentParser(prog="app.py", description="FlowHMR web demo")
    p.add_argument("--ckpt", default=os.path.join(REPO_ROOT, "checkpoints", "flowhmr_latest",
                                                  "flowhmr_latest.ckpt"),
                   help="FlowHMR checkpoint (config.yml next to it)")
    p.add_argument("--model_cfg", default=None)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--device", default="cuda")
    p.add_argument("--work_dir", default=os.path.join(REPO_ROOT, "output", "web_jobs"))
    return p.parse_args()


def main():
    global ENGINE, ARGS
    ARGS = parse_args()
    ARGS.ckpt = os.path.abspath(ARGS.ckpt)
    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(REPO_ROOT, f))]
    ARGS.work_dir = os.path.abspath(ARGS.work_dir)
    if not os.path.exists(ARGS.ckpt):
        missing.insert(0, ARGS.ckpt)
    if missing:
        sys.exit("missing files (see docs/install.md):\n  " + "\n  ".join(missing))
    os.makedirs(ARGS.work_dir, exist_ok=True)
    restore_jobs()
    os.chdir(REPO_ROOT)  # body-model / asset paths are repo-relative
    ENGINE = load_engine(ARGS)
    threading.Thread(target=worker_loop, daemon=True).start()
    print(f"\n  FlowHMR web demo ready:  http://{ARGS.host}:{ARGS.port}\n")
    app.run(host=ARGS.host, port=ARGS.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()

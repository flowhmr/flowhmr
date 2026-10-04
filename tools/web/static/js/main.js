// Page controller: wires the cards together and polls the current job.
import { $, show, createBus } from "./utils.js";
import * as api from "./api.js";
import { initUpload } from "./upload.js";
import { initParams } from "./params.js";
import { initProgress } from "./progress.js";
import { initPersonSelect } from "./person-select.js";
import { initPreview } from "./preview.js";
import { initHistory } from "./history.js";
import { initViewer } from "./viewer.js";

const bus = createBus();
let info = null;
let currentJob = null;
let pollTimer = null;
let loadedResult = null;
let paramsShownFor = null;
let submitting = false;

const upload = initUpload(bus);
const progress = initProgress();
const person = initPersonSelect(bus);
const preview = initPreview(bus);
const viewer = initViewer({ video: preview.video });
const history = initHistory({ onOpen: openJob, onRerun: rerun });
let params = null;

// ------------------------------------------------------------------ boot
try {
  info = await api.getInfo();
  $("modelInfo").textContent = `${info.model} · ${info.steps} steps · ${info.device}`;
  $("uploadHint").textContent = `MP4, MOV, AVI, MKV, WebM · up to ${info.max_upload_mb} MB · first ${info.max_frames} frames (30 fps) are processed`;
} catch {
  $("modelInfo").textContent = "server unavailable";
}
params = initParams(info ? info.defaults : {});
history.refresh();
syncSubmit();

// ------------------------------------------------------------------ wiring
bus.on("file:selected", (file) => {
  preview.showLocal(file);
  syncSubmit();
});
bus.on("file:cleared", () => {
  if (!currentJob) preview.clear();
  syncSubmit();
});
bus.on("preview:pick", (i) => person.select(i));
bus.on("person:confirmed", () => poll());
bus.on("person:reselect", async () => {
  const from = currentJob;
  if (!from || submitting) return;
  submitting = true;
  syncSubmit();
  resetJobView();
  try {
    const p = { ...params.read(), manual_select: true, reselect_person: true };
    const id = await api.rerunJob(from, p);
    await openJob(id);
  } catch (e) {
    progress.showError(e.message || "Re-run failed");
  } finally {
    submitting = false;
    syncSubmit();
  }
});

$("btnSubmit").addEventListener("click", submit);
$("btnRerun").addEventListener("click", async () => {
  if (!currentJob || submitting) return;
  submitting = true;
  syncSubmit();
  try {
    await rerun(currentJob);
  } catch (e) {
    progress.showError(e.message || "Re-run failed");
  } finally {
    submitting = false;
    syncSubmit();
  }
});

function syncSubmit() {
  const btn = $("btnSubmit");
  btn.disabled = !upload.getFile() || submitting;
  btn.textContent = submitting ? "Submitting..." : "Run FlowHMR";
}

async function submit() {
  const file = upload.getFile();
  if (!file) return;
  let p;
  try { p = params.read(); } catch (e) { upload.setError(e.message); return; }
  submitting = true;
  syncSubmit();
  resetJobView();
  try {
    const id = await api.uploadVideo(file, p, (frac) => progress.showUploading(file.name, frac));
    upload.clear();
    await openJob(id);
  } catch (e) {
    progress.showError(e.message);
  } finally {
    submitting = false;
    syncSubmit();
  }
}

async function rerun(jobId) {
  const p = params.read();
  const id = await api.rerunJob(jobId, p);
  await openJob(id);
}

function resetJobView() {
  clearTimeout(pollTimer);
  currentJob = null;
  loadedResult = null;
  person.reset();
  preview.setTrack(null);
  viewer.clear();
  show("btnRerun", false);
}

// ------------------------------------------------------------------ job polling
async function openJob(id) {
  resetJobView();
  currentJob = id;
  history.setCurrent(id);
  await poll();
}

async function poll() {
  clearTimeout(pollTimer);
  const id = currentJob;
  if (!id) return;
  let job;
  try {
    job = await api.getJob(id);
  } catch (e) {
    pollTimer = setTimeout(poll, 2000);
    return;
  }
  if (id !== currentJob) return;

  progress.render(job);
  viewer.setStatus(job.status === "done" ? "Loading 3D result..."
    : job.status === "failed" ? "Reconstruction failed - see Pipeline on the left"
    : job.status === "waiting_user" ? "Pick the person to reconstruct on the left"
    : "Reconstructing... the 3D motion appears here when the pipeline finishes");
  person.update(job);
  preview.showJob(api.videoUrl(id), job.filename);
  if (paramsShownFor !== id) { params.apply(job.params); paramsShownFor = id; }  // once, so edits stick
  history.refresh();

  if (["queued", "running", "waiting_user"].includes(job.status)) {
    pollTimer = setTimeout(poll, job.status === "waiting_user" ? 1500 : 1000);
  } else if (job.status === "done" && loadedResult !== id) {
    loadedResult = id;
    try {
      const meta = await viewer.load(id, job);
      loadTrack(id, meta);
    } catch (e) {
      progress.showError(`Could not load the 3D result: ${e.message}`);
    }
  }
}

async function loadTrack(id, meta) {
  try {
    const r = await fetch(`/api/jobs/${id}/track`);
    if (!r.ok || id !== currentJob) return;
    const t = await r.json();
    preview.setTrack({ start: t.start, end: t.end, boxes: t.boxes, meta });
  } catch { /* the overlay is optional */ }
}

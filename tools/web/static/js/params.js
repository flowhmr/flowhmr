// "Settings" card: person selection, camera mode, sampling and post-processing.
import { $ } from "./utils.js";

function bindToggle(id) {
  const btn = $(id);
  btn.addEventListener("click", () => {
    btn.setAttribute("aria-checked", btn.getAttribute("aria-checked") === "true" ? "false" : "true");
  });
  return {
    get: () => btn.getAttribute("aria-checked") === "true",
    set: (v) => btn.setAttribute("aria-checked", v ? "true" : "false"),
  };
}

function samChunkSize(value) {
  if (typeof value === "string" && /^\d+$/.test(value.trim())) value = Number(value.trim());
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new Error("SAM chunk must be a positive integer.");
  }
  return value;
}

export function initParams(defaults) {
  const manual = bindToggle("optManual");
  const post = bindToggle("optPostprocess");
  const autoFrames = bindToggle("optAutoFrames");
  const staticCamera = bindToggle("optStaticCamera");
  const seed = $("optSeed"), cfg = $("optCfg"), stride = $("optVggt");
  const steps = $("optSteps"), maxFrames = $("optMaxFrames");
  const samChunk = $("optSamChunk");
  const syncMaxFrames = () => { maxFrames.disabled = autoFrames.get(); };
  $("optAutoFrames").addEventListener("click", syncMaxFrames);
  const syncCamera = () => { stride.disabled = staticCamera.get(); };
  $("optStaticCamera").addEventListener("click", syncCamera);

  const apply = (p) => {
    const chunkSize = samChunkSize(p.sam_chunk_size ?? 16);
    manual.set(p.manual_select ?? true);
    post.set(!!p.postprocess);
    autoFrames.set(!!p.auto_frames);
    staticCamera.set(!!p.static_camera);
    seed.value = p.seed ?? 0;
    cfg.value = p.cfg_scale ?? 1.0;
    stride.value = p.vggt_interval ?? 1;
    steps.value = p.steps ?? 20;
    maxFrames.value = p.max_frames ?? 900;
    samChunk.value = chunkSize;
    syncMaxFrames();
    syncCamera();
  };
  apply(defaults || {});

  const read = () => {
    const out = {
      manual_select: manual.get(),
      postprocess: post.get(),
      static_camera: staticCamera.get(),
      seed: parseInt(seed.value, 10),
      cfg_scale: parseFloat(cfg.value),
      vggt_interval: parseInt(stride.value, 10),
      sam_chunk_size: samChunkSize(samChunk.value),
      steps: parseInt(steps.value, 10),
      max_frames: parseInt(maxFrames.value, 10),
      auto_frames: autoFrames.get(),
    };
    if (!Number.isInteger(out.seed) || out.seed < 0) throw new Error("Seed must be a non-negative integer.");
    if (!isFinite(out.cfg_scale) || out.cfg_scale < 0 || out.cfg_scale > 10) throw new Error("CFG scale must be between 0 and 10.");
    if (!Number.isInteger(out.vggt_interval) || out.vggt_interval < 1 || out.vggt_interval > 120) throw new Error("VGGT stride must be between 1 and 120.");
    if (!Number.isInteger(out.steps) || out.steps < 1 || out.steps > 100) throw new Error("Steps must be between 1 and 100.");
    if (!Number.isInteger(out.max_frames) || out.max_frames < 30 || out.max_frames > 10000) throw new Error("Max frames must be between 30 and 10000.");
    return out;
  };

  return { read, apply };
}

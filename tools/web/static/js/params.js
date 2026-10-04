// "Settings" card: manual person selection, seed, CFG scale, VGGT stride, post-processing.
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

export function initParams(defaults) {
  const manual = bindToggle("optManual");
  const post = bindToggle("optPostprocess");
  const seed = $("optSeed"), cfg = $("optCfg"), stride = $("optVggt");
  const steps = $("optSteps"), maxFrames = $("optMaxFrames");

  const apply = (p) => {
    manual.set(p.manual_select ?? true);
    post.set(!!p.postprocess);
    seed.value = p.seed ?? 0;
    cfg.value = p.cfg_scale ?? 1.0;
    stride.value = p.vggt_interval ?? 1;
    steps.value = p.steps ?? 20;
    maxFrames.value = p.max_frames ?? 900;
  };
  apply(defaults || {});

  const read = () => {
    const out = {
      manual_select: manual.get(),
      postprocess: post.get(),
      seed: parseInt(seed.value, 10),
      cfg_scale: parseFloat(cfg.value),
      vggt_interval: parseInt(stride.value, 10),
      steps: parseInt(steps.value, 10),
      max_frames: parseInt(maxFrames.value, 10),
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

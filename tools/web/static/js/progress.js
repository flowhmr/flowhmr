// "Pipeline" card: queue state, per-stage progress bars and elapsed time, errors.
import { $, show, el, formatSeconds, STATUS_LABEL } from "./utils.js";

export function initProgress() {
  const list = $("stageList");

  function renderStages(stages, now) {
    list.innerHTML = "";
    for (const s of stages) {
      const row = el("li", `stage ${s.status}`);
      row.title = s.detail || "";
      row.append(el("span", "stage-dot"), el("span", "stage-label", s.label));
      const bar = el("div", "stage-bar");
      const fill = el("div", "stage-fill");
      fill.style.width = `${s.status === "completed" ? 100 : s.progress}%`;
      bar.append(fill);
      row.append(bar);
      row.append(el("span", "stage-pct num", `${s.status === "completed" ? 100 : Math.round(s.progress)}%`));
      let t = "";
      if (s.start) t = formatSeconds((s.end || now) - s.start);
      if (s.status === "waiting_user") t = "waiting";
      row.append(el("span", "stage-time num", t));
      list.append(row);
    }
  }

  return {
    showUploading(name, frac) {
      show("jobCard", true);
      show("btnRerun", false);
      $("jobName").textContent = name;
      setBadge("uploading");
      show("queueBox", false);
      $("jobMsg").textContent = `Uploading ${Math.round(frac * 100)}%`;
      show("jobError", false);
    },
    render(job) {
      show("jobCard", true);
      show("btnRerun", ["done", "failed"].includes(job.status));
      $("jobName").textContent = job.filename;
      $("jobId").textContent = job.id;
      setBadge(job.status);
      const queued = job.status === "queued";
      show("queueBox", queued);
      if (queued) {
        $("queueText").textContent = job.queue_position > 0
          ? `${job.queue_position} job(s) ahead of yours` : "Starting soon";
      }
      renderStages(job.stages, job.now);
      let msg = job.message || "";
      if (job.status === "done") {
        const total = job.finished && job.started ? ` in ${formatSeconds(job.finished - job.started)}` : "";
        msg = `${job.num_frames} frames reconstructed${total}`;
      }
      $("jobMsg").textContent = msg;
      $("jobError").textContent = job.error || "";
      show("jobError", job.status === "failed" && !!job.error);
    },
    showError(msg) {
      show("jobCard", true);
      show("btnRerun", false);
      setBadge("failed");
      $("jobError").textContent = msg;
      show("jobError", true);
    },
  };

  function setBadge(status) {
    const b = $("jobBadge");
    b.textContent = STATUS_LABEL[status] || status;
    b.className = `badge ${status}`;
  }
}

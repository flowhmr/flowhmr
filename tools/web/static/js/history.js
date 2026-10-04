// "Recent runs" card: jobs of this server session, open / re-run.
import { $, show, el, formatRelative, STATUS_LABEL } from "./utils.js";
import * as api from "./api.js";

export function initHistory({ onOpen, onRerun }) {
  const list = $("historyList");
  let current = null;
  let busy = null;

  async function refresh() {
    let data;
    try { data = await api.getJobs(); } catch { return; }
    const { jobs, now } = data;
    list.innerHTML = "";
    show("historyEmpty", jobs.length === 0);
    for (const j of jobs) {
      const row = el("div", `run${j.id === current ? " current" : ""}`);
      const open = el("button", "run-open");
      open.type = "button";
      const info = el("div", "run-info");
      info.append(el("div", "run-name", j.filename),
        el("div", "run-sub", `${formatRelative(j.created, now)} · seed ${j.params.seed} · cfg ${j.params.cfg_scale}`));
      const tag = el("span", `badge ${j.status}`,
        j.status === "running" ? `${j.progress}%` : (STATUS_LABEL[j.status] || j.status));
      open.append(info, tag);
      open.addEventListener("click", () => onOpen(j.id));

      const rerun = el("button", "icon-btn");
      rerun.type = "button";
      rerun.title = "Re-run with the current settings (detections, camera and features are reused)";
      rerun.disabled = busy !== null || ["queued", "running", "waiting_user"].includes(j.status);
      rerun.innerHTML = '<svg viewBox="0 0 24 24"><path d="M16.02 9.35h4.99M2.98 19.64v-4.99m0 0h4.99m-4.99 0l3.18 3.18a8.25 8.25 0 0013.8-3.7M4.03 9.87a8.25 8.25 0 0113.8-3.7l3.18 3.18" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>';
      if (busy === j.id) rerun.classList.add("spin");
      rerun.addEventListener("click", async (e) => {
        e.stopPropagation();
        busy = j.id;
        setError("");
        refresh();
        try { await onRerun(j.id); } catch (err) { setError(err.message || "Re-run failed"); }
        busy = null;
        refresh();
      });
      row.append(open, rerun);
      list.append(row);
    }
  }

  function setError(msg) {
    $("historyError").textContent = msg;
    show("historyError", !!msg);
  }

  $("historyRefresh").addEventListener("click", refresh);
  return { refresh, setCurrent(id) { current = id; } };
}

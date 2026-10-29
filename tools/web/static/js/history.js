// "Recent runs" card: jobs of this server session, open / re-run.
import { $, show, el, formatRelative, STATUS_LABEL } from "./utils.js";
import * as api from "./api.js";

export function initHistory({ onOpen, onRerun, onDeleted }) {
  const list = $("historyList");
  let current = null;
  let busy = null;
  const selected = new Set();
  const deleteSelected = $("historyDeleteSelected");
  const deleteAll = $("historyDeleteAll");

  function updateActions() {
    $("historySelectedCount").textContent = selected.size;
    deleteSelected.disabled = busy !== null || selected.size === 0;
  }

  async function refresh() {
    let data;
    try { data = await api.getJobs(); } catch { return; }
    const { jobs, now } = data;
    const visibleIds = new Set(jobs.map((j) => j.id));
    for (const id of selected) if (!visibleIds.has(id)) selected.delete(id);
    deleteAll.disabled = busy !== null || data.deletable_count === 0;
    list.innerHTML = "";
    show("historyEmpty", jobs.length === 0);
    for (const j of jobs) {
      const row = el("div", `run${j.id === current ? " current" : ""}`);
      const active = ["queued", "running", "waiting_user", "cancel_requested", "uploading"].includes(j.status);
      const canDelete = !active;
      const check = el("input", "run-select");
      check.type = "checkbox";
      check.checked = selected.has(j.id);
      check.disabled = !canDelete || busy !== null;
      check.setAttribute("aria-label", `Select ${j.filename} for deletion`);
      check.addEventListener("change", () => {
        if (check.checked) selected.add(j.id);
        else selected.delete(j.id);
        updateActions();
      });
      const open = el("button", "run-open");
      open.type = "button";
      const info = el("div", "run-info");
      info.append(el("div", "run-name", j.filename),
        el("div", "run-sub", j.recovered ? "Recovered from disk · settings unavailable"
          : `${formatRelative(j.created, now)} · seed ${j.params.seed} · cfg ${j.params.cfg_scale} · ${j.params.static_camera ? "static camera" : "estimated camera"}`));
      const tag = el("span", `badge ${j.status}`,
        j.status === "running" ? `${j.progress}%` : (STATUS_LABEL[j.status] || j.status));
      open.append(info, tag);
      open.addEventListener("click", () => onOpen(j.id));

      const action = el("button", "icon-btn");
      action.type = "button";
      if (active) {
        action.title = j.status === "cancel_requested" ? "Stopping this run" : "Stop this run";
        action.setAttribute("aria-label", `${action.title}: ${j.filename}`);
        action.disabled = busy !== null || j.status === "cancel_requested";
        action.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor"/></svg>';
        action.addEventListener("click", async (e) => {
          e.stopPropagation();
          if (!window.confirm(`Stop processing ${j.filename}?`)) return;
          busy = `stop:${j.id}`;
          setError("");
          refresh();
          try { await api.cancelJob(j.id); } catch (err) { setError(err.message || "Could not stop this run"); }
          busy = null;
          refresh();
        });
      } else {
        action.title = "Re-run with the current settings";
        action.setAttribute("aria-label", `Re-run ${j.filename}`);
        action.disabled = busy !== null;
        action.innerHTML = '<svg viewBox="0 0 24 24"><path d="M16.02 9.35h4.99M2.98 19.64v-4.99m0 0h4.99m-4.99 0l3.18 3.18a8.25 8.25 0 0013.8-3.7M4.03 9.87a8.25 8.25 0 013.18-3.7l3.18 3.18" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>';
        if (busy === j.id) action.classList.add("spin");
        action.addEventListener("click", async (e) => {
          e.stopPropagation();
          busy = j.id;
          setError("");
          refresh();
          try { await onRerun(j.id); } catch (err) { setError(err.message || "Re-run failed"); }
          busy = null;
          refresh();
        });
      }
      row.append(check, open, action);
      list.append(row);
    }
    updateActions();
  }

  function setError(msg) {
    $("historyError").textContent = msg;
    show("historyError", !!msg);
  }

  $("historyRefresh").addEventListener("click", refresh);
  deleteSelected.addEventListener("click", async () => {
    const ids = [...selected];
    if (!ids.length || !window.confirm(`Delete ${ids.length} selected run${ids.length === 1 ? "" : "s"} and their files?`)) return;
    await removeRuns(ids);
  });
  deleteAll.addEventListener("click", async () => {
    if (!window.confirm("Delete all finished and failed runs and their files? Runs still processing will be kept.")) return;
    await removeRuns();
  });

  async function removeRuns(ids) {
    setError("");
    busy = "delete";
    updateActions();
    try {
      const result = await api.deleteJobs(ids);
      selected.clear();
      if (result.skipped.length) setError(`${result.deleted.length} deleted; ${result.skipped.length} could not be deleted.`);
      onDeleted(result.deleted);
      await refresh();
    } catch (err) {
      setError(err.message || "Could not delete runs");
    } finally {
      busy = null;
      await refresh();
    }
  }

  return { refresh, setCurrent(id) { current = id; } };
}

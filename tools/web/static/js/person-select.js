// Person selection card.
//  - choosing: one row per detected person (thumbnail, colour matching its box in the video
//    preview, frame count); click a row or a box in the preview, then confirm.
//  - locked:   after confirmation (or the timeout fallback) the card keeps showing which
//    person is being reconstructed, for the rest of the job and on its result.
import { $, show, el, personColor } from "./utils.js";
import * as api from "./api.js";

const CHECK_SVG = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>';

export function initPersonSelect(bus) {
  let jobId = null;
  let candidates = [];
  let defaultIdx = null;
  let selected = null;
  let mode = "hidden";      // "hidden" | "choosing" | "locked"
  let submitting = false;

  const list = $("personList");

  function thumb(i, cls) {
    const box = el("span", cls);
    box.style.borderColor = personColor(i);
    const img = el("img");
    img.alt = `Person #${i}`;
    img.loading = "lazy";
    img.src = api.personThumbUrl(jobId, i);
    img.onerror = () => { img.remove(); box.classList.add("no-img"); box.textContent = `${i}`; };
    box.append(img);
    return box;
  }

  function renderChoosing() {
    list.innerHTML = "";
    candidates.forEach((c, i) => {
      const isSel = selected === i;
      const btn = el("button", `person${isSel ? " selected" : ""}`);
      btn.type = "button";
      btn.setAttribute("aria-pressed", isSel ? "true" : "false");
      btn.append(thumb(i, "person-thumb"));
      const text = el("span", "person-text");
      const name = el("span", "person-name");
      const chip = el("span", "person-dot");
      chip.style.background = personColor(i);
      name.append(chip, document.createTextNode(`Person #${i}`));
      if (i === defaultIdx) name.append(el("span", "person-tag", "suggested"));
      text.append(name, el("span", "person-meta num",
        `${c.num_frames} frames · ${(c.num_frames / 30).toFixed(1)}s · frames ${c.start}–${c.end}`));
      btn.append(text);
      const mark = el("span", "person-check");
      mark.innerHTML = CHECK_SVG;
      btn.append(mark);
      btn.addEventListener("click", () => setSelected(i));
      list.append(btn);
    });
    const btn = $("personConfirm");
    btn.disabled = selected === null || submitting;
    btn.textContent = submitting ? "Submitting..."
      : selected === null ? "Pick a person" : `Reconstruct Person #${selected}`;
  }

  function renderLocked(job) {
    list.innerHTML = "";
    const row = el("div", "person locked");
    row.append(thumb(selected, "person-thumb large"));
    const text = el("span", "person-text");
    const name = el("span", "person-name");
    const chip = el("span", "person-dot");
    chip.style.background = "#10b981";
    name.append(chip, document.createTextNode(`Person #${selected}`));
    const c = candidates[selected];
    const how = job.selection_auto ? "auto-selected (no choice was made in time)" : "selected by you";
    text.append(name, el("span", "person-meta", how));
    if (c) text.append(el("span", "person-meta num", `${c.num_frames} frames · frames ${c.start}–${c.end}`));
    row.append(text);
    list.append(row);
  }

  function setSelected(i) {
    if (mode !== "choosing" || !candidates.length) return;
    selected = i;
    renderChoosing();
    bus.emit("person:selected", i);
  }

  $("personConfirm").addEventListener("click", async () => {
    if (selected === null || !jobId || mode !== "choosing") return;
    submitting = true;
    setError("");
    renderChoosing();
    try {
      await api.selectPerson(jobId, selected);
      bus.emit("person:confirmed", selected);
    } catch (e) {
      setError(e.message || "Selection failed");
    } finally {
      submitting = false;
      if (mode === "choosing") renderChoosing();
    }
  });

  function setError(msg) {
    $("personError").textContent = msg;
    show("personError", !!msg);
  }

  function setMode(m, job) {
    mode = m;
    const card = $("personCard");
    card.classList.toggle("attention", m === "choosing");
    $("personTitle").textContent = m === "locked" ? "Reconstructing" : "Select the person";
    show("personConfirm", m === "choosing");
    show("personCount", m === "choosing");
    if (m !== "locked") show("personReselect", false);
    show(card, m !== "hidden");
    if (m === "choosing") renderChoosing();
    if (m === "locked") renderLocked(job);
  }

  $("personReselect").addEventListener("click", () => bus.emit("person:reselect"));

  return {
    /** Called on every job poll. */
    update(job) {
      const cands = Array.isArray(job.candidates) ? job.candidates : [];
      if (!cands.length) {
        if (mode !== "hidden") { setMode("hidden"); bus.emit("person:candidates", { candidates: [], selected: null }); }
        return;
      }
      const newJob = jobId !== job.id;
      if (newJob) {
        jobId = job.id;
        candidates = cands;
        defaultIdx = job.default_person ?? null;
        selected = defaultIdx;
        setError("");
      }
      if (job.status === "waiting_user") {
        if (mode !== "choosing") {
          $("personCount").textContent = `${cands.length} people detected — click a person below or their box in the video.`;
          setMode("choosing", job);
          bus.emit("person:candidates", { candidates, selected });
        }
      } else if (job.selected_person != null) {
        const changed = mode !== "locked" || selected !== job.selected_person;
        selected = job.selected_person;
        if (changed) {
          setMode("locked", job);
          bus.emit("person:locked", { candidate: candidates[selected], index: selected });
        }
        $("personTitle").textContent = job.status === "done" ? "Reconstructed person"
          : job.status === "failed" ? "Selected person" : "Reconstructing";
      }
      // Offer a re-pick once the job has finished (needs a re-run of detection).
      show("personReselect", mode === "locked" && ["done", "failed"].includes(job.status));
    },
    select: setSelected,
    reset() {
      jobId = null; candidates = []; selected = null; defaultIdx = null;
      setMode("hidden");
      bus.emit("person:candidates", { candidates: [], selected: null });
    },
  };
}

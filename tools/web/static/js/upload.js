// "Video" card: drag & drop / browse, shows the chosen file.
import { $, show, formatBytes } from "./utils.js";

const VIDEO_RE = /\.(mp4|mov|avi|mkv|webm|m4v)$/i;

export function initUpload(bus) {
  const dz = $("dropzone");
  const input = $("fileInput");
  let file = null;

  const pick = (f) => {
    if (!f) return;
    if (!f.type.startsWith("video/") && !VIDEO_RE.test(f.name)) {
      setError("Please choose a video file (mp4 / mov / avi / mkv / webm).");
      return;
    }
    setError("");
    file = f;
    $("fileName").textContent = f.name;
    $("fileSize").textContent = formatBytes(f.size);
    show("fileRow", true);
    dz.classList.add("compact");
    bus.emit("file:selected", f);
  };

  const clear = () => {
    file = null;
    show("fileRow", false);
    dz.classList.remove("compact");
    bus.emit("file:cleared");
  };

  const setError = (msg) => {
    $("uploadError").textContent = msg;
    show("uploadError", !!msg);
  };

  dz.addEventListener("click", () => input.click());
  dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") input.click(); });
  input.addEventListener("change", (e) => { pick(e.target.files[0]); e.target.value = ""; });
  ["dragenter", "dragover"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => pick(e.dataTransfer.files[0]));
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());
  $("fileRemove").addEventListener("click", (e) => { e.stopPropagation(); clear(); });

  return { getFile: () => file, clear, setError };
}

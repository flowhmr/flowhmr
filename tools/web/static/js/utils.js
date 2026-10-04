export const $ = (id) => document.getElementById(id);

export function show(el, visible = true) {
  (typeof el === "string" ? $(el) : el).classList.toggle("hidden", !visible);
}

export function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

export function formatSeconds(s) {
  if (s == null || !isFinite(s)) return "";
  if (s < 60) return `${s.toFixed(1)}s`;
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
}

export function formatRelative(tsSeconds, nowSeconds) {
  const d = Math.max(0, (nowSeconds ?? Date.now() / 1000) - tsSeconds);
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.floor(d / 60)} min ago`;
  if (d < 86400) return `${Math.floor(d / 3600)} h ago`;
  return new Date(tsSeconds * 1000).toLocaleString();
}

export function formatBytes(n) {
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export const STATUS_LABEL = {
  uploading: "Uploading", queued: "Queued", running: "Running",
  waiting_user: "Select person", done: "Done", failed: "Failed",
};

/** Distinct colours for person boxes; index = candidate index. */
export const PERSON_COLORS = ["#3b82f6", "#f59e0b", "#8b5cf6", "#ec4899", "#14b8a6", "#ef4444", "#84cc16", "#0ea5e9"];
export const personColor = (i) => PERSON_COLORS[i % PERSON_COLORS.length];

/** Minimal event bus shared by the page modules. */
export function createBus() {
  const handlers = {};
  return {
    on(type, fn) { (handlers[type] ||= []).push(fn); },
    emit(type, payload) { (handlers[type] || []).forEach((fn) => fn(payload)); },
  };
}

// "Video preview" card: plays the local file before upload / the job video afterwards and
// overlays person boxes on a canvas (all candidates while choosing, the tracked person after).
import { $, show, personColor } from "./utils.js";

export function initPreview(bus) {
  const video = $("previewVideo");
  const canvas = $("previewCanvas");
  const ctx = canvas.getContext("2d");
  let localUrl = null;
  let candidates = [];      // [{frames, boxes, ...}] while waiting for a pick
  let selected = null;
  let track = null;         // {start, end, boxes} of the reconstructed person
  let locked = null;        // {candidate, index} once the person is confirmed
  let pickable = false;

  let baseSrc = null;       // source without the retry cache-buster
  let retries = 0;
  let retryTimer = null;

  function setSource(src, { local = false } = {}) {
    if (localUrl && src !== localUrl) { URL.revokeObjectURL(localUrl); localUrl = null; }
    if (local) localUrl = src;
    clearTimeout(retryTimer);
    baseSrc = src;
    retries = 0;
    video.src = src;
    video.load();
    show("previewBox", true);
  }

  // A job video can fail to load if it was requested while the server was still writing
  // it; retry a few times with a cache-buster so the browser does not reuse the bad copy.
  video.addEventListener("error", () => {
    if (!baseSrc || baseSrc === localUrl || retries >= 10) return;
    retries += 1;
    clearTimeout(retryTimer);
    retryTimer = setTimeout(() => {
      video.src = `${baseSrc}${baseSrc.includes("?") ? "&" : "?"}r=${Date.now()}`;
      video.load();
    }, 1500);
  });
  video.addEventListener("loadeddata", () => { retries = 0; });

  function clear() {
    if (localUrl) { URL.revokeObjectURL(localUrl); localUrl = null; }
    clearTimeout(retryTimer);
    baseSrc = null;
    video.removeAttribute("src");
    video.load();
    candidates = []; track = null; selected = null; locked = null;
    show("previewBox", false);
    $("previewTitle").textContent = "";
  }

  // Box of a candidate at a (30 fps) frame, linearly interpolated between its detections.
  function boxAt(c, f) {
    const fr = c.frames;
    if (!fr || !fr.length || f < fr[0] - 2 || f > fr[fr.length - 1] + 2) return null;
    if (f <= fr[0]) return c.boxes[0];
    if (f >= fr[fr.length - 1]) return c.boxes[fr.length - 1];
    let lo = 0, hi = fr.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (fr[mid] === f) return c.boxes[mid];
      if (fr[mid] < f) lo = mid + 1; else hi = mid - 1;
    }
    const a = Math.max(0, hi), b = Math.min(fr.length - 1, lo);
    if (a === b) return c.boxes[a];
    const t = (f - fr[a]) / (fr[b] - fr[a]);
    return c.boxes[a].map((v, k) => v + (c.boxes[b][k] - v) * t);
  }

  // Map video pixel coords -> canvas CSS coords. The canvas shares the video's top-left
  // corner (it only stops above the native control bar), and the video is object-fit:
  // contain, so the mapping is computed from the <video> box, not the canvas box.
  function layout() {
    const cw = video.clientWidth, ch = video.clientHeight;
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh || !cw || !ch) return null;
    const s = Math.min(cw / vw, ch / vh);
    return { s, ox: (cw - vw * s) / 2, oy: (ch - vh * s) / 2, cw, ch };
  }

  function drawBox(L, box, color, label, strong) {
    const [x1, y1, x2, y2] = box;
    const rx = x1 * L.s + L.ox, ry = y1 * L.s + L.oy, rw = (x2 - x1) * L.s, rh = (y2 - y1) * L.s;
    ctx.strokeStyle = color;
    ctx.lineWidth = strong ? 3 : 2;
    ctx.setLineDash(strong ? [] : [6, 4]);
    ctx.strokeRect(rx, ry, rw, rh);
    ctx.setLineDash([]);
    if (label) {
      ctx.font = "600 12px Inter, -apple-system, sans-serif";
      const tw = ctx.measureText(label).width + 10;
      ctx.fillStyle = color;
      ctx.fillRect(rx - (strong ? 1.5 : 1), Math.max(0, ry - 20), tw, 20);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, rx + 4, Math.max(14, ry - 6));
    }
  }

  function draw() {
    requestAnimationFrame(draw);
    const dpr = window.devicePixelRatio || 1;
    const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    const L = layout();
    if (!L) return;
    const f = Math.floor(video.currentTime * 30 + 1e-3);
    if (candidates.length) {
      candidates.forEach((c, i) => {
        const b = boxAt(c, f);
        if (b) drawBox(L, b, selected === i ? "#10b981" : personColor(i), `#${i}`, selected === i);
      });
    } else if (track && f >= track.start && f <= track.end) {
      const b = track.boxes[f - track.start];
      if (b) drawBox(L, b, "#10b981", locked ? `#${locked.index} ✓` : null, true);
    } else if (!track && locked && locked.candidate) {
      const b = boxAt(locked.candidate, f);
      if (b) drawBox(L, b, "#10b981", `#${locked.index} ✓`, true);
    }
  }
  requestAnimationFrame(draw);

  // Click on a box to pick that person.
  canvas.addEventListener("click", (e) => {
    if (!pickable || !candidates.length) return;
    const L = layout();
    if (!L) return;
    const r = canvas.getBoundingClientRect();
    const x = (e.clientX - r.left - L.ox) / L.s, y = (e.clientY - r.top - L.oy) / L.s;
    const f = Math.floor(video.currentTime * 30 + 1e-3);
    let best = null, bestArea = Infinity;
    candidates.forEach((c, i) => {
      const b = boxAt(c, f);
      if (!b || x < b[0] || x > b[2] || y < b[1] || y > b[3]) return;
      const area = (b[2] - b[0]) * (b[3] - b[1]);
      if (area < bestArea) { best = i; bestArea = area; }
    });
    if (best !== null) bus.emit("preview:pick", best);
  });

  bus.on("person:candidates", ({ candidates: c, selected: s }) => {
    candidates = c || [];
    selected = s;
    locked = null;
    pickable = candidates.length > 0;
    canvas.classList.toggle("pickable", pickable);
    if (pickable && video.paused) video.play().catch(() => {});
  });
  bus.on("person:selected", (i) => { selected = i; });
  bus.on("person:locked", (l) => {
    locked = l;
    candidates = [];
    pickable = false;
    canvas.classList.remove("pickable");
  });

  return {
    showLocal(file) { track = null; setSource(URL.createObjectURL(file), { local: true }); $("previewTitle").textContent = file.name; },
    showJob(url, name) { if (baseSrc !== url) setSource(url); $("previewTitle").textContent = name; },
    setTrack(t) { track = t; },
    clear,
    video,
  };
}

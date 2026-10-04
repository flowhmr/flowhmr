// "Result" card: three.js viewer for the generated SMPL-H mesh sequence, playback controls
// and synchronisation with the source video.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { $, show } from "./utils.js";
import * as api from "./api.js";

function float16ToFloat32(src) {
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    const h = src[i];
    const s = (h & 0x8000) ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const f = h & 0x3ff;
    out[i] = e === 0 ? s * 5.960464477539063e-8 * f
      : e === 31 ? (f ? NaN : s * Infinity)
      : s * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  return out;
}

function checkerTexture(renderer, floorSize, cell) {
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  const g = c.getContext("2d");
  g.fillStyle = "#f4f5f7"; g.fillRect(0, 0, 256, 256);
  g.fillStyle = "#dcdfe4"; g.fillRect(0, 0, 128, 128); g.fillRect(128, 128, 128, 128);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(floorSize / (2 * cell), floorSize / (2 * cell));
  t.anisotropy = renderer.capabilities.getMaxAnisotropy();
  return t;
}

export function initViewer({ video }) {
  const container = $("viewer");
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);

  const BG = 0xfbfbfc;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(BG);
  scene.fog = new THREE.Fog(BG, 10, 40);  // near/far follow the camera distance (see tick)

  const camera = new THREE.PerspectiveCamera(40, 1, 0.05, 200);
  camera.position.set(2.6, 1.6, 3.6);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.9, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 0.8;
  controls.maxDistance = 25;

  // Custom wheel zoom. The stock OrbitControls (r160) divides the wheel delta by
  // (devicePixelRatio | 0), which is 0 at browser zoom < 100% -> infinite jumps, and it
  // applies large trackpad/free-spin deltas in a single step. Here each wheel event only
  // moves a target distance by a bounded factor, and the camera eases toward it per frame.
  controls.enableZoom = false;
  const zoom = { target: camera.position.distanceTo(controls.target) };
  renderer.domElement.addEventListener("wheel", (e) => {
    e.preventDefault();
    const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? container.clientHeight : 1;
    const px = Math.max(-120, Math.min(120, e.deltaY * unit));  // cap per event (~1 notch)
    zoom.target = THREE.MathUtils.clamp(zoom.target * Math.pow(1.0015, px),
      controls.minDistance, controls.maxDistance);
  }, { passive: false });
  const zoomDir = new THREE.Vector3();
  function applyZoom() {
    const dist = camera.position.distanceTo(controls.target);
    if (Math.abs(dist - zoom.target) < 1e-4) return;
    const next = dist + (zoom.target - dist) * 0.2;
    zoomDir.subVectors(camera.position, controls.target).normalize();
    camera.position.copy(controls.target).addScaledVector(zoomDir, next);
  }
  window.__viewer = { camera, controls, zoom };  // for debugging / automated tests

  // Soft studio lighting: sky/ground ambient, a front-top key light that casts the floor
  // shadow (it follows the body, see setFrame), a cool fill from the side and a rim light.
  scene.add(new THREE.HemisphereLight(0xf5f7ff, 0xc8ccd8, 1.4));
  const sun = new THREE.DirectionalLight(0xffffff, 1.9);
  const SUN_OFFSET = new THREE.Vector3(-2.2, 6, 3.6);  // front-left: shadow falls back-right
  sun.position.copy(SUN_OFFSET);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.bias = -0.0004;
  sun.shadow.normalBias = 0.02;
  const fill = new THREE.DirectionalLight(0xe6ebff, 0.55);
  fill.position.set(4, 2, 2);
  const rim = new THREE.DirectionalLight(0xffffff, 0.7);
  rim.position.set(1.5, 3, -5);
  scene.add(fill, rim);
  Object.assign(sun.shadow.camera, { left: -4, right: 4, top: 4, bottom: -4, near: 0.5, far: 20 });
  scene.add(sun, sun.target);

  // Checkerboard floor (0.5 m squares) fading into the background with the fog,
  // plus a transparent plane on top that only draws the body's shadow.
  const FLOOR_SIZE = 200;
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(FLOOR_SIZE, FLOOR_SIZE),
    new THREE.MeshBasicMaterial({ map: checkerTexture(renderer, FLOOR_SIZE, 0.5) }));
  floor.rotation.x = -Math.PI / 2;
  const shadowPlane = new THREE.Mesh(new THREE.PlaneGeometry(FLOOR_SIZE, FLOOR_SIZE),
    new THREE.ShadowMaterial({ color: 0x4b5468, opacity: 0.34 }));
  shadowPlane.rotation.x = -Math.PI / 2;
  shadowPlane.position.y = 0.001;
  shadowPlane.receiveShadow = true;
  scene.add(floor, shadowPlane);

  // Matte lavender-blue body.
  const bodyMat = new THREE.MeshStandardMaterial({ color: 0xb4c0f2, roughness: 0.75, metalness: 0.0 });
  let body = null, trail = null, marker = null;
  const state = {
    jobId: null, meta: null, verts: null, frame: 0, playing: false, speed: 1, lastT: 0, acc: 0,
    prevRoot: new THREE.Vector3(),
  };

  function resize() {
    const w = container.clientWidth, h = container.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / Math.max(h, 1);
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(container);
  resize();

  function buildTrail(meta) {
    if (trail) { scene.remove(trail); trail.geometry.dispose(); }
    if (marker) scene.remove(marker);
    const pts = meta.root_trajectory.map(([x, z]) => new THREE.Vector3(x, 0.005, z));
    trail = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({ color: 0x2f6fed, transparent: true, opacity: 0.55 }));
    marker = new THREE.Mesh(new THREE.RingGeometry(0.08, 0.11, 32),
      new THREE.MeshBasicMaterial({ color: 0x2f6fed, side: THREE.DoubleSide }));
    marker.rotation.x = -Math.PI / 2;
    scene.add(trail, marker);
    trail.visible = marker.visible = $("optTrail").checked;
  }

  const frameRoot = (f) => {
    const [x, z] = state.meta.root_trajectory[f];
    return new THREE.Vector3(x, 0, z);
  };

  function setFrame(f) {
    const m = state.meta;
    if (!m || !body) return;
    f = Math.max(0, Math.min(m.num_frames - 1, f));
    state.frame = f;
    const n = m.num_vertices * 3;
    const pos = body.geometry.attributes.position;
    pos.array.set(state.verts.subarray(f * n, (f + 1) * n));
    pos.needsUpdate = true;
    body.geometry.computeVertexNormals();

    const root = frameRoot(f);
    if (marker) marker.position.set(root.x, 0.006, root.z);
    sun.position.copy(root).add(SUN_OFFSET);
    sun.target.position.copy(root);
    if ($("optFollow").checked) {
      const delta = root.clone().sub(state.prevRoot);
      camera.position.add(delta);
      controls.target.add(delta);
    }
    state.prevRoot.copy(root);
    $("scrub").value = f;
    $("frameLabel").textContent = `${f + 1} / ${m.num_frames}`;
  }

  function resetView() {
    const m = state.meta;
    if (!m) return;
    const root = frameRoot(state.frame);
    const height = Math.max(1.2, m.bbox_max[1] - m.bbox_min[1]);
    controls.target.set(root.x, height * 0.5, root.z);
    camera.position.set(root.x + 2.4, height * 0.85 + 0.4, root.z + 3.4);
    zoom.target = camera.position.distanceTo(controls.target);
    state.prevRoot.copy(root);
    controls.update();
  }

  // On loop: move camera + target back with the body, keeping the user's zoom and angle.
  function recenterOnFrame(f) {
    const root = frameRoot(f);
    const delta = root.clone().sub(state.prevRoot);
    camera.position.add(delta);
    controls.target.add(delta);
    state.prevRoot.copy(root);
  }

  // ---------------------------------------------------------------- video sync
  const videoTimeFor = (f) => ((state.meta.track_start || 0) + f + 0.5) / state.meta.fps;
  function syncVideo(force) {
    if (!state.meta || !$("optSync").checked || !video.src || video.readyState < 1) return;
    const t = videoTimeFor(state.frame);
    if (state.playing) {
      if (video.paused) video.play().catch(() => {});
      video.playbackRate = state.speed;
      if (force || Math.abs(video.currentTime - t) > 0.15) video.currentTime = t;
    } else {
      if (!video.paused) video.pause();
      if (force || Math.abs(video.currentTime - t) > 0.02) video.currentTime = t;
    }
  }
  function setPlaying(p) {
    state.playing = p;
    $("iconPlay").classList.toggle("hidden", p);
    $("iconPause").classList.toggle("hidden", !p);
    syncVideo(true);
  }

  $("btnPlay").onclick = () => {
    if (!state.meta) return;
    if (!state.playing && state.frame >= state.meta.num_frames - 1) setFrame(0);
    setPlaying(!state.playing);
  };
  $("scrub").addEventListener("input", (e) => {
    setPlaying(false);
    setFrame(parseInt(e.target.value, 10));
    syncVideo(true);
  });
  $("speed").onchange = (e) => { state.speed = parseFloat(e.target.value); syncVideo(true); };
  $("btnReset").onclick = resetView;
  $("optTrail").onchange = (e) => { if (trail) trail.visible = marker.visible = e.target.checked; };
  $("optSync").onchange = (e) => { if (!e.target.checked) video.pause(); else syncVideo(true); };
  window.addEventListener("keydown", (e) => {
    if (!state.meta || ["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
    if (e.code === "Space") { e.preventDefault(); $("btnPlay").click(); }
    if (e.code === "ArrowRight") { setPlaying(false); setFrame(state.frame + 1); syncVideo(true); }
    if (e.code === "ArrowLeft") { setPlaying(false); setFrame(state.frame - 1); syncVideo(true); }
  });

  function tick(t) {
    requestAnimationFrame(tick);
    const dt = state.lastT ? (t - state.lastT) / 1000 : 0;
    state.lastT = t;
    if (state.playing && state.meta) {
      state.acc += dt * state.meta.fps * state.speed;
      if (state.acc >= 1) {
        const steps = Math.floor(state.acc);
        state.acc -= steps;
        let next = state.frame + steps;
        if (next >= state.meta.num_frames) {
          next = 0;
          if ($("optFollow").checked) recenterOnFrame(0);
          syncVideo(true);
        }
        setFrame(next);
        syncVideo(false);
      }
    }
    applyZoom();
    controls.update();
    const d = camera.position.distanceTo(controls.target);
    scene.fog.near = d + 6;
    scene.fog.far = d + 30;
    renderer.render(scene, camera);
  }
  requestAnimationFrame(tick);

  // ---------------------------------------------------------------- public API
  return {
    async load(jobId, job) {
      const [meta, vbuf, fbuf] = await Promise.all([
        api.getMeta(jobId), api.getVertices(jobId), api.getFaces(jobId),
      ]);
      state.jobId = jobId;
      state.meta = meta;
      state.verts = float16ToFloat32(new Uint16Array(vbuf));
      if (body) { scene.remove(body); body.geometry.dispose(); }
      const geom = new THREE.BufferGeometry();
      geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(meta.num_vertices * 3), 3));
      geom.setIndex(new THREE.BufferAttribute(new Uint32Array(fbuf), 1));
      body = new THREE.Mesh(geom, bodyMat);
      body.castShadow = true;
      body.frustumCulled = false;
      scene.add(body);
      buildTrail(meta);

      $("scrub").max = meta.num_frames - 1;
      $("dlNpz").href = api.npzUrl(jobId);
      $("dlFbx").href = api.fbxUrl(jobId);
      show("dlFbx", !!(job && job.has_fbx));
      show("dlNpz", true);
      show("viewerEmpty", false);
      show("viewerTools", true);
      show("controls", true);
      $("resultInfo").textContent = `${meta.num_frames} frames · ${(meta.num_frames / meta.fps).toFixed(1)}s · 30 fps`;
      setFrame(0);
      resetView();
      setPlaying(true);
      return meta;
    },
    clear() {
      state.jobId = null; state.meta = null;
      if (body) { scene.remove(body); body.geometry.dispose(); body = null; }
      if (trail) { scene.remove(trail); trail = null; }
      if (marker) { scene.remove(marker); marker = null; }
      setPlaying(false);
      show("viewerEmpty", true);
      show("viewerTools", false);
      show("dlNpz", false);
      show("dlFbx", false);
      show("controls", false);
      $("resultInfo").textContent = "";
    },
    setStatus(text) { if (!state.meta) $("viewerEmpty").textContent = text; },
    get jobId() { return state.jobId; },
  };
}

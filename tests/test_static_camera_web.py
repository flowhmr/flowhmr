"""Web camera/SAM parameters, cache isolation and UI checks without models."""

import io
import importlib.util
import copy
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

# SAM exposes a top-level ``tools`` package too. Load the web module by path
# so test discovery order cannot resolve it to the third-party package.
_server_path = Path(__file__).resolve().parents[1] / "tools/web/server.py"
_server_spec = importlib.util.spec_from_file_location("_flowhmr_web_server_tests", _server_path)
server = importlib.util.module_from_spec(_server_spec)
sys.modules[_server_spec.name] = server
_server_spec.loader.exec_module(server)


CAMERA_CACHES = (
    "input_vggt_camera.npz", "input_sam3d_feat.pt",
    "input_static_camera.npz", "input_sam3d_feat_static.pt",
)


class StaticCameraWebTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.args = SimpleNamespace(work_dir=self.directory.name, ckpt="flowhmr_latest.ckpt",
                                    model_cfg="config.yml", device="cpu", host="127.0.0.1", port=8080)
        self.engine = SimpleNamespace(device="cpu", pipeline=SimpleNamespace(validation_steps=20))
        for name, value in (("ARGS", self.args), ("ENGINE", self.engine), ("JOBS", {})):
            patcher = patch.object(server, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(server, "enqueue")
        self.enqueue = patcher.start()
        self.addCleanup(patcher.stop)
        self.client = server.app.test_client()

    def make_source(self, static=False):
        job = server.new_job("walk.mp4", server.parse_params({"static_camera": static}), ".mp4")
        job["status"] = "done"
        Path(server.input_video(job["id"])).write_bytes(b"uploaded video")
        sub = Path(server.job_dir(job["id"])) / server.SUB
        sub.mkdir()
        for name in (*CAMERA_CACHES, "input_bbox.npz", "input_30fps.mp4", "input_seed0_origin.npz"):
            (sub / name).write_bytes(name.encode())
        # An interrupted camera extraction may leave a scratch directory.
        (sub / "input_vggt_camera.npz.frames").mkdir()
        return job, sub

    def rerun(self, job, params):
        response = self.client.post(f"/api/jobs/{job['id']}/rerun", json=params)
        self.assertEqual(response.status_code, 200, response.get_json())
        new_id = response.get_json()["job_id"]
        return server.JOBS[new_id], Path(server.job_dir(new_id)) / server.SUB

    def test_upload_form_and_info_round_trip_camera_mode(self):
        defaults = self.client.get("/api/info").get_json()["defaults"]
        self.assertFalse(defaults["static_camera"])
        self.assertEqual(defaults["sam_chunk_size"], 16)
        self.assertNotIn("sam_hand_batch_size", defaults)
        response = self.client.post("/api/upload", data={
            "video": (io.BytesIO(b"video"), "walk.mp4"), "static_camera": "true", "vggt_interval": "17",
            "sam_chunk_size": "32",
        }, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        job_id = response.get_json()["job_id"]
        job = self.client.get(f"/api/jobs/{job_id}").get_json()
        self.assertTrue(job["params"]["static_camera"])
        self.assertEqual(job["params"]["vggt_interval"], 17)
        self.assertEqual(job["params"]["sam_chunk_size"], 32)
        camera = next(stage for stage in job["stages"] if stage["name"] == "camera")
        self.assertEqual(camera["label"], "Static camera")
        self.assertIn("skipped", camera["detail"])
        self.enqueue.assert_called_once()

    def test_same_mode_rerun_preserves_compatible_caches_and_old_job_default(self):
        for static in (False, True):
            with self.subTest(static=static):
                source, _ = self.make_source(static)
                if not static:
                    source["params"].pop("static_camera")  # pre-option jobs still default to dynamic
                source["params"].pop("sam_chunk_size")
                source["params"]["sam_hand_batch_size"] = 8  # retired fields are ignored
                job, sub = self.rerun(source, {"seed": 8})
                self.assertEqual(job["params"]["static_camera"], static)
                self.assertEqual(job["params"]["sam_chunk_size"], 16)
                self.assertNotIn("sam_hand_batch_size", job["params"])
                for name in CAMERA_CACHES:
                    self.assertEqual((sub / name).read_bytes(), name.encode())
                self.assertTrue((sub / "input_bbox.npz").exists())
                self.assertFalse((sub / "input_seed0_origin.npz").exists())

    def test_switching_camera_modes_cannot_reuse_features_with_other_intrinsics(self):
        for static in (False, True):
            with self.subTest(static=static):
                source, source_sub = self.make_source(static)
                job, sub = self.rerun(source, {"static_camera": not static})
                self.assertEqual(job["params"]["static_camera"], not static)
                for name in CAMERA_CACHES:
                    self.assertFalse((sub / name).exists(), name)
                    self.assertTrue((source_sub / name).exists(), name)
                self.assertTrue((sub / "input_bbox.npz").exists())
                self.assertTrue((sub / "input_30fps.mp4").exists())

    def test_frame_limit_and_person_changes_clear_both_camera_cache_names(self):
        for change, keeps_bbox in (({"max_frames": 300}, True), ({"reselect_person": True}, False)):
            with self.subTest(change=change):
                source, _ = self.make_source(True)
                _, sub = self.rerun(source, change)
                for name in CAMERA_CACHES:
                    self.assertFalse((sub / name).exists(), name)
                self.assertEqual((sub / "input_bbox.npz").exists(), keeps_bbox)
                self.assertTrue((sub / "input_30fps.mp4").exists())

    def test_stride_change_invalidates_dynamic_features_but_keeps_static_features(self):
        for static in (False, True):
            with self.subTest(static=static):
                source, _ = self.make_source(static)
                _, sub = self.rerun(source, {"vggt_interval": 12})
                self.assertEqual((sub / "input_sam3d_feat.pt").exists(), static)
                self.assertEqual((sub / "input_vggt_camera.npz").exists(), static)
                self.assertTrue((sub / "input_sam3d_feat_static.pt").exists())
                self.assertTrue((sub / "input_static_camera.npz").exists())

    def test_sam_chunk_changes_clear_only_features_and_preserve_source(self):
        for static in (False, True):
            for change in ({"sam_chunk_size": 4}, {"sam_chunk_size": 32}):
                with self.subTest(static=static, change=change):
                    source, source_sub = self.make_source(static)
                    job, sub = self.rerun(source, change)
                    for key, value in change.items():
                        self.assertEqual(job["params"][key], value)
                    for name in ("input_sam3d_feat.pt", "input_sam3d_feat_static.pt"):
                        self.assertFalse((sub / name).exists())
                        self.assertTrue((source_sub / name).exists())
                    for name in ("input_bbox.npz", "input_30fps.mp4", "input_vggt_camera.npz",
                                 "input_static_camera.npz"):
                        self.assertEqual((sub / name).read_bytes(), name.encode())

    def test_sam_chunk_parameter_rejects_nonpositive_and_noninteger_values(self):
        for value in [0, -1, True, 2.5, 2.0, "1.5", "16oops", [], {}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                server.parse_params({"sam_chunk_size": value})
        self.assertEqual(server.parse_params({"sam_chunk_size": "1"})["sam_chunk_size"], 1)
        self.assertNotIn("sam_hand_batch_size", server.parse_params({"sam_hand_batch_size": "3"}))
        source, _ = self.make_source()
        response = self.client.post(f"/api/jobs/{source['id']}/rerun", json={"sam_chunk_size": 0})
        self.assertEqual(response.status_code, 400)
        self.assertIn("sam_chunk_size", response.get_json()["error"])
        self.enqueue.assert_not_called()

    def test_unchanged_custom_sam_chunk_reuses_feature_caches(self):
        source, _ = self.make_source()
        source["params"]["sam_chunk_size"] = 32
        job, sub = self.rerun(source, {"seed": 1})
        self.assertEqual(job["params"]["sam_chunk_size"], 32)
        for name in CAMERA_CACHES:
            self.assertTrue((sub / name).exists(), name)

    def test_run_job_passes_static_camera_to_engine_and_completes_stages(self):
        for static, chunk_size in ((False, 16), (True, 32)):
            with self.subTest(static=static, chunk_size=chunk_size):
                job = server.new_job("walk.mp4", server.parse_params({
                    "static_camera": static, "manual_select": False,
                    "sam_chunk_size": chunk_size,
                }), ".mp4")
                if not static:  # queued jobs saved before these options still work
                    job["params"].pop("sam_chunk_size")
                    job["params"]["sam_hand_batch_size"] = 8
                Path(server.input_video(job["id"])).write_bytes(b"video")

                def process_video(video, output_dir, **kwargs):
                    self.assertEqual(kwargs["static_camera"], static)
                    self.assertEqual(kwargs["sam_chunk_size"], chunk_size)
                    self.assertNotIn("sam_hand_batch_size", kwargs)
                    kwargs["progress_callback"]("camera", 100, "Camera estimation skipped" if static else "Camera ready")
                    sub = Path(output_dir) / server.SUB
                    sub.mkdir()
                    (sub / "input_seed0_origin.npz").write_bytes(b"motion")
                    return str(sub)

                def mesh(_motion, directory):
                    Path(directory).mkdir()
                    return {"num_frames": 2}

                self.engine.process_video = Mock(side_effect=process_video)
                with patch.object(server, "export_mesh_sequence", side_effect=mesh):
                    server.run_job(job["id"])
                self.assertEqual(job["status"], "done")
                self.assertTrue(all(stage["status"] == "completed" for stage in job["stages"]))
                self.assertEqual(self.engine.process_video.call_args.kwargs["vggt_frame_interval"], 1)

    def test_engine_loading_does_not_warm_vggt(self):
        pipeline_module = ModuleType("flowhmr.pipeline.v2m_inference")
        pipeline_module.V2MInferenceEngine = SimpleNamespace(build=Mock(return_value=self.engine))
        camera_module = ModuleType("flowhmr.utils.runtime_vggt_camera")
        camera_module.DEFAULT_VGGT_OMEGA_CKPT = "missing-vggt.pt"
        camera_module._load_vggt_model = Mock(side_effect=AssertionError("VGGT must load lazily"))
        with patch.dict(sys.modules, {
            pipeline_module.__name__: pipeline_module, camera_module.__name__: camera_module,
        }):
            self.assertIs(server.load_engine(self.args), self.engine)
        pipeline_module.V2MInferenceEngine.build.assert_called_once()
        camera_module._load_vggt_model.assert_not_called()

    def test_server_startup_allows_missing_vggt_weights(self):
        def exists(path):
            return "vggt" not in str(path).lower()

        with patch.object(server, "parse_args", return_value=self.args), patch.object(
            server.os.path, "exists", side_effect=exists,
        ), patch.object(server, "load_engine", return_value=self.engine) as load, patch.object(
            server.threading, "Thread",
        ), patch.object(server.os, "chdir"), patch.object(server.app, "run") as run:
            server.main()
        load.assert_called_once()
        run.assert_called_once()

    def test_restore_interrupted_stages_settles_selection_wait_and_freezes_time(self):
        for job_status in ("running", "waiting_user", "cancel_requested"):
            for stage_status in ("running", "waiting_user"):
                with self.subTest(job_status=job_status, stage_status=stage_status):
                    job = server.new_job("walk.mp4", server.default_params(), ".mp4")
                    job["status"] = job_status
                    stage = job["stages"][server.STAGE_INDEX["detection"]]
                    stage.update(status=stage_status, start=100.0)
                    if stage_status == "waiting_user":
                        stage.update(paused_at=130.0, paused_seconds=5.0)
                    server._persist_job_locked(job)
                    with patch.object(server.time, "time", return_value=180.0):
                        server.restore_jobs()
                    restored = server.JOBS[job["id"]]
                    expected_status = "cancelled" if job_status == "cancel_requested" else "failed"
                    self.assertEqual(restored["status"], expected_status)
                    self.assertEqual(restored["finished"], 180.0)
                    stage = restored["stages"][server.STAGE_INDEX["detection"]]
                    self.assertEqual(stage["status"], expected_status)
                    self.assertEqual(stage["end"], 180.0)
                    self.assertNotIn("paused_at", stage)
                    self.assertEqual(stage.get("paused_seconds", 0.0), 55.0 if stage_status == "waiting_user" else 0.0)
                    elapsed = stage["end"] - stage["start"] - stage.get("paused_seconds", 0.0)
                    self.assertEqual(elapsed, 25.0 if stage_status == "waiting_user" else 80.0)

    def test_restore_finished_history_keeps_existing_stage_timing(self):
        for status in ("done", "failed", "cancelled"):
            with self.subTest(status=status):
                job = server.new_job("walk.mp4", server.default_params(), ".mp4")
                job.update(status=status, finished=160.0)
                job["stages"][server.STAGE_INDEX["detection"]].update(
                    status="completed", start=100.0, end=150.0, paused_seconds=20.0,
                )
                stages = copy.deepcopy(job["stages"])
                server._persist_job_locked(job)
                with patch.object(server.time, "time", return_value=200.0):
                    server.restore_jobs()
                restored = server.JOBS[job["id"]]
                self.assertEqual(restored["status"], status)
                self.assertEqual(restored["finished"], 160.0)
                self.assertEqual(restored["stages"], stages)

    @unittest.skipUnless(shutil.which("node"), "Node is required for the browser parameter test")
    def test_page_toggle_disables_stride_and_keeps_value_across_apply_and_click(self):
        script = r'''
import assert from "node:assert/strict";
import fs from "node:fs";
class Element {
  constructor() { this.attributes = new Map(); this.listeners = []; this.value = ""; this.disabled = false; }
  getAttribute(key) { return this.attributes.get(key); }
  setAttribute(key, value) { this.attributes.set(key, value); }
  addEventListener(event, callback) { if (event === "click") this.listeners.push(callback); }
  click() { for (const callback of this.listeners) callback(); }
}
globalThis.elements = Object.fromEntries(["optManual", "optPostprocess", "optStaticCamera", "optAutoFrames", "optSeed", "optCfg", "optVggt", "optSteps", "optMaxFrames", "optSamChunk"].map(id => [id, new Element()]));
const source = fs.readFileSync(process.argv[2], "utf8").replace('import { $ } from "./utils.js";', 'const $ = id => globalThis.elements[id];');
const { initParams } = await import("data:text/javascript;base64," + Buffer.from(source).toString("base64"));
const params = initParams({vggt_interval: 17});
assert.equal(params.read().static_camera, false);
assert.equal(params.read().sam_chunk_size, 16);
assert.equal(Object.hasOwn(params.read(), "sam_hand_batch_size"), false);
assert.equal(elements.optVggt.disabled, false);
elements.optStaticCamera.click();
assert.equal(params.read().static_camera, true);
assert.equal(elements.optVggt.disabled, true);
assert.equal(params.read().vggt_interval, 17);
elements.optStaticCamera.click();
assert.equal(elements.optVggt.disabled, false);
assert.equal(params.read().vggt_interval, 17);
params.apply({static_camera: true, vggt_interval: 23});
assert.equal(elements.optVggt.disabled, true);
assert.equal(params.read().vggt_interval, 23);
params.apply({vggt_interval: 31});
assert.equal(params.read().static_camera, false);
assert.equal(elements.optVggt.disabled, false);
assert.equal(params.read().vggt_interval, 31);
params.apply({sam_chunk_size: 32});
assert.equal(params.read().sam_chunk_size, 32);
elements.optSamChunk.value = "4";
assert.equal(params.read().sam_chunk_size, 4);
params.apply({});
assert.equal(params.read().sam_chunk_size, 16);
params.apply({sam_hand_batch_size: 3});
assert.equal(params.read().sam_chunk_size, 16);
assert.equal(Object.hasOwn(params.read(), "sam_hand_batch_size"), false);
for (const value of [0, -1, 1.5, true, "1.5", "16oops", ""]) {
  assert.throws(() => params.apply({sam_chunk_size: value}), /integer/);
  params.apply({});
  elements.optSamChunk.value = value;
  assert.throws(() => params.read(), /integer/);
}
assert.throws(() => initParams({sam_chunk_size: 0}), /positive integer/);
'''
        path = Path(server.WEB_DIR) / "static/js/params.js"
        result = subprocess.run([shutil.which("node"), "--input-type=module", "-", str(path)],
                                input=script, text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        response = self.client.get("/")
        try:
            html = response.get_data(as_text=True)
        finally:
            response.close()
        self.assertIn('id="optStaticCamera"', html)
        self.assertIn('aria-label="Static camera"', html)
        self.assertIn('id="optSamChunk"', html)
        self.assertNotIn('id="optSamHandBatch"', html)
        self.assertIn("Hand batch is automatically twice the chunk size", html)


if __name__ == "__main__":
    unittest.main()

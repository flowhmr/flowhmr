"""CPU checks for the optional compiler boundary, configuration and recovery."""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import warnings

import torch
from torch._dynamo.exc import (
    BackendCompilerFailed, CacheLimitExceeded, InternalTorchDynamoError, InvalidBackend,
    TorchRuntimeError, Unsupported,
)

from flowhmr.utils.sam_compile import enable_sam_compile, resolve_sam_compile


class FakeBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))
        self.calls = 0

    def forward(self, value, *, offset=0):
        self.calls += 1
        return value * self.scale + offset


def make_model():
    model = torch.nn.Module()
    model.backbone = FakeBackbone()
    model.decoder = torch.nn.Identity()
    return model


class SamCompileTests(unittest.TestCase):
    def test_lazy_compile_preserves_modules_parameters_and_decoder_hooks(self):
        model = make_model()
        backbone, decoder = model.backbone, model.decoder
        original_keys = tuple(model.state_dict())
        original_parameters = tuple(model.parameters())
        decoder_outputs = []
        decoder.register_forward_hook(lambda module, args, output: decoder_outputs.append(output))
        compiled = Mock(side_effect=lambda value, offset=0: value * 2 + offset)
        with patch("torch.compile", return_value=compiled) as compiler:
            self.assertTrue(enable_sam_compile(model, "cuda:0"))
            compiler.assert_called_once_with(
                unittest.mock.ANY, fullgraph=True, dynamic=False,
                options={"triton.cudagraphs": False},
            )
            self.assertEqual(backbone._flowhmr_compile_state["status"], "pending")
            compiled.assert_not_called()
            output = model.decoder(model.backbone(torch.tensor([3.0]), offset=4))
            torch.testing.assert_close(output, torch.tensor([10.0]))
            compiled.assert_called_once()
            self.assertEqual(backbone.calls, 0)
            self.assertEqual(backbone._flowhmr_compile_state["status"], "active")
            self.assertTrue(enable_sam_compile(model, "cuda:0"))
            compiler.assert_called_once()
        self.assertIs(model.backbone, backbone)
        self.assertIs(model.decoder, decoder)
        self.assertEqual(tuple(model.state_dict()), original_keys)
        self.assertEqual(tuple(model.parameters()), original_parameters)
        self.assertEqual(len(decoder_outputs), 1)
        self.assertIs(decoder_outputs[0], output)

    def test_lazy_unsupported_recovers_once_and_saved_callable_stays_eager(self):
        model = make_model()
        eager_forward = model.backbone.forward
        compiled = Mock(side_effect=Unsupported("unsupported operation " * 100))
        with patch("torch.compile", return_value=compiled):
            enable_sam_compile(model, "cuda")
        saved_forward = model.backbone.forward
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            torch.testing.assert_close(saved_forward(torch.tensor([3.0]), offset=1), torch.tensor([7.0]))
            torch.testing.assert_close(saved_forward(torch.tensor([4.0]), offset=1), torch.tensor([9.0]))
        self.assertEqual(model.backbone.forward, eager_forward)
        self.assertEqual(model.backbone.calls, 2)
        compiled.assert_called_once()
        self.assertEqual(len(recorded), 1)
        state = model.backbone._flowhmr_compile_state
        self.assertEqual(state["status"], "failed")
        self.assertTrue(state["error"].startswith("Unsupported:"))
        self.assertLessEqual(len(state["error"]), 500)
        with patch("torch.compile") as compiler:
            self.assertFalse(enable_sam_compile(model, "cuda"))
            compiler.assert_not_called()

    def test_new_shape_cache_limit_recovers_after_a_successful_compile(self):
        model = make_model()
        compiled = Mock(side_effect=[torch.tensor([6.0]), CacheLimitExceeded("cache_size_limit reached")])
        with patch("torch.compile", return_value=compiled):
            enable_sam_compile(model, "cuda")
        model.backbone(torch.tensor([3.0]))
        self.assertEqual(model.backbone._flowhmr_compile_state["status"], "active")
        with self.assertWarnsRegex(RuntimeWarning, "CacheLimitExceeded"):
            result = model.backbone(torch.tensor([4.0, 5.0]))
        torch.testing.assert_close(result, torch.tensor([8.0, 10.0]))
        self.assertEqual(model.backbone._flowhmr_compile_state["status"], "failed")
        self.assertEqual(model.backbone.calls, 1)

    def test_compile_creation_error_restores_eager(self):
        for error in (InvalidBackend("invalid"), BackendCompilerFailed(lambda: None, RuntimeError("compiler failed"))):
            with self.subTest(error=type(error).__name__):
                model = make_model()
                original = model.backbone.forward
                with patch("torch.compile", side_effect=error), self.assertWarns(RuntimeWarning):
                    self.assertFalse(enable_sam_compile(model, "cuda"))
                self.assertEqual(model.backbone.forward, original)
                torch.testing.assert_close(model.backbone(torch.tensor([3.0])), torch.tensor([6.0]))
                self.assertEqual(model.backbone._flowhmr_compile_state["status"], "failed")

    def test_runtime_errors_and_oom_propagate_without_eager_retry(self):
        oom_cause = Unsupported("wrapped allocation")
        oom_cause.__cause__ = torch.OutOfMemoryError("out of memory")
        oom_context = InternalTorchDynamoError("wrapped allocation")
        oom_context.__context__ = torch.OutOfMemoryError("out of memory")
        errors = (
            RuntimeError("bad input"), ValueError("bad argument"), AssertionError("invalid shape"),
            TorchRuntimeError("invalid tensor operation"), torch.OutOfMemoryError("out of memory"),
            BackendCompilerFailed(lambda: None, torch.OutOfMemoryError("out of memory")),
            oom_cause, oom_context,
        )
        for error in errors:
            with self.subTest(error=type(error).__name__):
                model = make_model()
                compiled = Mock(side_effect=error)
                with patch("torch.compile", return_value=compiled):
                    enable_sam_compile(model, "cuda")
                with warnings.catch_warnings(record=True) as recorded:
                    warnings.simplefilter("always")
                    with self.assertRaises(type(error)) as caught:
                        model.backbone(torch.tensor([3.0]))
                self.assertIs(caught.exception, error)
                self.assertEqual(recorded, [])
                self.assertEqual(model.backbone.calls, 0)
                self.assertEqual(model.backbone._flowhmr_compile_state["status"], "pending")

    def test_creation_oom_propagates_without_restoring_or_warning(self):
        model = make_model()
        error = BackendCompilerFailed(lambda: None, torch.OutOfMemoryError("out of memory"))
        with patch("torch.compile", side_effect=error), warnings.catch_warnings(record=True) as recorded:
            with self.assertRaises(BackendCompilerFailed):
                enable_sam_compile(model, "cuda")
        self.assertEqual(recorded, [])
        self.assertEqual(model.backbone.calls, 0)

    def test_environment_resolution_explicit_override_and_device_validation(self):
        for value in ("0", " false ", "NO", "off", ""):
            with self.subTest(value=value), patch.dict(os.environ, {"FLOWHMR_SAM_COMPILE": value}):
                self.assertFalse(resolve_sam_compile("cpu"))
        for value in ("1", " true ", "YES", "on"):
            with self.subTest(value=value), patch.dict(os.environ, {"FLOWHMR_SAM_COMPILE": value}):
                self.assertTrue(resolve_sam_compile("cuda:0"))
                with self.assertRaisesRegex(ValueError, "CUDA device"):
                    resolve_sam_compile("cpu")
        with patch.dict(os.environ, {"FLOWHMR_SAM_COMPILE": "invalid"}):
            with self.assertRaisesRegex(ValueError, "FLOWHMR_SAM_COMPILE"):
                resolve_sam_compile("cuda")
            self.assertFalse(resolve_sam_compile("cpu", False))
            self.assertTrue(resolve_sam_compile("cuda", True))
        with self.assertRaises(TypeError):
            resolve_sam_compile("cuda", 1)
        with patch("torch.compile", None):
            with self.assertRaisesRegex(RuntimeError, "requires torch.compile"):
                resolve_sam_compile("cuda", True)

    def test_factory_compiles_after_dtype_conversion_and_respects_override(self):
        from flowhmr.utils import runtime_sam_features as runtime
        model, cfg = make_model(), SimpleNamespace()
        calls = []
        estimator = object()
        def load(**kwargs):
            calls.append("load")
            return model, cfg
        def convert(sam_model, device):
            self.assertIs(sam_model, model)
            calls.append("convert")
        def compile_model(sam_model, device):
            self.assertIs(sam_model, model)
            calls.append("compile")
            return True
        with patch.dict(os.environ, {"FLOWHMR_SAM_COMPILE": "1"}), \
             patch.object(runtime.osp, "exists", return_value=True), \
             patch.object(runtime, "load_sam_3d_body", side_effect=load), \
             patch.object(runtime, "_maybe_convert_backbone_to_fp16", side_effect=convert), \
             patch("flowhmr.utils.sam_compile.enable_sam_compile", side_effect=compile_model), \
             patch.object(runtime, "SAM3DBodyEstimator", return_value=estimator) as make_estimator, \
             patch.object(runtime, "Sam3DTokenExtractor") as make_extractor:
            runtime.build_sam3d_extractor(torch.device("cuda"))
            self.assertEqual(calls, ["load", "convert", "compile"])
            make_estimator.assert_called_once_with(sam_3d_body_model=model, model_cfg=cfg)
            make_extractor.assert_called_once_with(
                estimator, frame_batch_size=16, use_gpu_preprocess=True,
            )
            calls.clear()
            runtime.build_sam3d_extractor(torch.device("cuda"), use_compile=False)
            self.assertEqual(calls, ["load", "convert"])

    def test_factory_rejects_bad_configuration_before_loading_weights(self):
        from flowhmr.utils import runtime_sam_features as runtime
        with patch.object(runtime, "load_sam_3d_body") as load:
            with patch.dict(os.environ, {"FLOWHMR_SAM_COMPILE": "invalid"}):
                with self.assertRaisesRegex(ValueError, "FLOWHMR_SAM_COMPILE"):
                    runtime.build_sam3d_extractor(torch.device("cuda"))
            with self.assertRaisesRegex(ValueError, "CUDA device"):
                runtime.build_sam3d_extractor(torch.device("cpu"), use_compile=True)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()

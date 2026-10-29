"""CPU equivalence checks against the original dense attention definitions."""

import os
import unittest
from dataclasses import fields, is_dataclass
from unittest.mock import patch

import torch

from flowhmr.network.hymotion_mmdit_for_v2m import HunyuanMotionMMDiT
from flowhmr.network.mmdit.attention import attention
from flowhmr.network.mmdit.modulate_layers import ModulateDiT
from flowhmr.network.mmdit import query_batched_attention as batched


def dense_motion_mask(valid, motion_len, *, mask_mode="narrowband_v2m",
                      mm_window=2, mc_length=2, tm_length=0, tt_length=None):
    """Original four-quadrant visibility, independently built at full size.

    Padding applies to keys only in MMDiT. The original attention() itself
    handles rows that become fully masked, so leave those rows intact here.
    """
    batch_size, total_len = valid.shape
    condition_len = total_len - motion_len
    mask = torch.zeros(batch_size, 1, total_len, total_len, dtype=torch.float64)
    for query in range(total_len):
        for key in range(total_len):
            qm, km = query < motion_len, key < motion_len
            qt = query if qm else query - motion_len
            kt = key if km else key - motion_len
            visible = True
            if mask_mode == "narrowband_v2m":
                if qm and km:
                    visible = abs(qt - kt) <= mm_window
                elif qm:
                    visible = mc_length <= 0 or abs(qt - kt) <= round(mc_length)
                elif km:
                    visible = tm_length != 0 and (tm_length < 0 or abs(qt - kt) <= round(tm_length))
                else:
                    tt = mc_length if tt_length is None else tt_length
                    visible = tt <= 0 or abs(qt - kt) <= round(tt)
            elif not qm and km:
                visible = False
            elif qm and km:
                if mask_mode == "causal":
                    visible = key <= query
                elif mask_mode == "narrowband":
                    visible = abs(query - key) <= mm_window
                elif mask_mode is not None:
                    raise AssertionError(mask_mode)
            if not visible:
                mask[:, :, query, key] = float("-inf")
    mask.masked_fill_(~valid[:, None, None, :], float("-inf"))
    return mask


def dense_refiner_mask(valid):
    """Original global refiner: valid Q/K pairs, diagonal for masked queries."""
    batch_size, length = valid.shape
    visible = torch.zeros(batch_size, 1, length, length, dtype=torch.bool)
    for sample in range(batch_size):
        for query in range(length):
            if valid[sample, query]:
                visible[sample, 0, query] = valid[sample]
            else:
                visible[sample, 0, query, query] = True
    return torch.where(visible, 0.0, float("-inf"))


def random_qkv(batch_size, length, dtype=torch.float64):
    generator = torch.Generator().manual_seed(1701 + batch_size + length)
    return tuple(torch.randn(batch_size, length, 2, 4, generator=generator, dtype=dtype)
                 for _ in range(3))


def tiny_model(**options):
    config = dict(
        input_dim=6, feat_dim=32, output_dim=6, ctxt_input_dim=10,
        vtxt_input_dim=8, num_layers=3, num_heads=4, mlp_ratio=2.0,
        text_refiner_module="flowhmr/network/token_refiner.SingleTokenRefiner",
        text_refiner_cfg={"num_layers": 1}, final_layer_cfg={"act_type": "silu"},
        mask_mode="narrowband_v2m", narrowband_length=2 / 30,
        narrowband_v2m_length=2, narrowband_tm_length=0,
    )
    config.update(options)
    with torch.random.fork_rng():
        torch.manual_seed(719)
        model = HunyuanMotionMMDiT(**config)
        # Modulation gates default to zero. Nonzero gates ensure these tests
        # exercise attention outputs instead of skipping every residual branch.
        for module in model.modules():
            if isinstance(module, ModulateDiT):
                torch.nn.init.normal_(module.linear.weight, std=0.04)
                torch.nn.init.normal_(module.linear.bias, std=0.04)
    return model


def tiny_inputs(batch_size=2, motion_len=9, condition_len=7, padded=True):
    generator = torch.Generator().manual_seed(320)
    motion_valid = torch.ones(batch_size, motion_len, dtype=torch.bool)
    condition_valid = torch.ones(batch_size, condition_len, dtype=torch.bool)
    if padded:
        motion_valid[-1, -3:] = False
        condition_valid[-1, -2:] = False
    return dict(
        x=torch.randn(batch_size, motion_len, 6, generator=generator),
        ctxt_input=torch.randn(batch_size, condition_len, 10, generator=generator),
        vtxt_input=torch.randn(batch_size, 1, 8, generator=generator),
        timesteps=torch.tensor([0.35] * batch_size),
        x_mask_temporal=motion_valid,
        ctxt_mask_temporal=condition_valid,
    )


def local_plan(valid, motion_len, *, chunk_size=3, mm_window=2,
               mc_length=2, tm_length=0, tt_length=None):
    tt = mc_length if tt_length is None else tt_length
    return batched.build_local_attention_plan(
        motion_len, valid.shape[1] - motion_len, chunk_size=chunk_size,
        mm_window=mm_window,
        mc_window=round(mc_length) if mc_length > 0 else None,
        cm_window=-1 if tm_length == 0 else (round(tm_length) if tm_length > 0 else None),
        cc_window=round(tt) if tt > 0 else None,
        key_padding_mask=~valid,
    )


def retained_tensor_storage_bytes(value):
    """Count actual tensor storage once, including shared views/descriptors."""
    storages = {}
    visited = set()

    def visit(item):
        if id(item) in visited:
            return
        visited.add(id(item))
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            storages[(str(item.device), storage.data_ptr())] = storage.nbytes()
        elif is_dataclass(item):
            for field in fields(item):
                visit(getattr(item, field.name))
        elif isinstance(item, dict):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    return sum(storages.values())


class QueryBatchedAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def test_local_attention_matches_dense_windows_padding_and_tail_queries(self):
        cases = (
            dict(mm_window=2, mc_length=2, tm_length=0, tt_length=None),
            dict(mm_window=0, mc_length=1, tm_length=2, tt_length=3),
            dict(mm_window=20, mc_length=-1, tm_length=-1, tt_length=-1),
            dict(mm_window=-1, mc_length=0, tm_length=0, tt_length=1),
            dict(mm_window=1, mc_length=3, tm_length=-1, tt_length=0),
            dict(mm_window=0, mc_length=0.2, tm_length=0.2, tt_length=0.2),
        )
        for dtype in (torch.float64, torch.float32):
            for batch_size, motion_len, condition_len in ((1, 7, 5), (2, 5, 9)):
                generator = torch.Generator().manual_seed(190)
                valid = torch.rand(batch_size, motion_len + condition_len, generator=generator) > 0.3
                q, k, v = random_qkv(batch_size, valid.shape[1], dtype)
                for options in cases:
                    for chunk_size in (1, 3, 32):
                        with self.subTest(dtype=dtype, batch=batch_size, lengths=(motion_len, condition_len),
                                          chunk=chunk_size, **options):
                            reference = attention(q, k, v, mode="torch", training=False,
                                                  attn_mask=dense_motion_mask(valid, motion_len, **options))
                            plan = local_plan(valid, motion_len, chunk_size=chunk_size, **options)
                            actual = batched.query_batched_attention(q, k, v, plan)
                            tolerance = 2e-12 if dtype == torch.float64 else 1e-6
                            torch.testing.assert_close(actual, reference, atol=tolerance, rtol=tolerance)

    def test_fully_masked_motion_rows_fall_back_to_all_keys_including_padding(self):
        valid = torch.zeros(2, 12, dtype=torch.bool)
        # Some rows in sample 0 see this distant key, while earlier rows do not.
        # Sample 1 has no valid key anywhere. Both kinds of empty row must use
        # the original full-key fallback, even across disabled stream edges.
        valid[0, -1] = True
        options = dict(mm_window=0, mc_length=1, tm_length=0, tt_length=1)
        q, k, v = random_qkv(2, 12)
        dense_mask = dense_motion_mask(valid, 7, **options)
        expected = attention(q, k, v, mode="torch", training=False, attn_mask=dense_mask)
        actual = batched.query_batched_attention(q, k, v, local_plan(valid, 7, **options))
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
        unmasked = attention(q, k, v, mode="torch", training=False)
        empty_rows = torch.isinf(dense_mask[:, 0]).all(-1)
        self.assertTrue(empty_rows[0].any())
        self.assertFalse(empty_rows[0].all())
        torch.testing.assert_close(actual[empty_rows], unmasked[empty_rows], atol=2e-12, rtol=2e-12)

    def test_global_refiner_keeps_all_valid_keys_and_padding_query_diagonal(self):
        valid = torch.tensor([[True, False, True, True, False, False, True],
                              [False, False, False, False, False, False, False]])
        for dtype in (torch.float64, torch.float32):
            q, k, v = random_qkv(2, 7, dtype)
            for chunk_size in (1, 3, 16):
                with self.subTest(dtype=dtype, chunk=chunk_size):
                    plan = batched.build_global_attention_plan(7, chunk_size=chunk_size, valid_tokens=valid)
                    actual = batched.query_batched_attention(q, k, v, plan)
                    expected = attention(q, k, v, mode="torch", training=False,
                                         attn_mask=dense_refiner_mask(valid))
                    tolerance = 2e-12 if dtype == torch.float64 else 1e-6
                    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
                    # A diagonal-only query must return its own value exactly.
                    torch.testing.assert_close(actual[~valid], v.flatten(2)[~valid], atol=0, rtol=0)
        q, k, v = random_qkv(1, 9)
        actual = batched.query_batched_attention(
            q, k, v, batched.build_global_attention_plan(9, chunk_size=4))
        torch.testing.assert_close(actual, attention(q, k, v, mode="torch", training=False),
                                   atol=2e-12, rtol=2e-12)

    def test_joint_softmax_keeps_motion_and_condition_normalization_together(self):
        # Equal logits mean the output is the average over ALL visible keys,
        # not an average/sum of independently normalized stream outputs.
        q = torch.zeros(1, 8, 1, 2, dtype=torch.float64)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        v[:, 5:] = 8
        plan = batched.build_local_attention_plan(5, 3, chunk_size=2, mm_window=20,
                                                  mc_window=None, cm_window=None, cc_window=None)
        actual = batched.query_batched_attention(q, k, v, plan)
        torch.testing.assert_close(actual, torch.full((1, 8, 2), 3.0, dtype=torch.float64),
                                   atol=0, rtol=0)

    def test_plans_do_not_store_full_sequence_square_masks(self):
        motion_len, condition_len, chunk_size, window = 257, 251, 16, 3
        valid = torch.ones(2, motion_len + condition_len, dtype=torch.bool)
        plan = batched.build_local_attention_plan(
            motion_len, condition_len, chunk_size=chunk_size, mm_window=window,
            mc_window=window, cm_window=window, cc_window=window, key_padding_mask=~valid,
        )
        covered = []
        total_mask_elements = 0
        for chunk in plan.chunks:
            q_len = chunk.query_stop - chunk.query_start
            self.assertGreater(q_len, 0)
            self.assertLessEqual(q_len, chunk_size)
            covered.extend(range(chunk.query_start, chunk.query_stop))
            self.assertIsNotNone(chunk.key_indices)
            self.assertLessEqual(chunk.key_indices.numel(), 2 * (chunk_size + 2 * window))
            self.assertEqual(chunk.key_indices.numel(), chunk.key_indices.unique().numel())
            if chunk.attn_mask is not None:
                self.assertEqual(chunk.attn_mask.shape[-2], q_len)
                self.assertLessEqual(chunk.attn_mask.shape[-1], 2 * (chunk_size + 2 * window))
                total_mask_elements += chunk.attn_mask.numel()
        self.assertEqual(covered, list(range(motion_len + condition_len)))
        self.assertLessEqual(total_mask_elements,
                             2 * (motion_len + condition_len) * 2 * (chunk_size + 2 * window))
        global_plan = batched.build_global_attention_plan(valid.shape[1], chunk_size=chunk_size,
                                                           valid_tokens=valid)
        self.assertEqual(global_plan.valid_tokens.shape, valid.shape)
        for chunk in global_plan.chunks:
            self.assertIsNone(chunk.key_indices)
            self.assertIsNone(chunk.attn_mask)

    def test_full_window_and_all_masked_fallback_plans_keep_bounded_storage(self):
        length, chunk_size, batch_size = 2048, 16, 2
        for all_masked in (False, True):
            with self.subTest(all_masked=all_masked):
                padding = torch.full((batch_size, length), all_masked, dtype=torch.bool)
                window = 3 if all_masked else None
                plan = batched.build_local_attention_plan(
                    1031, length - 1031, chunk_size=chunk_size, mm_window=window,
                    mc_window=window, cm_window=window, cc_window=window,
                    key_padding_mask=padding,
                )
                # A retained BxLxL bool mask would require 8 MiB here. This
                # bound also catches hiding quadratic data in chunk descriptors.
                self.assertLessEqual(retained_tensor_storage_bytes(plan), 16 * length * chunk_size)


class QueryAttentionConfigurationTests(unittest.TestCase):
    def test_defaults_environment_and_explicit_precedence(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(batched.resolve_query_attention_config(), ("original", 256))
            model = tiny_model()
            self.assertEqual(model.attention_backend, "original")
            self.assertEqual(model.attention_query_chunk_size, 256)
        with patch.dict(os.environ, {"FLOWHMR_ATTENTION_BACKEND": "query_batched",
                                     "FLOWHMR_ATTENTION_QUERY_CHUNK_SIZE": "17"}):
            self.assertEqual(batched.resolve_query_attention_config(), ("query_batched", 17))
            self.assertEqual(batched.resolve_query_attention_config("original", 4), ("original", 4))
            self.assertEqual(batched.resolve_query_attention_config(chunk_size=8), ("query_batched", 8))
            model = tiny_model()
            self.assertEqual(model.attention_backend, "query_batched")
            self.assertEqual(model.attention_query_chunk_size, 17)
            model = tiny_model(attention_backend="original", attention_query_chunk_size=4)
            self.assertEqual(model.attention_backend, "original")
            self.assertEqual(model.attention_query_chunk_size, 4)
        with patch.dict(os.environ, {"FLOWHMR_ATTENTION_BACKEND": "invalid",
                                     "FLOWHMR_ATTENTION_QUERY_CHUNK_SIZE": "invalid"}):
            self.assertEqual(batched.resolve_query_attention_config("original", 4), ("original", 4))

    def test_rejects_unknown_backend_and_nonpositive_or_noninteger_chunk(self):
        for backend in ("fast", "torch", "", True, 2):
            with self.subTest(backend=backend), self.assertRaises((TypeError, ValueError)):
                batched.resolve_query_attention_config(backend, 4)
        for chunk_size in (0, -1, True, False, 1.5, 2.0, "2.5", "bad", ""):
            with self.subTest(chunk=chunk_size), self.assertRaises((TypeError, ValueError)):
                batched.resolve_query_attention_config("query_batched", chunk_size)
        for backend, chunk_size in (("invalid", "4"), ("query_batched", "0"),
                                    ("query_batched", "2.0")):
            with self.subTest(backend=backend, chunk=chunk_size), \
                 patch.dict(os.environ, {"FLOWHMR_ATTENTION_BACKEND": backend,
                                         "FLOWHMR_ATTENTION_QUERY_CHUNK_SIZE": chunk_size}), \
                 self.assertRaises((TypeError, ValueError)):
                batched.resolve_query_attention_config()


class QueryAttentionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def models(self, **options):
        original = tiny_model(attention_backend="original", attention_query_chunk_size=3, **options)
        candidate = tiny_model(attention_backend="query_batched", attention_query_chunk_size=3, **options)
        self.assertEqual(tuple(original.state_dict()), tuple(candidate.state_dict()))
        loaded = candidate.load_state_dict(original.state_dict(), strict=True)
        self.assertEqual(loaded.missing_keys, [])
        self.assertEqual(loaded.unexpected_keys, [])
        return original.eval(), candidate.eval()

    def test_eval_network_matches_original_with_refiner_rope_start_token_and_padding(self):
        cases = (
            {},
            {"insert_start_token": True},
            {"apply_rope_to_single_branch": False},
            {"narrowband_tm_length": -1, "narrowband_tt_length": 1,
             "narrowband_v2m_length": -1},
            {"narrowband_length": 0, "narrowband_tm_length": 0.2,
             "narrowband_tt_length": 0.2, "narrowband_v2m_length": 0.2},
        )
        for batch_size in (1, 2):
            for options in cases:
                with self.subTest(batch_size=batch_size, options=options):
                    original, candidate = self.models(**options)
                    inputs = tiny_inputs(batch_size=batch_size)
                    with torch.no_grad():
                        expected = original(**inputs)
                        # A passing comparison alone could accidentally test
                        # two dense paths. Reject either full mask allocation.
                        with patch.object(candidate, "_build_dmm_attn_mask_shared",
                                          side_effect=AssertionError("dense double-stream mask used")), \
                             patch.object(candidate, "_build_smm_attn_mask_shared",
                                          side_effect=AssertionError("dense single-stream mask used")), \
                             patch("flowhmr.network.token_refiner.attention",
                                   side_effect=AssertionError("dense refiner path used")):
                            actual = candidate(**inputs)
                    self.assertTrue(torch.isfinite(actual).all())
                    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def test_training_preserves_original_dropout_and_backward(self):
        original, candidate = self.models(dropout=0.1)
        original.train()
        candidate.train()
        inputs = tiny_inputs()
        with torch.random.fork_rng():
            torch.manual_seed(781)
            expected = original(**inputs)
            torch.manual_seed(781)
            with patch("flowhmr.network.hymotion_mmdit_for_v2m.build_local_attention_plan",
                       side_effect=AssertionError("batched path used while training")):
                actual = candidate(**inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected.square().mean().backward()
        actual.square().mean().backward()
        actual_parameters = dict(candidate.named_parameters())
        for name, parameter in original.named_parameters():
            actual_gradient = actual_parameters[name].grad
            if parameter.grad is None:
                self.assertIsNone(actual_gradient)
            else:
                torch.testing.assert_close(actual_gradient, parameter.grad, rtol=0, atol=0)

    def test_other_mask_modes_and_attention_inspection_keep_original_path(self):
        for mask_mode in (None, "causal", "narrowband"):
            with self.subTest(mask_mode=mask_mode):
                original, candidate = self.models(mask_mode=mask_mode)
                inputs = tiny_inputs(padded=False)
                with torch.no_grad(), \
                     patch("flowhmr.network.hymotion_mmdit_for_v2m.build_local_attention_plan",
                           side_effect=AssertionError("batched path used for unsupported mode")):
                    torch.testing.assert_close(candidate(**inputs), original(**inputs), rtol=0, atol=0)
                    expected = original.forward_with_attn(**inputs)
                    actual = candidate.forward_with_attn(**inputs)
                torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                self.assertEqual(actual[2:], expected[2:])
                self.assertEqual(len(actual[1]), 3)
                for actual_attention, expected_attention in zip(actual[1], expected[1]):
                    torch.testing.assert_close(actual_attention, expected_attention, rtol=0, atol=0)

    def test_all_masked_condition_keeps_original_refiner_and_main_fallbacks(self):
        original, candidate = self.models()
        inputs = tiny_inputs()
        inputs["ctxt_mask_temporal"][:] = False
        inputs["x_mask_temporal"][-1] = False
        with torch.no_grad():
            expected = original(**inputs)
            actual = candidate(**inputs)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def test_dictionary_conditions_and_cfg_shaped_batch_preserve_outputs(self):
        original, candidate = self.models(ctxt_input_dim={"sam": 6, "camera": 4})
        inputs = tiny_inputs(batch_size=2)
        context = inputs["ctxt_input"]
        inputs["ctxt_input"] = {"sam": context[..., :6], "camera": context[..., 6:]}
        # Model-side conditional/unconditional pairs share the same time axis.
        # The second member's missing condition must preserve mask fallbacks.
        inputs["ctxt_mask_temporal"][1] = False
        for value in inputs["ctxt_input"].values():
            value[1] = 0
        with torch.no_grad():
            expected = original(**inputs)
            actual = candidate(**inputs)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def test_direct_block_plan_rejects_training_and_attention_map_requests(self):
        model = tiny_model()
        features = torch.zeros(1, 8, 32)
        adapter = torch.zeros(1, 1, 32)
        plan = batched.build_local_attention_plan(5, 3, chunk_size=2, mm_window=2)
        calls = (
            (model.double_blocks[0], dict(motion_feat=features[:, :5], text_feat=features[:, 5:],
                                         adapter=adapter)),
            (model.single_blocks[0], dict(x=features, split_len=5, adapter=adapter)),
        )
        for block, inputs in calls:
            with self.subTest(block=type(block).__name__):
                block.train()
                with self.assertRaisesRegex(ValueError, "evaluation"):
                    block(**inputs, attention_plan=plan)
                block.eval()
                with self.assertRaisesRegex(ValueError, "attention-map"):
                    block(**inputs, attention_plan=plan, return_attn=True)
        refiner = model.text_refiner.individual_token_refiner.blocks[0]
        refiner.train()
        with self.assertRaisesRegex(ValueError, "evaluation"):
            refiner(features, adapter, attention_plan=batched.build_global_attention_plan(8))


if __name__ == "__main__":
    unittest.main()

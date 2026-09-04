import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch as mock_patch

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sparse.llada_patch import (
    _BlockDualCache,
    _adamas_prefix_indices,
    _attention_output_lse,
    _cached_forward,
    _compact_prefix_cache,
    _dual_cache_from_dense,
    _hadamard_transform,
    _legacy_prefix_cache,
    _merge_attention_states,
    _new_losa_state,
    _select_positions,
    _transfer_tokens,
    patch_model,
)
from src.sparse.core import _TRITON_AVAILABLE, _fused_kv_index_copy_


MODEL_PATH = "/data0/ysy/models/LLaDA2.1-mini"


def _tiny_model(layers=6):
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    values = {
        "vocab_size": 128,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 126,
        "hidden_size": 32,
        "intermediate_size": 64,
        "moe_intermediate_size": 16,
        "num_hidden_layers": layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "num_experts": None,
        "num_shared_experts": None,
        "first_k_dense_replace": 99,
        "max_position_embeddings": 64,
        "use_qk_norm": False,
        "use_cache": True,
    }
    for name, value in values.items():
        setattr(config, name, value)
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(config, trust_remote_code=True).eval()


def _block_mask(num_blocks, block_length, dtype):
    allowed = (
        torch.tril(torch.ones(num_blocks, num_blocks, dtype=torch.bool))
        .repeat_interleave(block_length, 0)
        .repeat_interleave(block_length, 1)[None, None]
    )
    return torch.zeros(allowed.shape, dtype=dtype).masked_fill(
        ~allowed, torch.finfo(dtype).min
    )


class BlockCacheSparsePatchTest(unittest.TestCase):
    def test_prefix_compaction_selects_once_and_shares_indices(self):
        cache = DynamicCache.from_legacy_cache(
            tuple(
                (torch.randn(1, 1, 4, 2), torch.randn(1, 1, 4, 2))
                for _ in range(2)
            )
        )
        model = SimpleNamespace(
            config=SimpleNamespace(model_type="llada2_moe"),
            model=SimpleNamespace(
                layers=[None, None],
                rotary_emb=lambda query, positions: (
                    torch.ones(1, positions.shape[-1], query.shape[-1]),
                    torch.zeros(1, positions.shape[-1], query.shape[-1]),
                ),
            )
        )
        queries = [torch.randn(1, 1, 2, 2) for _ in range(2)]
        with mock_patch(
            "src.sparse.llada_patch._adamas_prefix_indices",
            return_value=torch.tensor([1, 3]),
        ) as selector:
            compact, indices = _compact_prefix_cache(
                model, cache, 4, queries, torch.arange(2).unsqueeze(0), 2, 2
            )

        selector.assert_called_once()
        self.assertEqual(
            selector.call_args.kwargs["bucket_thresholds"],
            ((-1.73, 0.0, 1.72), (-2.74, 0.0, 2.69)),
        )
        self.assertTrue(all(value.tolist() == [1, 3] for value in indices))
        self.assertTrue(all(key.shape[-2] == 2 for key, _ in compact))

    def test_losa_online_merge_matches_concatenated_attention(self):
        torch.manual_seed(1)
        query = torch.randn(1, 4, 3, 8)
        prefix_key = torch.randn(1, 2, 5, 8)
        prefix_value = torch.randn(1, 2, 5, 8)
        block_key = torch.randn(1, 2, 4, 8)
        block_value = torch.randn(1, 2, 4, 8)
        prefix_mask = torch.zeros(1, 1, 3, 5)
        block_mask = torch.zeros(1, 1, 3, 4)

        prefix_output, prefix_lse = _attention_output_lse(
            query, prefix_key, prefix_value, prefix_mask, num_key_value_groups=2
        )
        block_output, block_lse = _attention_output_lse(
            query, block_key, block_value, block_mask, num_key_value_groups=2
        )
        merged_output, merged_lse = _merge_attention_states(
            prefix_output, prefix_lse, block_output, block_lse
        )
        dense_output, dense_lse = _attention_output_lse(
            query,
            torch.cat((prefix_key, block_key), dim=2),
            torch.cat((prefix_value, block_value), dim=2),
            torch.cat((prefix_mask, block_mask), dim=-1),
            num_key_value_groups=2,
        )

        torch.testing.assert_close(merged_output, dense_output, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(merged_lse, dense_lse, rtol=1e-5, atol=1e-5)

    def test_losa_prefix_state_keeps_float32_output(self):
        state = _new_losa_state(torch.zeros(1, 2, 4, 8, dtype=torch.bfloat16), 4)

        self.assertEqual(state["prefix_output"].dtype, torch.float32)
        self.assertEqual(state["prefix_lse"].dtype, torch.float32)

    def test_losa_attention_keeps_model_dtype_and_float_lse(self):
        query = torch.randn(1, 2, 3, 8, dtype=torch.bfloat16)
        key = torch.randn(1, 1, 4, 8, dtype=torch.bfloat16)
        value = torch.randn_like(key)
        mask = torch.zeros(1, 1, 3, 4, dtype=torch.bfloat16)

        output, lse = _attention_output_lse(query, key, value, mask, 2)

        self.assertEqual(output.dtype, query.dtype)
        self.assertEqual(lse.dtype, torch.float32)

    def test_query_selection_uses_decode_confidence(self):
        class ConfidenceModel:
            def __init__(self):
                self.lm_head = torch.nn.Identity()
                self.model = SimpleNamespace(norm=torch.nn.Identity())
                self.seen = None

            def _sample_with_temperature_topk_topp(self, logits, **kwargs):
                self.seen = kwargs
                confidence = torch.tensor([[0.1, 0.9, 0.2, 0.8]], device=logits.device)
                return torch.argmax(logits, dim=-1), confidence

        model = ConfidenceModel()
        hidden = torch.eye(4).view(1, 4, 4)
        positions = _select_positions(
            model,
            hidden,
            torch.full((1, 4), 127),
            mask_id=127,
            ratio=0.5,
            top_k=8,
            temperature=0.7,
            top_p=0.9,
        )

        self.assertEqual(positions.tolist(), [1, 3])
        self.assertEqual(model.seen, {"temperature": 0.7, "top_k": 8, "top_p": 0.9})

    def test_query_selection_matches_llada_confidence_threshold_rule(self):
        class ConfidenceModel:
            lm_head = torch.nn.Identity()
            model = SimpleNamespace(norm=torch.nn.Identity())

            @staticmethod
            def _sample_with_temperature_topk_topp(logits, **kwargs):
                confidence = torch.tensor([[0.8, 0.9, 0.1, 0.85]])
                return torch.argmax(logits, dim=-1), confidence

        selected = _select_positions(
            ConfidenceModel(),
            torch.eye(4).view(1, 4, 4),
            torch.full((1, 4), 127),
            mask_id=127,
            ratio=0.5,
            top_k=0,
            threshold=0.7,
        )

        self.assertEqual(selected.tolist(), [0, 1, 3])

    def test_query_selection_unions_losa_priority_without_duplicates(self):
        class ConfidenceModel:
            lm_head = torch.nn.Identity()
            model = SimpleNamespace(norm=torch.nn.Identity())
            _llada_losa_context = {
                "query_priority_positions": torch.tensor([2, 3])
            }

            @staticmethod
            def _sample_with_temperature_topk_topp(logits, **kwargs):
                confidence = torch.tensor([[0.1, 0.9, 0.2, 0.8]])
                return torch.argmax(logits, dim=-1), confidence

        selected = _select_positions(
            ConfidenceModel(),
            torch.eye(4).view(1, 4, 4),
            torch.full((1, 4), 127),
            mask_id=127,
            ratio=0.5,
            top_k=0,
        )

        self.assertEqual(selected.tolist(), [1, 3, 2])

    def test_transfer_uses_llada_confidence_threshold_rule(self):
        class ConfidenceModel:
            @staticmethod
            def _sample_with_temperature_topk_topp(logits, **kwargs):
                return (
                    torch.tensor([[10, 11, 12, 13]]),
                    torch.tensor([[0.8, 0.9, 0.1, 0.85]]),
                )

        tokens, transferred = _transfer_tokens(
            ConfidenceModel(),
            torch.full((1, 4), 127),
            torch.full((1, 4), 127),
            torch.zeros(4, dtype=torch.bool),
            torch.ones(1, 4, dtype=torch.bool),
            torch.zeros(1, 4, 2),
            temperature=0.0,
            top_p=None,
            top_k=None,
            threshold=0.7,
            editing_threshold=0.9,
            num_to_transfer=2,
        )

        self.assertEqual(transferred.tolist(), [[True, True, False, True]])
        self.assertEqual(tokens.tolist(), [[10, 11, 127, 13]])

    def test_dual_cache_overwrites_only_selected_current_kv(self):
        prefix = torch.randn(1, 2, 2, 3)
        current = torch.randn(1, 2, 4, 3)
        cache = _BlockDualCache(
            [
                (
                    torch.cat((prefix, current), dim=2),
                    torch.cat((prefix, current + 1), dim=2),
                )
            ],
            [2],
        )
        original_key = cache.key_cache[0].clone()
        original_value = cache.value_cache[0].clone()
        cache.set_positions(torch.tensor([1, 3]))
        replacement_key = torch.randn(1, 2, 2, 3)
        replacement_value = torch.randn(1, 2, 2, 3)
        key, value = cache.update(replacement_key, replacement_value, 0)

        torch.testing.assert_close(key[:, :, [3, 5]], replacement_key)
        torch.testing.assert_close(value[:, :, [3, 5]], replacement_value)
        torch.testing.assert_close(key[:, :, [0, 1, 2, 4]], original_key[:, :, [0, 1, 2, 4]])
        torch.testing.assert_close(value[:, :, [0, 1, 2, 4]], original_value[:, :, [0, 1, 2, 4]])

    @unittest.skipUnless(
        torch.cuda.is_available() and _TRITON_AVAILABLE,
        "requires CUDA and Triton",
    )
    def test_fused_kv_index_copy_matches_torch(self):
        positions = torch.tensor([1, 3], device="cuda")
        key_states = torch.randn(1, 2, 2, 3, device="cuda")
        value_states = torch.randn(1, 2, 3, 2, device="cuda").transpose(-1, -2)
        actual_key = torch.randn(1, 2, 8, 3, device="cuda")
        actual_value = torch.randn_like(actual_key)
        expected_key = actual_key.clone()
        expected_value = actual_value.clone()

        expected_key.index_copy_(2, positions + 2, key_states)
        expected_value.index_copy_(2, positions + 2, value_states)
        _fused_kv_index_copy_(
            actual_key,
            actual_value,
            positions,
            key_states,
            value_states,
            prefix_length=2,
        )

        torch.testing.assert_close(actual_key, expected_key)
        torch.testing.assert_close(actual_value, expected_value)

    def test_python_adamas_selector_returns_sorted_indices(self):
        values = torch.arange(8, dtype=torch.float32)
        transformed = _hadamard_transform(values)
        expected = values @ (torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, -1, 1, -1, 1, -1, 1, -1],
                [1, 1, -1, -1, 1, 1, -1, -1],
                [1, -1, -1, 1, 1, -1, -1, 1],
                [1, 1, 1, 1, -1, -1, -1, -1],
                [1, -1, 1, -1, -1, 1, -1, 1],
                [1, 1, -1, -1, -1, -1, 1, 1],
                [1, -1, -1, 1, -1, 1, 1, -1],
            ],
            dtype=torch.float32,
        ) / 8**0.5)
        torch.testing.assert_close(transformed, expected)

        torch.manual_seed(0)
        query = torch.randn(1, 4, 3, 8)
        key = torch.randn(1, 2, 11, 8)
        indices = _adamas_prefix_indices(query, key, token_budget=4, chunk_size=3)
        self.assertGreaterEqual(indices.numel(), 4)
        self.assertTrue(torch.all(indices[1:] > indices[:-1]).item())
        self.assertTrue(torch.all((0 <= indices) & (indices < 11)).item())

    def test_adamas_selector_preserves_union_beyond_budget(self):
        query = torch.tensor([[[[2.0, 0.0], [-2.0, 0.0], [0.0, 2.0]]]])
        key = torch.tensor([[[[4.0, 0.0], [-4.0, 0.0], [0.0, 4.0]]]])
        indices = _adamas_prefix_indices(query, key, token_budget=2, chunk_size=1)

        self.assertEqual(indices.tolist(), [0, 1, 2])

    def test_dense_cached_forward_matches_full_forward(self):
        model = _tiny_model()
        tokens = torch.tensor([[1, 2, 3, 4, 127, 127, 127, 127]])
        positions = torch.arange(8).unsqueeze(0)
        attention_mask = _block_mask(2, 4, next(model.parameters()).dtype)

        with torch.no_grad():
            dense = model(
                tokens,
                attention_mask=attention_mask,
                position_ids=positions,
                use_cache=True,
                return_dict=True,
            )
            prefix_cache = _legacy_prefix_cache(dense.past_key_values, 4)
            cached_logits, selected, logit_positions = _cached_forward(
                model,
                tokens[:, 4:],
                attention_mask[:, :, 4:, :],
                positions[:, 4:],
                prefix_cache,
                {"positions": None, "step": 0},
                mask_id=127,
                ratio=1.0,
                top_k=8,
                selection_interval=2,
                dense_fallback_mask_count=0,
                query_sparse=False,
            )

        self.assertIsNone(selected)
        self.assertIsNone(logit_positions)
        torch.testing.assert_close(cached_logits, dense.logits[:, 4:], rtol=1e-5, atol=1e-5)

    def test_query_sparse_returns_logits_only_for_selected_masks(self):
        model = _tiny_model()
        tokens = torch.tensor([[1, 2, 3, 4, 127, 127, 127, 127]])
        positions = torch.arange(8).unsqueeze(0)
        attention_mask = _block_mask(2, 4, next(model.parameters()).dtype)

        with torch.no_grad():
            dense = model(
                tokens,
                attention_mask=attention_mask,
                position_ids=positions,
                use_cache=True,
                return_dict=True,
            )
            prefix_cache = _legacy_prefix_cache(dense.past_key_values, 4)
            selection_state = {
                "positions": None,
                "step": 0,
                "sparse_cache": _dual_cache_from_dense(
                    dense.past_key_values, prefix_cache, 4, 8
                ),
            }
            _, selected, logit_positions = _cached_forward(
                model,
                tokens[:, 4:],
                attention_mask[:, :, 4:, :],
                positions[:, 4:],
                prefix_cache,
                selection_state,
                mask_id=127,
                ratio=0.5,
                top_k=8,
                selection_interval=1,
                dense_fallback_mask_count=0,
                query_sparse=True,
                original_prefix_length=4,
            )

        self.assertIsNotNone(selected)
        self.assertIsNotNone(logit_positions)
        self.assertLess(selected.numel(), 4)
        torch.testing.assert_close(logit_positions, selected)

    def test_query_selection_runs_after_configured_layer(self):
        model = _tiny_model()
        tokens = torch.tensor([[1, 2, 3, 4, 127, 127, 127, 127]])
        positions = torch.arange(8).unsqueeze(0)
        attention_mask = _block_mask(2, 4, next(model.parameters()).dtype)
        completed_layers = []
        observed_layers = []
        handles = [
            layer.register_forward_hook(
                lambda _module, _inputs, _output, idx=idx: completed_layers.append(idx)
            )
            for idx, layer in enumerate(model.model.layers)
        ]

        try:
            with torch.no_grad():
                dense = model(
                    tokens,
                    attention_mask=attention_mask,
                    position_ids=positions,
                    use_cache=True,
                    return_dict=True,
                )
                completed_layers.clear()
                with mock_patch(
                    "src.sparse.llada_patch._select_positions",
                    side_effect=lambda *args, **kwargs: observed_layers.append(
                        completed_layers.copy()
                    ),
                ):
                    _cached_forward(
                        model,
                        tokens[:, 4:],
                        attention_mask[:, :, 4:, :],
                        positions[:, 4:],
                        _legacy_prefix_cache(dense.past_key_values, 4),
                        {"positions": None, "step": 0},
                        mask_id=127,
                        ratio=0.5,
                        top_k=8,
                        selection_interval=1,
                        dense_fallback_mask_count=0,
                        query_sparse=True,
                        selection_layer=3,
                        original_prefix_length=4,
                    )
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(observed_layers, [[0, 1, 2, 3]])

    def test_patch_stores_llada_selection_layer(self):
        model = _tiny_model()

        patch_model(model, selection_layer=3)

        self.assertEqual(model.config.llada_query_selection_layer, 3)

    def test_losa_first_cached_forward_matches_dense_forward_exactly(self):
        model = _tiny_model()
        patch_model(
            model,
            ratio=1.0,
            query_sparse=False,
            prefix_sparse=False,
            losa=True,
            losa_active_topk=1,
        )
        tokens = torch.tensor([[1, 2, 3, 4, 127, 127, 127, 127]])
        positions = torch.arange(8).unsqueeze(0)
        attention_mask = _block_mask(2, 4, next(model.parameters()).dtype)
        selection_state = {"positions": None, "step": 0, "sparse_cache": None}

        with torch.no_grad():
            dense = model._llada_block_cache_dense_forward(
                tokens,
                attention_mask=attention_mask,
                position_ids=positions,
                use_cache=True,
                return_dict=True,
            )
            cached_logits, selected, logit_positions = _cached_forward(
                model,
                tokens[:, 4:],
                attention_mask[:, :, 4:, :],
                positions[:, 4:],
                _legacy_prefix_cache(dense.past_key_values, 4),
                selection_state,
                mask_id=127,
                ratio=1.0,
                top_k=8,
                selection_interval=2,
                dense_fallback_mask_count=0,
                query_sparse=False,
                original_prefix_length=4,
            )

        self.assertIsNone(selected)
        self.assertIsNone(logit_positions)
        torch.testing.assert_close(cached_logits, dense.logits[:, 4:], rtol=0, atol=0)
        self.assertEqual(len(selection_state["losa_states"]), len(model.model.layers))
        self.assertTrue(
            all(
                state["prefix_output"].dtype is torch.float32
                for state in selection_state["losa_states"].values()
            )
        )

    def test_losa_full_active_budget_uses_losa_after_dense_init(self):
        model = _tiny_model()
        patch_model(
            model,
            ratio=1.0,
            query_sparse=False,
            prefix_sparse=False,
            losa=True,
            losa_active_topk=4,
        )
        tokens = torch.tensor([[1, 2, 3, 4, 127, 127, 127, 127]])
        positions = torch.arange(8).unsqueeze(0)
        attention_mask = _block_mask(2, 4, next(model.parameters()).dtype)
        selection_state = {"positions": None, "step": 0, "sparse_cache": None}

        with mock_patch.object(
            model.model.layers[0].attention,
            "_llada_losa_dense_forward",
            wraps=model.model.layers[0].attention._llada_losa_dense_forward,
        ) as dense_forward, torch.no_grad():
            dense = model._llada_block_cache_dense_forward(
                tokens,
                attention_mask=attention_mask,
                position_ids=positions,
                use_cache=True,
                return_dict=True,
            )
            cached_logits, _, _ = _cached_forward(
                model,
                tokens[:, 4:],
                attention_mask[:, :, 4:, :],
                positions[:, 4:],
                _legacy_prefix_cache(dense.past_key_values, 4),
                selection_state,
                mask_id=127,
                ratio=1.0,
                top_k=8,
                selection_interval=2,
                dense_fallback_mask_count=0,
                query_sparse=False,
                original_prefix_length=4,
            )
            first_call_count = dense_forward.call_count
            _cached_forward(
                model,
                tokens[:, 4:],
                attention_mask[:, :, 4:, :],
                positions[:, 4:],
                _legacy_prefix_cache(dense.past_key_values, 4),
                selection_state,
                mask_id=127,
                ratio=1.0,
                top_k=8,
                selection_interval=2,
                dense_fallback_mask_count=0,
                query_sparse=False,
                original_prefix_length=4,
            )

        self.assertEqual(first_call_count, 2)
        self.assertEqual(dense_forward.call_count, first_call_count)
        self.assertTrue(
            all(state["valid"].all().item() for state in selection_state["losa_states"].values())
        )

    def test_compact_prefix_matches_dense_masked_prefix(self):
        model = _tiny_model()
        tokens = torch.tensor([[1, 2, 3, 4, 127, 127, 127, 127]])
        positions = torch.arange(8).unsqueeze(0)
        attention_mask = _block_mask(2, 4, next(model.parameters()).dtype)
        prefix_indices = torch.tensor([0, 3])
        masked_attention = attention_mask.clone()
        masked_attention[:, :, 4:, 1:3] = torch.finfo(masked_attention.dtype).min

        with torch.no_grad():
            dense = model(
                tokens,
                attention_mask=masked_attention,
                position_ids=positions,
                use_cache=True,
                return_dict=True,
            )
            prefix_cache = tuple(
                (
                    key[:, :, :4].index_select(2, prefix_indices).contiguous(),
                    value[:, :, :4].index_select(2, prefix_indices).contiguous(),
                )
                for key, value in dense.past_key_values.to_legacy_cache()
            )
            cached_logits, selected, logit_positions = _cached_forward(
                model,
                tokens[:, 4:],
                attention_mask[:, :, 4:, :],
                positions[:, 4:],
                prefix_cache,
                {"positions": None, "step": 0},
                mask_id=127,
                ratio=0.5,
                top_k=8,
                selection_interval=2,
                dense_fallback_mask_count=0,
                query_sparse=False,
                prefix_indices=tuple(prefix_indices for _ in prefix_cache),
                original_prefix_length=4,
            )

        self.assertIsNone(selected)
        self.assertIsNone(logit_positions)
        torch.testing.assert_close(cached_logits, dense.logits[:, 4:], rtol=1e-5, atol=1e-5)

    def test_sparse_multiblock_generation_uses_llada_selector(self):
        model = _tiny_model()
        patch_model(
            model,
            ratio=0.5,
            top_k=8,
            selection_interval=3,
            dense_fallback_mask_count=0,
            query_sparse=True,
            prefix_sparse=True,
            prefix_token_budget=2,
            prefix_chunk_size=2,
        )
        output = model.generate(
            inputs=torch.tensor([[1, 2, 3, 4]]),
            gen_length=8,
            block_length=4,
            steps=4,
            threshold=2.0,
            editing_threshold=0.0,
            max_post_steps=2,
            eos_early_stop=False,
            mask_id=127,
            eos_id=126,
            num_to_transfer=1,
        )

        self.assertEqual(output.shape, (1, 8))
        self.assertFalse(torch.any(output == 127).item())

    def test_llada_generate_rejects_sdar_strategy_arguments(self):
        model = _tiny_model()
        patch_model(model, selection_layer=3)

        with self.assertRaisesRegex(TypeError, "remasking_strategy"):
            model.generate(
                inputs=torch.tensor([[1, 2, 3, 4]]),
                remasking_strategy="sequential",
            )


if __name__ == "__main__":
    unittest.main()

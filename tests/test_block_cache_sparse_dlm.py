import unittest
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sparse.block_cache_sparse_dlm_patch import (
    _BlockDualCache,
    _adamas_prefix_indices,
    _cached_forward,
    _hadamard_transform,
    _legacy_prefix_cache,
    _select_positions,
    patch_model,
)


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
    def test_query_selection_uses_decode_confidence(self):
        class ConfidenceModel:
            def __init__(self):
                self.lm_head = torch.nn.Identity()
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

    def test_python_adamas_selector_respects_budget(self):
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
        self.assertEqual(indices.shape, (4,))
        self.assertTrue(torch.all(indices[1:] > indices[:-1]).item())
        self.assertTrue(torch.all((0 <= indices) & (indices < 11)).item())

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

    def test_sparse_multiblock_generation_finishes(self):
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


if __name__ == "__main__":
    unittest.main()

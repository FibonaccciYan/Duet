import unittest
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sparse.block_cache_sparse_dlm_patch import (
    _cached_forward,
    _legacy_prefix_cache,
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
            cached_logits, selected = _cached_forward(
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
            )

        self.assertIsNone(selected)
        torch.testing.assert_close(cached_logits, dense.logits[:, 4:], rtol=1e-5, atol=1e-5)

    def test_sparse_multiblock_generation_finishes(self):
        model = _tiny_model()
        patch_model(
            model,
            ratio=0.5,
            top_k=8,
            selection_interval=3,
            dense_fallback_mask_count=0,
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

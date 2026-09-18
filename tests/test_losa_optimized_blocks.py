"""Driver regression: each new block must obtain logits from its own input."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
import src.optimized.losa.generation as generation


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(mask_token_id=99)
        self.generation_config = SimpleNamespace(eos_token_id=None)


class BlockDriverTest(unittest.TestCase):
    def test_current_logits_for_every_block(self):
        calls = []

        def prefix(model, x, length, positions, **kwargs):
            t = torch.zeros(1, 1, length, 1)
            return ((t, t),)

        def forward(model, family, ids, mask, positions, **kwargs):
            start = int(positions[0, 0])
            calls.append((start, bool((ids == 99).any())))
            logits = torch.full((*ids.shape, 128), -100.0)
            logits[..., 10 + start // 32] = 100.0
            t = torch.zeros(1, 1, start + ids.shape[1], 1)
            return SimpleNamespace(
                logits=logits,
                past_key_values=SimpleNamespace(to_legacy_cache=lambda: ((t, t),)),
            ), []

        with patch.object(generation, "build_sdar_prefix_cache", prefix), \
             patch.object(generation, "model_forward", forward):
            result = generation.block_diffusion_generate(
                FakeModel(), family="sdar", inputs=torch.ones(1,32,dtype=torch.long),
                gen_length=64, block_length=32, steps=32, mask_id=99,
                use_losa=False, remasking_strategy="low_confidence_dynamic",
                temperature=0.0, threshold=0.95, eos_early_stop=False,
            )
        self.assertEqual(result.tokens[0,:32].tolist(), [11]*32)
        self.assertEqual(result.tokens[0,32:].tolist(), [12]*32)
        self.assertEqual([s for s,mask_present in calls if mask_present], [32,64])

    def test_store_kv_is_forwarded(self):
        with patch.object(generation, "layerwise_cached_forward", return_value=("out", [])) as inner:
            result = generation.model_forward(
                FakeModel(), "sdar", torch.ones(1,32,dtype=torch.long), None, None,
                prefix_cache=((torch.zeros(1,1,32,1), torch.zeros(1,1,32,1)),),
                store_kv=False,
            )
        self.assertEqual(result, ("out", []))
        self.assertIs(inner.call_args.kwargs["store_kv"], False)


if __name__ == "__main__":
    unittest.main()

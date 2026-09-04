import unittest
from types import SimpleNamespace

import torch

from scripts.analyze_llada_layer_overlap import (
    block_attention_mask,
    layer_logits,
    selector_mask,
)


class LLaDALayerOverlapTest(unittest.TestCase):
    def test_block_attention_mask_allows_current_and_prior_blocks(self):
        mask = block_attention_mask(4, 2, torch.float32, "cpu")[0, 0]

        self.assertEqual(mask[0, 1].item(), 0.0)
        self.assertEqual(mask[2, 1].item(), 0.0)
        self.assertEqual(mask[0, 2].item(), torch.finfo(torch.float32).min)

    def test_selector_expands_above_minimum_at_same_threshold(self):
        transfer = selector_mask(
            torch.ones(1, 4, dtype=torch.bool),
            torch.tensor([[0.8, 0.9, 0.1, 0.85]]),
            minimum=2,
            threshold=0.7,
        )

        self.assertEqual(transfer.tolist(), [[True, True, False, True]])

    def test_final_hidden_state_is_not_normalized_twice(self):
        class CountingNorm(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, hidden_states):
                self.calls += 1
                return hidden_states

        norm = CountingNorm()
        model = SimpleNamespace(
            model=SimpleNamespace(norm=norm),
            lm_head=torch.nn.Identity(),
        )
        outputs = SimpleNamespace(hidden_states=(torch.ones(1),) * 3)

        layer_logits(model, outputs, layer=1, num_layers=2)
        layer_logits(model, outputs, layer=2, num_layers=2)

        self.assertEqual(norm.calls, 1)


if __name__ == "__main__":
    unittest.main()

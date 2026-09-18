import unittest
from types import SimpleNamespace

import torch

from scripts.original.overlap.sdar_layer_overlap import (
    candidate_mask,
    layer_logits,
    overlap_metrics,
    token_prediction_metrics,
)


class SDARLayerOverlapTest(unittest.TestCase):
    def test_candidate_mask_uses_runtime_ceil_budget(self):
        selected = candidate_mask(
            torch.tensor([[True, True, True]]),
            torch.tensor([[0.1, 0.9, 0.8]]),
            ratio=0.5,
        )

        self.assertEqual(selected.tolist(), [[False, True, True]])

    def test_overlap_metrics(self):
        metrics = overlap_metrics(
            torch.tensor([[True, True, False, False]]),
            torch.tensor([[False, True, True, False]]),
        )

        self.assertEqual(metrics["overlap_count"], 1)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["jaccard"], 1 / 3)

    def test_token_prediction_metrics_uses_only_transfer_positions(self):
        final = torch.tensor([[[4.0, 1.0, 0.0], [0.0, 3.0, 1.0]]])
        shallow = torch.tensor([[[3.0, 2.0, 1.0], [4.0, 1.0, 0.0]]])
        transfer = torch.tensor([[True, False]])
        selected = final[transfer].float()
        log_probs = selected.log_softmax(-1)

        metrics = token_prediction_metrics(
            shallow, transfer, selected.argmax(-1), log_probs, log_probs.exp()
        )

        self.assertEqual(metrics["count"], 1)
        self.assertEqual(metrics["top1"], 1.0)
        self.assertEqual(metrics["top5"], 1.0)
        self.assertGreaterEqual(metrics["kl"], 0.0)

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
            model=SimpleNamespace(norm=norm), lm_head=torch.nn.Identity()
        )
        outputs = SimpleNamespace(hidden_states=(torch.ones(1),) * 3)

        layer_logits(model, outputs, layer=1, num_layers=2)
        layer_logits(model, outputs, layer=2, num_layers=2)

        self.assertEqual(norm.calls, 1)


if __name__ == "__main__":
    unittest.main()

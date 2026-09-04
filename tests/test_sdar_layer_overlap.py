import unittest

import torch

from scripts.analyze_sdar_layer_overlap import candidate_mask, overlap_metrics


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


if __name__ == "__main__":
    unittest.main()

import unittest

import torch

from scripts.analyze_llada_query_losa_correlation import summarize_records
from src.sparse.block_cache_sparse_dlm_patch import (
    _losa_active_indices,
    _new_losa_state,
)


def _record(sample, confidence, delta, transfer):
    positions = list(range(len(confidence)))
    return {
        "sample": sample,
        "block": 0,
        "step": sample,
        "mask_positions": positions,
        "decoded_positions": [],
        "query": {
            "mode": "fresh",
            "selected_positions": positions,
            "selected_mask_positions": positions,
            "confidence": {
                str(position): value for position, value in enumerate(confidence)
            },
        },
        "losa": [
            {
                "layer": 1,
                "positions": positions,
                "valid_positions": positions,
                "missing_positions": [],
                "delta": {str(position): value for position, value in enumerate(delta)},
                "active_positions": positions[-2:],
            }
        ],
        "transfer_positions": [transfer],
    }


class QueryLosaCorrelationTest(unittest.TestCase):
    def test_losa_metadata_uses_global_validity_and_local_active_indices(self):
        query = torch.zeros(1, 2, 3, 4)
        query[:, :, 2] = 1
        state = _new_losa_state(query, block_length=4)
        state["valid"][0, [0, 2]] = True

        active, valid, delta = _losa_active_indices(
            state,
            query,
            torch.tensor([0, 1, 2]),
            active_topk=2,
            return_metadata=True,
        )

        self.assertEqual(active.tolist(), [1, 2])
        self.assertEqual(valid.tolist(), [True, False, True])
        self.assertEqual(delta[[0, 2]].tolist(), [0.0, 1.0])
        self.assertTrue(torch.isnan(delta[1]))

    def test_perfectly_aligned_scores_have_perfect_metrics(self):
        records = [
            _record(0, [0.1, 0.2, 0.9], [1.0, 2.0, 9.0], 2),
            _record(1, [0.2, 0.3, 0.8], [2.0, 3.0, 8.0], 2),
        ]

        summary = summarize_records(records, selection_layer=1, active_topk=2)
        primary = summary["primary_selection_layer"]

        self.assertAlmostEqual(primary["spearman_confidence_delta"], 1.0)
        self.assertAlmostEqual(primary["mean_topk_jaccard"], 1.0)
        self.assertEqual(primary["nontrivial_topk_steps"], 2)
        self.assertAlmostEqual(primary["mean_nontrivial_topk_jaccard"], 1.0)
        self.assertAlmostEqual(primary["transfer_auroc_confidence"], 1.0)
        self.assertAlmostEqual(primary["transfer_auroc_delta"], 1.0)
        self.assertAlmostEqual(primary["transfer_auroc_confidence_step_rank"], 1.0)
        self.assertAlmostEqual(primary["transfer_auroc_delta_step_rank"], 1.0)
        self.assertAlmostEqual(primary["transfer_auroc_combined_rank_sum"], 1.0)


if __name__ == "__main__":
    unittest.main()

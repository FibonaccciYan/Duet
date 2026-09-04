import unittest
from unittest.mock import patch

import torch

from scripts.analyze_llada_query_losa_correlation import summarize_records
from src.sparse.llada_patch import (
    _losa_active_indices,
    _losa_key_energy,
    _new_losa_state,
    _queue_losa_active_update,
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
    def test_key_energy_uses_uniform_samples_and_expands_gqa_heads(self):
        key = torch.arange(1, 9, dtype=torch.float32).reshape(1, 1, 8, 1)

        energy = _losa_key_energy(key, num_key_value_groups=2, sample_count=2)

        expected = torch.tensor([[(1.0**2 + 5.0**2) / 2]]).expand(2, 1)
        torch.testing.assert_close(energy, expected)

    def test_losa_query_reference_updates_exactly_with_prefix_state(self):
        context = {"pending_losa_queries": [], "pending_losa": []}
        positions = torch.tensor([1, 4, 7])
        query = torch.arange(24.0).reshape(1, 2, 3, 4)
        active = torch.tensor([0, 2])
        prefix_output = torch.zeros(1, 2, 2, 4)
        prefix_lse = torch.zeros(1, 2, 2)

        _queue_losa_active_update(
            context, 3, positions, query, active, prefix_output, prefix_lse
        )

        query_layer, query_positions, queued_query = context[
            "pending_losa_queries"
        ][0]
        prefix_layer, prefix_positions, _, _ = context["pending_losa"][0]
        self.assertEqual(query_layer, prefix_layer)
        self.assertEqual(query_layer, 3)
        torch.testing.assert_close(query_positions, torch.tensor([1, 7]))
        torch.testing.assert_close(query_positions, prefix_positions)
        torch.testing.assert_close(queued_query, query[:, :, [0, 2]])

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

    def test_losa_triton_fast_path_preserves_metadata_contract(self):
        query = torch.zeros(1, 2, 3, 4)
        state = _new_losa_state(query, block_length=3)
        state["valid"].fill_(True)
        state["fully_valid"] = True

        with patch(
            "src.sparse.llada_patch.losa_query_delta",
            return_value=torch.tensor([0.1, 0.9, 0.2]),
        ):
            active, valid, delta = _losa_active_indices(
                state,
                query,
                torch.arange(3),
                active_topk=2,
                return_metadata=True,
            )

        self.assertEqual(active.tolist(), [1, 2])
        self.assertTrue(valid.all())
        torch.testing.assert_close(delta, torch.tensor([0.1, 0.9, 0.2]))

    def test_key_diagonal_score_weights_query_drift(self):
        query = torch.zeros(1, 1, 2, 2)
        query[0, 0, 0, 0] = 2
        query[0, 0, 1, 1] = 1
        state = _new_losa_state(query, block_length=2)
        state["valid"].fill_(True)
        state["key_energy"] = torch.tensor([[0.01, 10.0]])

        active = _losa_active_indices(
            state,
            query,
            torch.arange(2),
            active_topk=1,
            score_mode="key_diag",
        )

        self.assertEqual(active.tolist(), [1])

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

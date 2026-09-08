import types
import unittest
from unittest.mock import patch

from src.sparse import patch_model, resolve_model_family


class SparseApiTest(unittest.TestCase):
    def test_api_routes_all_feature_switches_to_llada(self):
        model = types.SimpleNamespace(
            config=types.SimpleNamespace(model_type="llada2_moe")
        )
        with patch(
            "src.sparse.api.patch_llada_model"
        ) as patch_llada, patch(
            "src.sparse.api.patch_moe_experts"
        ) as patch_moe:
            patch_model(
                model,
                model_name="llada",
                query_sparse=True,
                prefix_sparse=True,
                losa=True,
                selection_layer=3,
                moe_expert_patch=True,
            )

        kwargs = patch_llada.call_args.kwargs
        self.assertTrue(kwargs["query_sparse"])
        self.assertTrue(kwargs["prefix_sparse"])
        self.assertTrue(kwargs["losa"])
        self.assertEqual(kwargs["selection_layer"], 3)
        self.assertEqual(kwargs["query_dense_threshold"], 20)
        self.assertEqual(kwargs["query_min_prefix_length"], 24576)
        self.assertEqual(kwargs["prefix_min_prefix_length"], 4096)
        self.assertEqual(kwargs["prefix_chunk_size"], 1024)
        self.assertEqual(model.config.llada_sparse_config["prefix_chunk_size"], 1024)
        patch_moe.assert_called_once_with(model)
        self.assertEqual(model._sparse_patch_family, "llada")

    def test_explicit_model_name_validates_checkpoint(self):
        model = types.SimpleNamespace(config=types.SimpleNamespace(model_type="sdar"))
        with self.assertRaisesRegex(ValueError, "checkpoint model_type"):
            resolve_model_family(model, "llada")

    def test_api_routes_prefix_sparse_and_losa_to_sdar(self):
        model = types.SimpleNamespace(config=types.SimpleNamespace(model_type="sdar"))
        with patch(
            "src.sparse.api.patch_sdar_model"
        ) as patch_sdar, patch("src.sparse.api.patch_moe_experts") as patch_moe:
            patch_model(
                model,
                model_name="sdar",
                query_sparse=False,
                prefix_sparse=True,
                prefix_token_budget=128,
                losa=True,
                losa_active_topk=7,
            )

        kwargs = patch_sdar.call_args.kwargs
        self.assertFalse(kwargs["query_sparse"])
        self.assertTrue(kwargs["prefix_sparse"])
        self.assertEqual(kwargs["prefix_token_budget"], 128)
        self.assertTrue(kwargs["losa"])
        self.assertEqual(kwargs["losa_active_topk"], 7)
        self.assertEqual(kwargs["selection_layer"], 5)
        self.assertEqual(kwargs["refresh_step"], -1)
        self.assertEqual(kwargs["prefix_min_prefix_length"], 24576)
        self.assertEqual(kwargs["prefix_chunk_size"], 1024)
        self.assertEqual(model.config.sdar_sparse_config["prefix_chunk_size"], 1024)
        patch_moe.assert_not_called()
        self.assertEqual(model._sparse_patch_family, "sdar")


if __name__ == "__main__":
    unittest.main()

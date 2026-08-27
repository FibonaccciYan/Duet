import types
import unittest
from unittest.mock import patch

from src.sparse import patch_model, resolve_model_family


class SparseCoreTest(unittest.TestCase):
    def test_core_routes_all_feature_switches_to_llada(self):
        model = types.SimpleNamespace(
            config=types.SimpleNamespace(model_type="llada2_moe")
        )
        with patch(
            "src.sparse.block_cache_sparse_dlm_patch.patch_llada_model"
        ) as patch_llada, patch(
            "src.sparse.core.patch_moe_experts"
        ) as patch_moe:
            patch_model(
                model,
                model_name="llada",
                query_sparse=True,
                prefix_sparse=True,
                losa=True,
                moe_expert_patch=True,
            )

        kwargs = patch_llada.call_args.kwargs
        self.assertTrue(kwargs["query_sparse"])
        self.assertTrue(kwargs["prefix_sparse"])
        self.assertTrue(kwargs["losa"])
        patch_moe.assert_called_once_with(model)
        self.assertEqual(model._sparse_patch_family, "llada")

    def test_explicit_model_name_validates_checkpoint(self):
        model = types.SimpleNamespace(config=types.SimpleNamespace(model_type="sdar"))
        with self.assertRaisesRegex(ValueError, "checkpoint model_type"):
            resolve_model_family(model, "llada")
        with self.assertRaisesRegex(ValueError, "query_sparse only"):
            patch_model(model, model_name="sdar", prefix_sparse=True)


if __name__ == "__main__":
    unittest.main()

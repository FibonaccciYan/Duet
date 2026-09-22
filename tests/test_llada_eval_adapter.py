import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch


EVAL_ROOT = Path(__file__).parents[1] / "eval_instruct"
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from lm_eval.models import llada, sdar


class LLaDAEvalAdapterTest(unittest.TestCase):
    def test_qk_then_raw_are_isolated_and_forwarded(self):
        import src.reference.sparse.sparse_ops as ops
        original = ops._prefix_indices
        for adapter, typ in ((llada.LLaDA, "llada2_moe"), (sdar.SDAR, "sdar")):
            model = MagicMock()
            model.eval.return_value = model
            model.config.model_type = typ
            accelerator = SimpleNamespace(num_processes=1, device=torch.device("cpu"))
            with (
                patch.object(llada, "Accelerator", return_value=accelerator),
                patch.object(llada.transformers.AutoModelForCausalLM, "from_pretrained", return_value=model),
                patch.object(llada.transformers.AutoTokenizer, "from_pretrained"),
                patch.object(llada, "patch_method") as install,
            ):
                for selector in ("qk", "qk_tc", "raw_l1"):
                    adapter(pretrained="fake", device="cpu", method="sparse",
                            implementation="optimized", prefix_selector=selector)
                    self.assertEqual(install.call_args.kwargs["prefix_selector"], selector)
                    self.assertEqual(install.call_args.kwargs["method"], "sparse_optimized")
                for legacy in ("adamas", "hadamard_qk"):
                    adapter(pretrained="fake", device="cpu", prefix_selector=legacy)
                    self.assertEqual(install.call_args.kwargs["prefix_selector"], legacy)
            self.assertIs(ops._prefix_indices, original)

    def test_query_losa_union_is_forwarded_to_sparse_patch(self):
        model = MagicMock()
        model.eval.return_value = model
        model.config.model_type = "llada2_moe"
        accelerator = SimpleNamespace(num_processes=1, device=torch.device("cpu"))

        with (
            patch.object(llada, "Accelerator", return_value=accelerator),
            patch.object(
                llada.transformers.AutoModelForCausalLM,
                "from_pretrained",
                return_value=model,
            ),
            patch.object(llada.transformers.AutoTokenizer, "from_pretrained"),
            patch.object(llada, "patch_method") as method_patch,
        ):
            llada.LLaDA(
                pretrained="fake",
                device="cpu",
                sparse_dlm=True,
                query_losa_union=True,
            )

        self.assertEqual(method_patch.call_args.kwargs["method"], "sparse")
        self.assertTrue(method_patch.call_args.kwargs["query_losa_union"])

    def test_dense_method_uses_common_moe_expert_patch(self):
        model = MagicMock()
        model.eval.return_value = model
        model.config.model_type = "llada2_moe"
        accelerator = SimpleNamespace(
            num_processes=1,
            device=torch.device("cpu"),
        )

        with (
            patch.object(llada, "Accelerator", return_value=accelerator),
            patch.object(
                llada.transformers.AutoModelForCausalLM,
                "from_pretrained",
                return_value=model,
            ),
            patch.object(llada.transformers.AutoTokenizer, "from_pretrained"),
            patch.object(llada, "patch_method") as method_patch,
            patch.object(llada, "patch_moe_experts", return_value=12) as moe_patch,
        ):
            adapter = llada.LLaDA(
                pretrained="fake", device="cpu", method="dense"
            )

        moe_patch.assert_called_once_with(model)
        self.assertEqual(method_patch.call_args.kwargs["method"], "dense")
        self.assertEqual(adapter.method, "dense")
        self.assertEqual(adapter.threshold, 0.7)
        self.assertEqual(adapter.editing_threshold, 0.5)

    def test_focus_method_uses_its_own_defaults(self):
        model = MagicMock()
        model.eval.return_value = model
        model.config.model_type = "llada2_moe"
        accelerator = SimpleNamespace(num_processes=1, device=torch.device("cpu"))

        with (
            patch.object(llada, "Accelerator", return_value=accelerator),
            patch.object(
                llada.transformers.AutoModelForCausalLM,
                "from_pretrained",
                return_value=model,
            ),
            patch.object(llada.transformers.AutoTokenizer, "from_pretrained"),
            patch.object(llada, "patch_method") as method_patch,
            patch.object(llada, "patch_moe_experts"),
        ):
            adapter = llada.LLaDA(
                pretrained="fake",
                device="cpu",
                method="focus",
                focus_alpha=1.75,
            )

        self.assertEqual(method_patch.call_args.kwargs["method"], "focus")
        self.assertEqual(method_patch.call_args.kwargs["alpha"], 1.75)
        self.assertEqual(adapter.threshold, 0.7)
        self.assertEqual(adapter.editing_threshold, 0.5)

    def test_sdar_dynamic_remasking_defaults_to_point_95(self):
        model = MagicMock()
        model.eval.return_value = model
        model.config.model_type = "sdar"
        accelerator = SimpleNamespace(num_processes=1, device=torch.device("cpu"))

        with (
            patch.object(llada, "Accelerator", return_value=accelerator),
            patch.object(
                llada.transformers.AutoModelForCausalLM,
                "from_pretrained",
                return_value=model,
            ),
            patch.object(llada.transformers.AutoTokenizer, "from_pretrained"),
            patch.object(llada, "patch_method"),
        ):
            adapter = sdar.SDAR(
                pretrained="fake",
                device="cpu",
                method="focus",
            )

        self.assertEqual(adapter.threshold, 0.95)
        self.assertEqual(adapter.remasking_strategy, "low_confidence_dynamic")

if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch


EVAL_ROOT = Path(__file__).parents[1] / "eval_instruct"
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from lm_eval.models import llada


class LLaDAEvalAdapterTest(unittest.TestCase):
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
        self.assertEqual(adapter.threshold, 0.95)
        self.assertEqual(adapter.editing_threshold, 0.9)

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
        self.assertEqual(adapter.threshold, 0.95)
        self.assertEqual(adapter.editing_threshold, 0.9)

    def test_integrated_runtime_modes_use_their_own_patchers(self):
        model = MagicMock()
        model.eval.return_value = model
        model.config.model_type = "llada2_moe"
        accelerator = SimpleNamespace(num_processes=1, device=torch.device("cpu"))

        for mode, patch_name in (
            ("dense", "patch_dense_model"),
            ("losa", "patch_losa_model"),
            ("focus", "patch_focus_model"),
        ):
            with (
                self.subTest(mode=mode),
                patch.object(llada, "Accelerator", return_value=accelerator),
                patch.object(
                    llada.transformers.AutoModelForCausalLM,
                    "from_pretrained",
                    return_value=model,
                ),
                patch.object(llada.transformers.AutoTokenizer, "from_pretrained"),
                patch.object(llada, "patch_model") as sparse_patch,
                patch.object(llada, patch_name) as integrated_patch,
            ):
                llada.LLaDA(
                    pretrained="fake",
                    device="cpu",
                    runtime_mode=mode,
                    sparse_dlm=False,
                    moe_expert_patch=False,
                )

            integrated_patch.assert_called_once()
            sparse_patch.assert_not_called()


if __name__ == "__main__":
    unittest.main()

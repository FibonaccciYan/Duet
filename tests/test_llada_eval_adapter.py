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
    def test_dense_llada_still_applies_moe_expert_patch(self):
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
            patch.object(llada, "patch_model") as sparse_patch,
            patch.object(llada, "patch_moe_experts", return_value=12) as moe_patch,
        ):
            llada.LLaDA(pretrained="fake", device="cpu", sparse_dlm=False)

        moe_patch.assert_called_once_with(model)
        sparse_patch.assert_not_called()


if __name__ == "__main__":
    unittest.main()

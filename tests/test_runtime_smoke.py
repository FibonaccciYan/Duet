from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch

from src.focus import (
    attention_importance,
    patch_model as patch_focus_model,
    select_retained_positions,
)
from src.focus.model import focus_forward
from src.kernels.losa import GQAMode, adapted_quest_attention_step, losa_attention_step
from src.losa.generation import transfer_llada
from src.runtime import load_runtime


class IntegratedRuntimeSmokeTest(unittest.TestCase):
    def test_runtime_dispatch(self):
        self.assertEqual(type(load_runtime("dense", family="llada")).__name__, "DenseRuntime")
        self.assertEqual(type(load_runtime("losa", family="llada")).__name__, "LoSARuntime")
        self.assertEqual(type(load_runtime("focus", family="llada")).__name__, "FocusRuntime")

    def test_focus_exposes_sparse_compatible_patch_entrypoint(self):
        model = types.SimpleNamespace(
            config=types.SimpleNamespace(model_type="llada2_moe"),
            generate=lambda **_: None,
        )
        patched = patch_focus_model(model, model_name="llada", alpha=1.75)
        self.assertIs(patched, model)
        self.assertEqual(model._focus_patch_family, "llada")
        self.assertEqual(model.config.focus_config, {"alpha": 1.75, "family": "llada"})

    def test_focus_attention_importance_shape(self):
        query = torch.randn(1, 4, 6, 8)
        key = torch.randn(1, 2, 6, 8)
        positions = torch.tensor([1, 3, 4])
        scores = attention_importance(query, key, positions)
        self.assertEqual(scores.shape, positions.shape)
        self.assertTrue(torch.isfinite(scores).all())

    def test_focus_selector_evicts_after_warmup_progress(self):
        retained = select_retained_positions(
            torch.full((4,), 63),
            63,
            torch.zeros(4),
            torch.tensor([0.0, 0.0, 0.0, 10.0]),
            alpha=1.0,
            average_decoded_tokens=1.0,
            block_progress=3,
        )
        self.assertEqual(retained.tolist(), [2, 3])

    @unittest.skipUnless(
        Path("/data0/ysy/models/LLaDA2.1-mini/modeling_llada2_moe.py").exists(),
        "LLaDA2.1 checkpoint source is unavailable",
    )
    def test_focus_all_retained_matches_official_llada21_layers(self):
        root = Path("/data0/ysy/models/LLaDA2.1-mini")
        package = types.ModuleType("test_llada21_checkpoint")
        package.__path__ = [str(root)]
        sys.modules[package.__name__] = package
        for name in ("configuration_llada2_moe", "modeling_llada2_moe"):
            qualified = f"{package.__name__}.{name}"
            spec = importlib.util.spec_from_file_location(qualified, root / f"{name}.py")
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[qualified] = module
            spec.loader.exec_module(module)
        config_module = sys.modules[f"{package.__name__}.configuration_llada2_moe"]
        model_module = sys.modules[f"{package.__name__}.modeling_llada2_moe"]
        config = config_module.LLaDA2MoeConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            num_experts=None,
            num_shared_experts=0,
            pad_token_id=0,
            partial_rotary_factor=1.0,
            attention_dropout=0.0,
            use_qkv_bias=False,
            use_bias=False,
        )
        config._attn_implementation = "sdpa"
        torch.manual_seed(0)
        model = model_module.LLaDA2MoeModelLM(config).eval()
        tokens = torch.tensor([[1, 63, 3, 63]])
        positions = torch.arange(4).unsqueeze(0)
        dense = focus_forward(
            model,
            family="llada",
            input_ids=tokens,
            position_ids=positions,
            mask_id=-1,
            alpha=100,
            average_decoded_tokens=1,
            block_progress=-1,
        )
        retained = focus_forward(
            model,
            family="llada",
            input_ids=tokens,
            position_ids=positions,
            mask_id=63,
            alpha=100,
            average_decoded_tokens=1,
            block_progress=-1,
        )
        self.assertEqual(retained.positions.tolist(), [0, 1, 2, 3])
        torch.testing.assert_close(retained.logits, dense.logits, atol=1e-6, rtol=1e-5)

    def test_losa_step_shapes(self):
        q = torch.randn(4, 4, 8)
        k_prefix = torch.randn(16, 2, 8)
        v_prefix = torch.randn(16, 2, 8)
        k_block = torch.randn(4, 2, 8)
        v_block = torch.randn(4, 2, 8)
        result = losa_attention_step(
            q,
            k_prefix,
            v_prefix,
            k_block,
            v_block,
            page_size=4,
            token_budget=8,
            active_count=2,
            mode=GQAMode.PER_QUERY_HEAD,
        )
        self.assertEqual(result.output.shape, q.shape)
        self.assertEqual(result.lse.shape, q.shape[:-1])

    def test_adapted_quest_step_shapes(self):
        q = torch.randn(4, 4, 8)
        k_prefix = torch.randn(16, 2, 8)
        v_prefix = torch.randn(16, 2, 8)
        k_block = torch.randn(4, 2, 8)
        v_block = torch.randn(4, 2, 8)
        result = adapted_quest_attention_step(
            q,
            k_prefix,
            v_prefix,
            k_block,
            v_block,
            page_size=4,
            token_budget=8,
            mode=GQAMode.PER_QUERY_HEAD,
        )
        self.assertEqual(result.output.shape, q.shape)

    def test_llada_zero_editing_threshold_still_edits_generated_tokens(self):
        block = torch.tensor([[3, 4]])
        old_block = block.clone()
        logits = torch.full((1, 2, 8), -10.0)
        logits[0, 0, 7] = 10.0
        logits[0, 1, 6] = 10.0
        updated, transfer = transfer_llada(
            object(),
            block,
            old_block,
            torch.tensor([True, False]),
            torch.tensor([[False, False]]),
            logits,
            temperature=0.0,
            top_p=None,
            top_k=None,
            threshold=0.5,
            editing_threshold=0.0,
            num_to_transfer=1,
        )
        self.assertEqual(updated.tolist(), [[3, 6]])
        self.assertEqual(transfer.tolist(), [[False, True]])

    def test_integrated_runtimes_have_no_sibling_workspace_dependency(self):
        root = Path(__file__).resolve().parents[1]
        sources = [
            root / "src/losa/api.py",
            root / "src/losa/attention_patch.py",
            root / "src/losa/generation.py",
            root / "src/losa/operators.py",
            root / "src/losa/triton_ops.py",
            root / "src/focus/api.py",
            root / "src/focus/algorithm.py",
            root / "src/focus/generation.py",
            root / "src/focus/model.py",
        ]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in sources)
        self.assertNotIn("losa_accuracy_prep", combined)
        self.assertNotIn("/data0/gs/LoSA-original", combined)
        self.assertNotIn("/data0/gs/FOCUS", combined)


if __name__ == "__main__":
    unittest.main()

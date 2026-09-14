from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import torch
import unittest


def _load_llada_model_class():
    root = Path("/data0/ysy/models/LLaDA2.1-mini")
    if not root.exists():
        raise unittest.SkipTest("LLaDA2.1 checkpoint source is unavailable")
    package = types.ModuleType("test_focus_v2_llada21_checkpoint")
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package
    modules = {}
    for name in ("configuration_llada2_moe", "modeling_llada2_moe"):
        qualified = f"{package.__name__}.{name}"
        spec = importlib.util.spec_from_file_location(qualified, root / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return modules["configuration_llada2_moe"].LLaDA2MoeConfig, modules["modeling_llada2_moe"].LLaDA2MoeModelLM


class FocusV2ReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.focus.model import focus_forward
        from src.focus_v2.model import focus_v2_forward

        cls.focus_forward = staticmethod(focus_forward)
        cls.focus_v2_forward = staticmethod(focus_v2_forward)
        config_class, model_class = _load_llada_model_class()
        config = config_class(
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
            rope_scaling=None,
        )
        config._attn_implementation = "sdpa"
        torch.manual_seed(17)
        cls.model = model_class(config).eval()
        cls.tokens = torch.tensor([[1, 63, 62, 7, 62, 11]])
        cls.positions = torch.arange(cls.tokens.shape[1]).unsqueeze(0)

    def test_no_eviction_matches_reference(self):
        kwargs = dict(
            family="llada",
            input_ids=self.tokens,
            position_ids=self.positions,
            prefix_cache=(),
            mask_id=-1,
            alpha=1.5,
            average_decoded_tokens=1.0,
            block_progress=-1,
        )
        with torch.no_grad():
            old = self.focus_forward(model=self.model, **kwargs)
            new = self.focus_v2_forward(self.model, **kwargs)
        self.assertEqual(new.positions.tolist(), old.positions.tolist())
        self.assertTrue(torch.isfinite(new.logits).all())
        torch.testing.assert_close(new.logits, old.logits, atol=2e-5, rtol=2e-5)

    def test_selection_matches_reference_on_first_step(self):
        kwargs = dict(
            family="llada",
            input_ids=self.tokens,
            position_ids=self.positions,
            prefix_cache=(),
            mask_id=62,
            alpha=1.0,
            average_decoded_tokens=1.0,
            block_progress=-1,
        )
        with torch.no_grad():
            old = self.focus_forward(model=self.model, **kwargs)
            new = self.focus_v2_forward(self.model, **kwargs)
        self.assertEqual(new.positions.tolist(), old.positions.tolist())
        self.assertEqual(int(new.positions.numel()), 6)
        self.assertEqual(new.trace[0]["stage"], "selection")
        self.assertEqual(new.trace[0]["mask_positions"], 2)
        torch.testing.assert_close(
            new.logits.index_select(1, new.positions),
            old.logits.index_select(1, old.positions),
            atol=2e-5,
            rtol=2e-5,
        )

    def test_patch_model_contract_without_moe(self):
        from src.focus_v2 import patch_model

        model = self.model
        # A fresh model object is unnecessary for this contract test; v2 marks
        # its own attributes separately from the old runtime.
        previous_generate = model.generate
        patch_model(model, model_name="llada", alpha=1.25, moe_expert_patch=False)
        self.assertEqual(model._focus_v2_patch_family, "llada")
        self.assertEqual(model._focus_v2_alpha, 1.25)
        self.assertEqual(model.config.focus_v2_config["alpha"], 1.25)
        self.assertIsNot(model.generate, previous_generate)


def _load_sdar_model_class():
    from torch.nn import functional as F

    from src.focus_v2.compat import install_runtime_compat

    install_runtime_compat()
    root = Path("/data0/ysy/models/SDAR-8B-Chat-b32")
    if not (root / "modeling_sdar.py").exists():
        raise unittest.SkipTest("SDAR checkpoint source is unavailable")
    package = types.ModuleType("test_focus_v2_sdar_checkpoint")
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package
    modules = {}
    for name in ("configuration_sdar", "modeling_sdar"):
        qualified = f"{package.__name__}.{name}"
        spec = importlib.util.spec_from_file_location(qualified, root / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
        modules[name] = module

    def cpu_flash(q, k, v, causal=False, softmax_scale=None, **_):
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        groups = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=causal, scale=softmax_scale
        ).transpose(1, 2)

    def cpu_rms_norm(norm, value):
        dtype = value.dtype
        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(-1, keepdim=True) + norm.variance_epsilon
        )
        return normalized.to(dtype) * norm.weight

    modeling = modules["modeling_sdar"]
    modeling.flash_attn_func = cpu_flash
    modeling.SDARRMSNorm.forward = cpu_rms_norm
    return modules["configuration_sdar"].SDARConfig, modeling.SDARForCausalLM


class FocusV2SdarReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.focus.model import focus_forward
        from src.focus_v2.model import focus_v2_forward

        cls.focus_forward = staticmethod(focus_forward)
        cls.focus_v2_forward = staticmethod(focus_v2_forward)
        config_class, model_class = _load_sdar_model_class()
        config = config_class(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            pad_token_id=0,
            attention_dropout=0.0,
            use_sliding_window=False,
        )
        config._attn_implementation = "sdpa"
        torch.manual_seed(23)
        cls.model = model_class(config).eval()
        cls.tokens = torch.tensor([[1, 63, 62, 7, 62, 11]])
        cls.positions = torch.arange(cls.tokens.shape[1]).unsqueeze(0)

    def _run(self, mask_id):
        kwargs = dict(
            family="sdar",
            input_ids=self.tokens,
            position_ids=self.positions,
            prefix_cache=(),
            mask_id=mask_id,
            alpha=1.0,
            average_decoded_tokens=1.0,
            block_progress=-1,
        )
        with torch.no_grad():
            old = self.focus_forward(model=self.model, **kwargs)
            new = self.focus_v2_forward(self.model, **kwargs)
        return old, new

    def test_no_eviction_matches_reference(self):
        old, new = self._run(-1)
        self.assertEqual(new.positions.tolist(), old.positions.tolist())
        torch.testing.assert_close(new.logits, old.logits, atol=2e-5, rtol=2e-5)

    def test_selection_matches_reference_on_first_step(self):
        old, new = self._run(62)
        self.assertEqual(new.positions.tolist(), old.positions.tolist())
        torch.testing.assert_close(
            new.logits.index_select(1, new.positions),
            old.logits.index_select(1, old.positions),
            atol=2e-5,
            rtol=2e-5,
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch
from torch.nn import functional as F

from src.reference.focus.model import focus_forward


class FocusSDAROfficialModelTest(unittest.TestCase):
    def test_all_retained_matches_official_sdar_layers(self):
        root = Path("/data0/ysy/models/SDAR-8B-Chat-b32")
        if not (root / "modeling_sdar.py").exists():
            self.skipTest("SDAR checkpoint source is unavailable")
        package = types.ModuleType("test_sdar_checkpoint")
        package.__path__ = [str(root)]
        sys.modules[package.__name__] = package
        try:
            for name in ("configuration_sdar", "modeling_sdar"):
                qualified = f"{package.__name__}.{name}"
                spec = importlib.util.spec_from_file_location(qualified, root / f"{name}.py")
                assert spec is not None and spec.loader is not None
                module = importlib.util.module_from_spec(spec)
                sys.modules[qualified] = module
                spec.loader.exec_module(module)
        except ImportError as exc:
            self.skipTest(f"active transformers build cannot load SDAR: {exc}")

        config_module = sys.modules[f"{package.__name__}.configuration_sdar"]
        model_module = sys.modules[f"{package.__name__}.modeling_sdar"]

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

        model_module.flash_attn_func = cpu_flash
        model_module.SDARRMSNorm.forward = cpu_rms_norm
        config = config_module.SDARConfig(
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
        torch.manual_seed(0)
        model = model_module.SDARForCausalLM(config).eval()
        tokens = torch.tensor([[1, 63, 3, 63]])
        positions = torch.arange(4).unsqueeze(0)
        dense = focus_forward(
            model, family="sdar", input_ids=tokens, position_ids=positions,
            mask_id=-1, alpha=100, average_decoded_tokens=1, block_progress=-1,
        )
        retained = focus_forward(
            model, family="sdar", input_ids=tokens, position_ids=positions,
            mask_id=63, alpha=100, average_decoded_tokens=1, block_progress=-1,
        )
        self.assertEqual(retained.positions.tolist(), [0, 1, 2, 3])
        torch.testing.assert_close(retained.logits, dense.logits, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()

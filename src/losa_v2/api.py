from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from src.losa.generation import load_model_and_tokenizer

from .attention_patch import install_losa_v2_attention
from .generation import block_diffusion_generate

try:  # pragma: no cover - optional matched MoE optimization
    from src.losa.moe_patch import patch_moe_experts
except Exception:  # pragma: no cover
    patch_moe_experts = None


DEFAULT_MODEL_PATHS = {
    "llada": "/data0/ysy/models/LLaDA2.1-mini",
    "sdar": "/data0/ysy/models/SDAR-8B-Chat-b32",
}


@dataclass
class LoSAV2Runtime:
    family: str
    model_path: str | None = None
    dtype: str | None = None
    attn_implementation: str = "sdpa"
    moe_expert_patch: bool = False
    losa_page_size: int = 16
    losa_token_budget: int = 256
    losa_active_topk: int = 5
    losa_gqa_mode: str = "group_mean"
    losa_backend: str = "auto"
    losa_trace_detail: bool = False
    model: Any | None = field(init=False, default=None)
    tokenizer: Any | None = field(init=False, default=None)
    moe_patch_report: Any | None = field(init=False, default=None)

    def load(self):
        model_path = self.model_path or DEFAULT_MODEL_PATHS[self.family]
        dtype_name = self.dtype or (
            "bfloat16" if self.family == "llada" else "float16"
        )
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.family,
            model_path=model_path,
            dtype=dtype_name,
            attn_implementation=self.attn_implementation,
        )
        install_losa_v2_attention(self.model, self.family)
        if self.moe_expert_patch:
            if patch_moe_experts is None:
                raise RuntimeError("the optional Triton MoE backend is unavailable")
            self.moe_patch_report = patch_moe_experts(
                self.model, family=self.family
            )
        return self.model, self.tokenizer

    def generate(self, inputs: torch.Tensor, **kwargs):
        if self.model is None or self.tokenizer is None:
            self.load()
        assert self.model is not None
        install_losa_v2_attention(self.model, self.family)
        return block_diffusion_generate(
            self.model,
            family=self.family,
            inputs=inputs,
            use_losa=True,
            losa_page_size=self.losa_page_size,
            losa_token_budget=self.losa_token_budget,
            losa_active_topk=self.losa_active_topk,
            losa_gqa_mode=self.losa_gqa_mode,
            losa_backend=self.losa_backend,
            losa_trace_detail=self.losa_trace_detail,
            **kwargs,
        )


__all__ = ["LoSAV2Runtime"]

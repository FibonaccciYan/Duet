from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from src.reference.losa.generation import load_model_and_tokenizer

from .attention_patch import install_losa_optimized_attention
from .generation import block_diffusion_generate

try:  # pragma: no cover - optional matched MoE optimization
    from src.reference.losa.moe_patch import patch_moe_experts
except Exception:  # pragma: no cover
    patch_moe_experts = None


DEFAULT_MODEL_PATHS = {
    "llada": "/data0/ysy/models/LLaDA2.1-mini",
    "sdar": "/data0/ysy/models/SDAR-8B-Chat-b32",
}


@dataclass
class LoSAOptimizedRuntime:
    family: str
    model_path: str | None = None
    dtype: str | None = None
    attn_implementation: str = "sdpa"
    moe_expert_patch: bool = True
    losa_page_size: int = 16
    losa_token_budget: int = 256
    losa_active_topk: int = 5
    losa_gqa_mode: str = "group_mean"
    losa_backend: str = "auto"
    losa_trace_detail: bool = False
    losa_fused_state: bool = True
    exact_rope: bool = False
    shared_kv: bool = False
    graph_replay: bool = True
    verify_graph: bool = False
    kv_stats: bool = False
    kv_stats_chunk_size: int = 1024
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
        install_losa_optimized_attention(self.model, self.family)
        if self.family == "sdar":
            from src.kernels.optimized.sdar_norm import install
            self.model._versioned_direct_norm_count = install(self.model)
        if self.moe_expert_patch:
            if patch_moe_experts is None:
                raise RuntimeError("the optional Triton MoE backend is unavailable")
            self.moe_patch_report = patch_moe_experts(
                self.model, family=self.family
            )
        if self.moe_expert_patch:
            from src.kernels.optimized.small_moe import install_small_moe
            install_small_moe(self.model)
        if self.exact_rope:
            from src.kernels.optimized.rope_runtime import install
            self.model._versioned_rope_patch_count = install(self.model)
        return self.model, self.tokenizer

    def generate(self, inputs: torch.Tensor, **kwargs):
        if self.model is None or self.tokenizer is None:
            self.load()
        assert self.model is not None
        install_losa_optimized_attention(self.model, self.family)
        self.model._losa_optimized_shared_kv = self.shared_kv
        self.model._losa_optimized_graph_replay = self.graph_replay
        self.model._losa_optimized_verify_graph = self.verify_graph
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
            losa_fused_state=self.losa_fused_state,
            kv_stats=kwargs.pop("kv_stats", self.kv_stats),
            kv_stats_chunk_size=kwargs.pop("kv_stats_chunk_size", self.kv_stats_chunk_size),
            **kwargs,
        )

    def export_kv_stats(self, output_dir=None, include_heads=False):
        """Export the last request's statistics; returns None when disabled."""
        from .kv_stats import export_kv_stats
        return export_kv_stats(self.model, output_dir, include_heads)


def patch_model(model, model_name="auto", *, page_size=16, token_budget=256,
                active_topk=5, gqa_mode="group_mean", backend="auto",
                moe_expert_patch=True, graph_replay=True,
                kv_stats=False, kv_stats_chunk_size=1024, **kwargs):
    """Bind the optimized runtime to a model already loaded by an evaluator."""
    import types
    family = {"llada2_moe": "llada", "sdar": "sdar"}.get(model.config.model_type)
    if family is None or model_name not in ("auto", family):
        raise ValueError("LoSA model family mismatch")
    if min(page_size, token_budget, active_topk) <= 0:
        raise ValueError("LoSA budgets and sizes must be positive")
    if gqa_mode not in {"per_query_head", "group_mean", "group_max_score"}:
        raise ValueError("unsupported LoSA GQA mode")
    if backend not in {"auto", "torch", "triton"}:
        raise ValueError("unsupported LoSA backend")
    if kwargs:
        raise TypeError(f"Unsupported LoSA options: {sorted(kwargs)}")
    install_losa_optimized_attention(model, family)
    if family == "sdar":
        from src.kernels.optimized.sdar_norm import install
        install(model)
    if moe_expert_patch:
        if patch_moe_experts is None:
            raise RuntimeError("MoE backend unavailable")
        patch_moe_experts(model, family=family)
        from src.kernels.optimized.small_moe import install_small_moe
        install_small_moe(model)
    model._losa_optimized_graph_replay = bool(graph_replay)
    model._losa_optimized_shared_kv = False
    model._losa_optimized_verify_graph = False
    def generate(self, *args, **options):
        inputs = options.pop("inputs", args[0] if args else None)
        if inputs is None:
            raise ValueError("inputs required")
        return block_diffusion_generate(
            self, family=family, inputs=inputs, use_losa=True,
            losa_page_size=page_size, losa_token_budget=token_budget,
            losa_active_topk=active_topk, losa_gqa_mode=gqa_mode,
            losa_backend=backend, losa_fused_state=True,
            kv_stats=options.pop("kv_stats", kv_stats),
            kv_stats_chunk_size=options.pop("kv_stats_chunk_size", kv_stats_chunk_size),
            **options).tokens
    model.generate = types.MethodType(torch.inference_mode()(generate), model)
    return model


__all__ = ["LoSAOptimizedRuntime", "patch_model"]

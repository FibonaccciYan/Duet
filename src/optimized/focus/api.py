"""Public API for the optimized FOCUS runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
import types
from typing import Any

import torch

from src.reference.focus.api import resolve_model_family
from src.reference.losa.generation import load_model_and_tokenizer, set_seed

from .compat import install_runtime_compat
from .backend import resolve_attention_backend
from .generation import focus_optimized_generate

try:  # pragma: no cover - the packed MoE backend is optional on CPU-only hosts
    from src.reference.losa.moe_patch import patch_moe_experts
except Exception:  # pragma: no cover
    patch_moe_experts = None


DEFAULT_MODEL_PATHS = {
    "llada": "/data0/ysy/models/LLaDA2.1-mini",
    "sdar": "/data0/ysy/models/SDAR-8B-Chat-b32",
}


def _input_tensor(inputs: Any) -> torch.Tensor:
    if torch.is_tensor(inputs):
        return inputs
    if hasattr(inputs, "input_ids"):
        inputs = inputs.input_ids
    return torch.as_tensor(inputs)


def _focus_optimized_generate(self, *args, **kwargs):
    inputs = kwargs.pop("inputs", args[0] if args else None)
    if inputs is None:
        raise ValueError("optimized FOCUS generation requires `inputs`")
    return focus_optimized_generate(
        self,
        family=self._focus_optimized_patch_family,
        inputs=inputs,
        alpha=self._focus_optimized_alpha,
        attention_backend=self._focus_optimized_attention_backend,
        **kwargs,
    ).tokens


def patch_model(
    model: Any,
    model_name: str = "auto",
    *,
    alpha: float = 1.5,
    attention_backend: str = "auto",
    exact_rope: bool = False,
    moe_expert_patch: bool = True,
    **_: Any,
):
    """Install optimized FOCUS through the same model-patching contract as other modes.

    ``moe_expert_patch`` defaults to true because the speed comparison should
    use the same packed routed-MoE backend as dense and LoSA.  For non-MoE
    models the patch is a no-op.
    """

    family = resolve_model_family(model, model_name)
    attention_backend = resolve_attention_backend(family, attention_backend)
    if family == "sdar":
        from src.kernels.optimized.sdar_norm import install
        model._versioned_direct_norm_count = install(model)
    if alpha < 1.0:
        raise ValueError("optimized FOCUS alpha must be at least 1")

    if not hasattr(model, "_focus_optimized_original_generate"):
        model._focus_optimized_original_generate = model.generate
        model.generate = types.MethodType(_focus_optimized_generate, model)

    model._focus_optimized_patch_family = family
    if attention_backend not in {"sdpa", "flash"}:
        raise ValueError("unsupported attention backend")
    model._focus_optimized_attention_backend = attention_backend
    model._focus_optimized_alpha = float(alpha)
    model.config.focus_optimized_config = {
        "alpha": float(alpha),
        "attention_backend": attention_backend,
        "family": family,
        "moe_expert_patch_requested": bool(moe_expert_patch),
    }

    report = None
    if moe_expert_patch:
        if patch_moe_experts is None:
            raise RuntimeError("the optional packed MoE backend is unavailable")
        report = patch_moe_experts(model, family=family)
    if moe_expert_patch:
        from src.kernels.optimized.small_moe import install_small_moe
        install_small_moe(model)
    if exact_rope:
        from src.kernels.optimized.rope_runtime import install
        model._versioned_rope_patch_count = install(model)
    model._focus_optimized_moe_report = report
    return model


@dataclass
class FocusOptimizedRuntime:
    family: str
    model_path: str | None = None
    dtype: str | None = None
    attn_implementation: str = "sdpa"
    alpha: float = 1.5
    attention_backend: str = "auto"
    exact_rope: bool = False
    moe_expert_patch: bool = True
    model: Any | None = field(init=False, default=None)
    tokenizer: Any | None = field(init=False, default=None)
    moe_patch_report: Any | None = field(init=False, default=None)

    def __post_init__(self):
        self.attention_backend = resolve_attention_backend(self.family, self.attention_backend)

    def load(self):
        install_runtime_compat()
        model_path = self.model_path or DEFAULT_MODEL_PATHS[self.family]
        dtype_name = self.dtype or ("bfloat16" if self.family == "llada" else "float16")
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.family,
            model_path=model_path,
            dtype=dtype_name,
            attn_implementation=self.attn_implementation,
        )
        patch_model(
            self.model,
            model_name=self.family,
            alpha=self.alpha,
            attention_backend=self.attention_backend,
            moe_expert_patch=self.moe_expert_patch,
            exact_rope=self.exact_rope,
        )
        self.moe_patch_report = getattr(self.model, "_focus_optimized_moe_report", None)
        return self.model, self.tokenizer

    def generate(self, inputs: Any, **kwargs):
        if self.model is None or self.tokenizer is None:
            self.load()
        assert self.model is not None
        return focus_optimized_generate(
            self.model,
            family=self.family,
            inputs=_input_tensor(inputs).to(self.model.device),
            alpha=self.alpha,
            attention_backend=self.attention_backend,
            **kwargs,
        )


__all__ = [
    "DEFAULT_MODEL_PATHS",
    "FocusOptimizedRuntime",
    "focus_optimized_generate",
    "patch_model",
    "set_seed",
]

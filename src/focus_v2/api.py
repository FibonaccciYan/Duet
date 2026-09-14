"""Public API for the FOCUS v2 runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
import types
from typing import Any

import torch

from src.focus.api import resolve_model_family
from src.losa.generation import load_model_and_tokenizer, set_seed

from .compat import install_runtime_compat
from .generation import focus_v2_generate

try:  # pragma: no cover - the packed MoE backend is optional on CPU-only hosts
    from src.losa.moe_patch import patch_moe_experts
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


def _focus_v2_generate(self, *args, **kwargs):
    inputs = kwargs.pop("inputs", args[0] if args else None)
    if inputs is None:
        raise ValueError("FOCUS v2 generation requires `inputs`")
    return focus_v2_generate(
        self,
        family=self._focus_v2_patch_family,
        inputs=inputs,
        alpha=self._focus_v2_alpha,
        **kwargs,
    ).tokens


def patch_model(
    model: Any,
    model_name: str = "auto",
    *,
    alpha: float = 1.5,
    moe_expert_patch: bool = True,
    **_: Any,
):
    """Install FOCUS v2 through the same model-patching contract as other modes.

    ``moe_expert_patch`` defaults to true because the speed comparison should
    use the same packed routed-MoE backend as dense and LoSA.  For non-MoE
    models the patch is a no-op.
    """

    family = resolve_model_family(model, model_name)
    if alpha < 1.0:
        raise ValueError("FOCUS v2 alpha must be at least 1")

    if not hasattr(model, "_focus_v2_original_generate"):
        model._focus_v2_original_generate = model.generate
        model.generate = types.MethodType(_focus_v2_generate, model)

    model._focus_v2_patch_family = family
    model._focus_v2_alpha = float(alpha)
    model.config.focus_v2_config = {
        "alpha": float(alpha),
        "family": family,
        "moe_expert_patch_requested": bool(moe_expert_patch),
    }

    report = None
    if moe_expert_patch:
        if patch_moe_experts is None:
            raise RuntimeError("the optional packed MoE backend is unavailable")
        report = patch_moe_experts(model, family=family)
    model._focus_v2_moe_report = report
    return model


@dataclass
class FocusV2Runtime:
    family: str
    model_path: str | None = None
    dtype: str | None = None
    attn_implementation: str = "sdpa"
    alpha: float = 1.5
    moe_expert_patch: bool = True
    model: Any | None = field(init=False, default=None)
    tokenizer: Any | None = field(init=False, default=None)
    moe_patch_report: Any | None = field(init=False, default=None)

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
            moe_expert_patch=self.moe_expert_patch,
        )
        self.moe_patch_report = getattr(self.model, "_focus_v2_moe_report", None)
        return self.model, self.tokenizer

    def generate(self, inputs: Any, **kwargs):
        if self.model is None or self.tokenizer is None:
            self.load()
        assert self.model is not None
        return focus_v2_generate(
            self.model,
            family=self.family,
            inputs=_input_tensor(inputs).to(self.model.device),
            alpha=self.alpha,
            **kwargs,
        )


__all__ = [
    "DEFAULT_MODEL_PATHS",
    "FocusV2Runtime",
    "focus_v2_generate",
    "patch_model",
    "set_seed",
]

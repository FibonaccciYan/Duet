"""Public API for the integrated Hugging Face FOCUS runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
import types
from typing import Any

import torch

from src.losa.generation import load_model_and_tokenizer, set_seed
from src.sparse.api import resolve_model_family

from .generation import focus_generate


DEFAULT_MODEL_PATHS = {
    "llada": "/data0/ysy/models/LLaDA2.1-mini",
    "sdar": "/data0/ysy/models/SDAR-8B-Chat-b32",
}


def _focus_generate(self, *args, **kwargs):
    inputs = kwargs.pop("inputs", args[0] if args else None)
    if inputs is None:
        raise ValueError("FOCUS generation requires `inputs`")
    return focus_generate(
        self,
        family=self._focus_patch_family,
        inputs=inputs,
        alpha=self._focus_alpha,
        **kwargs,
    ).tokens


def patch_model(
    model: Any,
    model_name: str = "auto",
    *,
    alpha: float = 1.5,
    **_: Any,
):
    """Install FOCUS through the same model-patching contract as sparse."""

    family = resolve_model_family(model, model_name)
    if alpha < 1.0:
        raise ValueError("FOCUS alpha must be at least 1")
    if not hasattr(model, "_focus_original_generate"):
        model._focus_original_generate = model.generate
        model.generate = types.MethodType(_focus_generate, model)
    model._focus_patch_family = family
    model._focus_alpha = float(alpha)
    model.config.focus_config = {"alpha": float(alpha), "family": family}
    return model


@dataclass
class FocusRuntime:
    family: str
    model_path: str | None = None
    dtype: str | None = None
    attn_implementation: str = "sdpa"
    alpha: float = 1.5
    model: Any | None = field(init=False, default=None)
    tokenizer: Any | None = field(init=False, default=None)

    def load(self):
        model_path = self.model_path or DEFAULT_MODEL_PATHS[self.family]
        dtype = self.dtype or ("bfloat16" if self.family == "llada" else "float16")
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.family,
            model_path=model_path,
            dtype=dtype,
            attn_implementation=self.attn_implementation,
        )
        patch_model(self.model, model_name=self.family, alpha=self.alpha)
        return self.model, self.tokenizer

    def generate(self, inputs: torch.Tensor, **kwargs):
        if self.model is None or self.tokenizer is None:
            self.load()
        assert self.model is not None
        return focus_generate(
            self.model,
            family=self.family,
            inputs=inputs,
            alpha=self.alpha,
            **kwargs,
        )


__all__ = [
    "DEFAULT_MODEL_PATHS",
    "FocusRuntime",
    "load_model_and_tokenizer",
    "patch_model",
    "set_seed",
]

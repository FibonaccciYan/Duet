"""Compatibility shims for model code across transformers releases."""

from __future__ import annotations

from typing import Any, TypedDict

import torch

try:  # transformers >= 4.54
    from transformers.utils import TransformersKwargs
except ImportError:  # transformers <= 4.53
    class TransformersKwargs(TypedDict, total=False):
        pass


def create_bidirectional_mask(
    config: Any,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    **kwargs: Any,
) -> torch.Tensor | None:
    """Fallback for the masking API added after transformers 4.57.1."""
    del config, kwargs
    if attention_mask is None:
        return None
    if attention_mask.dim() == 4:
        return attention_mask
    if bool(attention_mask.all()):
        return None
    mask = attention_mask[:, None, None, :].to(dtype=inputs_embeds.dtype)
    return torch.zeros_like(mask).masked_fill(mask == 0, float("-inf"))


def resolve_rope_scaling(config: Any) -> dict[str, Any]:
    """Return a dictionary even for configs written before RoPE standardization."""
    rope_scaling = getattr(config, "rope_scaling", None)
    if not isinstance(rope_scaling, dict):
        rope_scaling = {
            "rope_type": "default",
            "rope_theta": getattr(config, "rope_theta", 10000.0),
            "partial_rotary_factor": getattr(config, "partial_rotary_factor", 1.0),
        }
        setattr(config, "rope_scaling", rope_scaling)
    return rope_scaling


__all__ = ["TransformersKwargs", "create_bidirectional_mask", "resolve_rope_scaling"]

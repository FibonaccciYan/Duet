"""Compatibility shims for checkpoint remote code on newer transformers."""

from __future__ import annotations

from typing import TypedDict


def install_rope_default_compat() -> bool:
    """Restore the historical ``default`` RoPE initializer.

    LLaDA2.1 remote code looks up this name.  Newer transformers renamed the
    equivalent implementation to ``proportional``.
    """

    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    except Exception:
        return False
    if "default" in ROPE_INIT_FUNCTIONS:
        return False

    import torch

    def _compute_default_rope_parameters(config, device=None, seq_len=None, **_):
        # This mirrors the historical HF default initializer used by LLaDA2.1
        # remote code.  Do not use the newer proportional initializer here: it
        # pads non-rotary dimensions with zero frequencies.
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = getattr(config, "hidden_size", 0) // getattr(
                config, "num_attention_heads", 1
            )
        rotary_dim = int(
            head_dim
            * float(getattr(config, "partial_rotary_factor", 1.0))
        )
        base = float(getattr(config, "rope_theta", 10000.0))
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, rotary_dim, 2, dtype=torch.int64, device=device).float()
                / rotary_dim
            )
        )
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters
    return True


def install_dynamic_cache_compat() -> bool:
    """Add the removed ``DynamicCache.to_legacy_cache`` tuple representation."""

    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        return False

    if hasattr(DynamicCache, "to_legacy_cache"):
        return False

    def to_legacy_cache(self):
        layers = getattr(self, "layers", ())
        return tuple((layer.keys, layer.values) for layer in layers)

    DynamicCache.to_legacy_cache = to_legacy_cache

    def _unused():  # pragma: no cover
        return None

    if not hasattr(DynamicCache, "from_legacy_cache"):
        @classmethod
        def from_legacy_cache(cls, legacy_cache):
            cache = cls()
            for layer_idx, (key_states, value_states) in enumerate(legacy_cache or ()):
                cache.update(key_states, value_states, layer_idx)
            return cache

        DynamicCache.from_legacy_cache = from_legacy_cache
    return True


def install_sdar_transformers_compat() -> bool:
    """Provide a type-only kwargs symbol missing from newer transformers."""

    try:
        import transformers.utils
    except Exception:
        return False
    if hasattr(transformers.utils, "LossKwargs"):
        return False

    class LossKwargs(TypedDict, total=False):
        pass

    transformers.utils.LossKwargs = LossKwargs
    return True


def install_runtime_compat() -> tuple[bool, bool, bool]:
    rope = install_rope_default_compat()
    sdar = install_sdar_transformers_compat()
    cache = install_dynamic_cache_compat()
    return rope, sdar, cache

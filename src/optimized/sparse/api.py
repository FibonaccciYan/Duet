"""Frozen optimized Sparse: unchanged selectors/generation, versioned execution kernels."""
import inspect
import types
import torch
from src.reference.sparse.api import patch_model as _original_patch
from src.reference.sparse.sparse_ops import _compact_prefix_cache
from src.kernels.optimized.small_moe import install_small_moe
from .prefix import prefix_indices
from src.kernels.optimized.function_binding import _bind_globals
from src.kernels.optimized.prefix_cache_view import install as install_cache


def patch_model(model, model_name="auto", exact_rope=False, **kwargs):
    result = _original_patch(model, model_name=model_name, **kwargs)
    model._sparse_optimized_small_moe_count = install_small_moe(model)
    if getattr(model.config, "model_type", None) == "sdar":
        from src.kernels.optimized.sdar_norm import install
        model._versioned_direct_norm_count = install(model)
    model._sparse_original_generate = model.generate
    compactor, _ = _bind_globals(_compact_prefix_cache, {"_prefix_indices": prefix_indices})
    driver, namespace = _bind_globals(
        inspect.unwrap(model.generate).__func__ if hasattr(inspect.unwrap(model.generate), "__func__")
        else inspect.unwrap(model.generate),
        {"_compact_prefix_cache": compactor})
    model.generate = types.MethodType(torch.inference_mode()(driver), model)
    model._sparse_optimized_generation_globals = install_cache(model)
    model._sparse_optimized_generate = model.generate
    if exact_rope:
        from src.kernels.optimized.rope_runtime import install as install_rope
        model._versioned_rope_patch_count = install_rope(model)
    return result

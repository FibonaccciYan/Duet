"""Per-model cache-container construction without immutable-prefix copies."""
import inspect
import types
import torch
from transformers.cache_utils import DynamicCache, DynamicLayer
from .function_binding import _bind_globals

class PrefixViewCache(DynamicCache):
    @classmethod
    def from_legacy_cache(cls, past_key_values):
        cache = cls()
        for key, value in (() if past_key_values is None else past_key_values):
            layer = DynamicLayer()
            layer.lazy_initialization(key)
            layer.keys, layer.values = key, value
            cache.layers.append(layer)
        return cache

def install(model):
    original = model.generate
    driver = inspect.unwrap(original)
    if hasattr(driver, "__func__"):
        driver = driver.__func__
    overrides = {"DynamicCache": PrefixViewCache}
    for name in ("_cached_forward", "_sparse_cached_forward"):
        function = driver.__globals__.get(name)
        if function is not None:
            overrides[name], _ = _bind_globals(function, {"DynamicCache": PrefixViewCache})
    cloned, namespace = _bind_globals(driver, overrides)
    model.generate = types.MethodType(torch.inference_mode()(cloned), model)
    return namespace

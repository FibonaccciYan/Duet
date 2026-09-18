"""Immutable prefix plus replaceable current-block KV for FOCUS.

Every layer update replaces the whole visible current-block suffix. Omitted
FOCUS rows are not retained; this differs intentionally from BlockDualCache.
"""
import torch
from transformers.cache_utils import DynamicCache


class BlockScratchCache(DynamicCache):
    def __init__(self, prefix_cache, block_capacity):
        super().__init__()
        self.prefix_cache = prefix_cache
        self.block_capacity = block_capacity
        self.storage = {}
        self.prefix_lengths = [k.shape[-2] for k, _ in prefix_cache]

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        count = key_states.shape[-2]
        if count > self.block_capacity:
            raise ValueError("current-block KV exceeds scratch capacity")
        prefix = self.prefix_lengths[layer_idx] if layer_idx < len(self.prefix_lengths) else 0
        if layer_idx not in self.storage:
            shape = (*key_states.shape[:-2], prefix+self.block_capacity, key_states.shape[-1])
            k, v = key_states.new_empty(shape), value_states.new_empty(shape)
            if prefix:
                pk,pv = self.prefix_cache[layer_idx]
                k[..., :prefix, :].copy_(pk)
                v[..., :prefix, :].copy_(pv)
            self.storage[layer_idx] = k,v
            # Initialize Transformers metadata using short, initialized tensors.
            super().update(key_states, value_states, layer_idx, cache_kwargs)
        k,v = self.storage[layer_idx]
        k[..., prefix:prefix+count, :].copy_(key_states)
        v[..., prefix:prefix+count, :].copy_(value_states)
        self.layers[layer_idx].keys = k[..., :prefix+count, :]
        self.layers[layer_idx].values = v[..., :prefix+count, :]
        return self.layers[layer_idx].keys, self.layers[layer_idx].values

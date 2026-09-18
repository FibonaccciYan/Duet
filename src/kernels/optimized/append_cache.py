"""Append-only storage for sequential prefill; no denoising cache reuse."""
import torch
from transformers.cache_utils import DynamicCache


class AppendCache(DynamicCache):
    def __init__(self, initial_capacity=256):
        super().__init__()
        self.initial_capacity = initial_capacity
        self.storage = {}
        self.lengths = {}

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if torch.is_grad_enabled() and (key_states.requires_grad or value_states.requires_grad):
            raise ValueError("AppendCache is inference-only")
        length = self.lengths.get(layer_idx, 0)
        end = length + key_states.shape[-2]
        buffers = self.storage.get(layer_idx)
        if buffers is None or buffers[0].shape[-2] < end:
            capacity = max(self.initial_capacity, 1 << (end - 1).bit_length())
            shape = (*key_states.shape[:-2], capacity, key_states.shape[-1])
            new_key, new_value = key_states.new_empty(shape), value_states.new_empty(shape)
            if buffers is not None:
                new_key[..., :length, :].copy_(buffers[0][..., :length, :])
                new_value[..., :length, :].copy_(buffers[1][..., :length, :])
            buffers = new_key, new_value
            self.storage[layer_idx] = buffers
        buffers[0][..., length:end, :].copy_(key_states)
        buffers[1][..., length:end, :].copy_(value_states)
        if layer_idx >= len(self.layers) or not self.layers[layer_idx].is_initialized:
            # Initialize Transformers' layer metadata once, on the short block.
            super().update(key_states, value_states, layer_idx, cache_kwargs)
        layer = self.layers[layer_idx]
        layer.keys = buffers[0][..., :end, :]
        layer.values = buffers[1][..., :end, :]
        self.lengths[layer_idx] = end
        return layer.keys, layer.values

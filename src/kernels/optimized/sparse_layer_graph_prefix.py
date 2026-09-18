"""Experimental immutable-prefix staging for caller-stream SDAR graphs.

Restricted to the inference drivers whose cache prefix is immutable within a
block. Holding source tensor references prevents allocator address reuse from
being mistaken for unchanged cache identity. Not suitable for external in-place
prefix mutation. Graph outputs and the mutable block tail remain private.
"""
import torch
from .sparse_layer_graph_caller import SparseLayerGraph as _Base


class SparseLayerGraph(_Base):
    def load(self, hidden, kwargs):
        cache = kwargs["past_key_value"]
        key, value = self.source(cache)
        prefix = cache.prefix_lengths[self.index] if self.sparse else key.shape[-2]
        if (hidden.shape != self.hidden.shape or prefix != self.prefix
                or key.shape != self.key.shape or value.shape != self.value.shape
                or (self.sparse and cache.positions.shape != self.positions.shape)):
            raise ValueError("graph shape/cache layout changed")
        self.hidden.copy_(hidden)
        same = (getattr(self, "_source_key", None) is key
                and getattr(self, "_source_value", None) is value)
        if not same:
            self.key.copy_(key)
            self.value.copy_(value)
            self._source_key, self._source_value = key, value
        elif self.sparse:
            self.key[..., prefix:, :].copy_(key[..., prefix:, :])
            self.value[..., prefix:, :].copy_(value[..., prefix:, :])
        if self.sparse:
            self.positions.copy_(cache.positions)
        self.kwargs["position_ids"].copy_(kwargs["position_ids"])
        for dest, source in zip(self.kwargs["position_embeddings"],
                                kwargs["position_embeddings"]):
            dest.copy_(source)
        for name in ("attention_mask", "cache_position"):
            if isinstance(kwargs.get(name), torch.Tensor):
                self.kwargs[name].copy_(kwargs[name])

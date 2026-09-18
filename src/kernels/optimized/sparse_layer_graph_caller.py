"""Experimental caller-stream replay; not approved until eager/state audits pass."""
import time
import types
import torch
from transformers.cache_utils import DynamicCache, DynamicLayer
from src.reference.sparse.sparse_ops import _BlockDualCache
from .mlp_graph import copy_tree


class SparseLayerGraph:
    def __init__(self, layer, hidden, kwargs):
        self.layer = layer
        self.function = getattr(layer, "_sparse_graph_original", layer.forward)
        self.index = layer.self_attn.layer_idx
        cache = kwargs["past_key_value"]
        self.sparse = isinstance(cache, _BlockDualCache)
        if not self.sparse and not isinstance(cache, DynamicCache):
            raise ValueError("unsupported cache type")
        if kwargs.get("output_attentions") or bool(kwargs.get("store_kv")) != self.sparse:
            raise ValueError("unsupported layer output/cache options")
        source_key,source_value=self.source(cache)
        self.prefix = cache.prefix_lengths[self.index] if self.sparse else source_key.shape[-2]
        self.key = source_key.clone()
        self.value = source_value.clone()
        self.positions = cache.positions.clone() if self.sparse else None
        if self.sparse:
            self.cache = _BlockDualCache(
                [(self.key, self.value)]*(self.index+1), [self.prefix]*(self.index+1))
            self.cache.set_positions(self.positions)
        else:
            self.cache = DynamicCache()
            self.cache.layers=[DynamicLayer() for _ in range(self.index+1)]
            target=self.cache.layers[self.index]
            target.lazy_initialization(self.key)
            target.keys,target.values=self.key,self.value
        self.hidden = torch.empty_like(hidden)
        self.kwargs = dict(kwargs)
        self.kwargs["past_key_value"] = self.cache
        self.kwargs["position_ids"] = kwargs["position_ids"].clone()
        self.kwargs["position_embeddings"] = tuple(x.clone() for x in kwargs["position_embeddings"])
        for name in ("attention_mask","cache_position"):
            if isinstance(kwargs.get(name),torch.Tensor):
                self.kwargs[name]=kwargs[name].clone()
        self.graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        self.stream = stream
        stream.wait_stream(torch.cuda.current_stream())
        start = time.perf_counter()
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.load(hidden, kwargs)
                self.function(self.hidden, **self.kwargs)
            self.load(hidden, kwargs)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        with torch.cuda.graph(self.graph, stream=stream):
            self.output = self.function(self.hidden, **self.kwargs)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.capture_seconds = time.perf_counter()-start

    def source(self,cache):
        return ((cache.key_cache[self.index],cache.value_cache[self.index])
                if self.sparse else cache[self.index])

    def load(self, hidden, kwargs):
        cache=kwargs["past_key_value"]
        key,value=self.source(cache)
        prefix=cache.prefix_lengths[self.index] if self.sparse else key.shape[-2]
        if (hidden.shape!=self.hidden.shape or prefix!=self.prefix
                or key.shape!=self.key.shape
                or (self.sparse and cache.positions.shape!=self.positions.shape)):
            raise ValueError("graph shape/cache layout changed")
        self.hidden.copy_(hidden)
        self.key.copy_(key)
        self.value.copy_(value)
        if self.sparse:
            self.positions.copy_(cache.positions)
        self.kwargs["position_ids"].copy_(kwargs["position_ids"])
        for dest,source in zip(self.kwargs["position_embeddings"],kwargs["position_embeddings"]):
            dest.copy_(source)
        for name in ("attention_mask","cache_position"):
            if isinstance(kwargs.get(name),torch.Tensor):
                self.kwargs[name].copy_(kwargs[name])

    def __call__(self, hidden, **kwargs):
        # Copies, graph replay and writeback use the caller stream in order.
        # Captures have private graph pools and constructor synchronization.
        source = kwargs["past_key_value"]
        source_key, source_value = self.source(source)
        self.load(hidden, kwargs)
        self.graph.replay()
        if self.sparse:
            source_key[..., self.prefix:, :].copy_(self.key[..., self.prefix:, :])
            source_value[..., self.prefix:, :].copy_(self.value[..., self.prefix:, :])
        return copy_tree(self.output)


def _forward(self, hidden_states, **kwargs):
    cache=kwargs.get("past_key_value")
    sparse=isinstance(cache,_BlockDualCache) and kwargs.get("store_kv") and cache.positions is not None
    dense=isinstance(cache,DynamicCache) and not kwargs.get("store_kv") and len(cache)>self.self_attn.layer_idx
    eligible=(not self.training and not torch.is_grad_enabled()
              and (sparse or dense) and not kwargs.get("output_attentions")
              and hidden_states.is_cuda and hidden_states.shape[0]==1
              and hidden_states.shape[1]<=32)
    if not eligible or torch.cuda.is_current_stream_capturing():
        return self._sparse_graph_original(hidden_states,**kwargs)
    index=self.self_attn.layer_idx
    source_key=cache.key_cache[index] if sparse else cache[index][0]
    mask=kwargs.get("attention_mask")
    key=(bool(sparse),hidden_states.device,hidden_states.dtype,tuple(hidden_states.shape),
         tuple(source_key.shape),cache.prefix_lengths[index] if sparse else source_key.shape[-2],
         None if mask is None else (tuple(mask.shape),mask.dtype))
    graphs=self._sparse_graphs
    if key not in graphs:
        if len(graphs)>=self._sparse_graph_limit:
            return self._sparse_graph_original(hidden_states,**kwargs)
        graphs[key]=SparseLayerGraph(self,hidden_states,kwargs)
    self._sparse_graph_calls+=1
    if getattr(self,"_sparse_graph_verify",False):
        source_key,source_value=graphs[key].source(cache)
        before_key,before_value=source_key.clone(),source_value.clone()
        expected=self._sparse_graph_original(hidden_states,**kwargs)
        expected_key,expected_value=source_key.clone(),source_value.clone()
        source_key.copy_(before_key);source_value.copy_(before_value)
        actual=graphs[key](hidden_states,**kwargs)
        torch.testing.assert_close(actual[0],expected[0],rtol=0,atol=0,
                                   msg=f"layer {index} replay hidden differs")
        torch.testing.assert_close(source_key,expected_key,rtol=0,atol=0,
                                   msg=f"layer {index} replay K differs")
        torch.testing.assert_close(source_value,expected_value,rtol=0,atol=0,
                                   msg=f"layer {index} replay V differs")
        return actual
    return graphs[key](hidden_states,**kwargs)


def install(model, max_graphs_per_layer=16, verify=False, sync_probe=True, sync_mode="both"):
    if model.config.model_type!="sdar" or getattr(model.config,"sdar_losa",False):
        raise ValueError("experimental graphs require SDAR Sparse without embedded LoSA")
    for layer in model.model.layers:
        if not hasattr(layer,"_sparse_graph_original"):
            layer._sparse_graph_original=layer.forward
            layer._sparse_graphs={}
            layer._sparse_graph_calls=0
            layer.forward=types.MethodType(_forward,layer)
        layer._sparse_graph_limit=max_graphs_per_layer
        layer._sparse_graph_verify=verify
        layer._sparse_graph_sync_probe=sync_probe
        layer._sparse_graph_sync_mode=sync_mode


def stats(model):
    graphs=[graph for layer in model.model.layers
            for graph in getattr(layer,"_sparse_graphs",{}).values()]
    return dict(graph_count=len(graphs),capture_seconds=sum(g.capture_seconds for g in graphs),
                calls=sum(getattr(layer,"_sparse_graph_calls",0) for layer in model.model.layers))

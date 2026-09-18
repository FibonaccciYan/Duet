"""Opt-in exact-shape FFN graph replay with dynamic GPU routing.

Routing runs inside the graph on every call. Outputs are copied before returning
so the next replay cannot overwrite a tensor retained by another caller.
"""
import time
import types
import torch


def copy_tree(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(copy_tree(x) for x in value)
    if isinstance(value, list):
        return [copy_tree(x) for x in value]
    if value is None:
        return None
    raise TypeError(f"unsupported graph output: {type(value)}")


class ExactShapeGraph:
    def __init__(self, function, example):
        self.function = function
        self.input = torch.empty_like(example, memory_format=torch.contiguous_format)
        self.input.copy_(example)
        self.graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=example.device)
        stream.wait_stream(torch.cuda.current_stream(example.device))
        start = time.perf_counter()
        with torch.cuda.stream(stream):
            for _ in range(2):
                function(self.input)
        torch.cuda.current_stream(example.device).wait_stream(stream)
        torch.cuda.synchronize(example.device)
        with torch.cuda.graph(self.graph, stream=stream):
            self.output = function(self.input)
        torch.cuda.current_stream(example.device).wait_stream(stream)
        torch.cuda.synchronize(example.device)
        self.capture_seconds = time.perf_counter() - start
        self.calls = 0

    def __call__(self, value):
        if value.shape != self.input.shape or value.dtype != self.input.dtype or value.device != self.input.device:
            raise ValueError("exact graph shape/dtype/device changed")
        self.input.copy_(value)
        self.graph.replay()
        self.calls += 1
        return copy_tree(self.output)


def _forward(self, hidden_states, *args, **kwargs):
    original = self._exact_graph_original
    eligible = (
        self._exact_graph_enabled and not self.training and not args and not kwargs
        and hidden_states.is_cuda and hidden_states.ndim == 3
        and hidden_states.shape[0] == 1
        and hidden_states.shape[1] in self._exact_graph_rows
        and not torch.is_grad_enabled()
    )
    if not eligible or torch.cuda.is_current_stream_capturing():
        return original(hidden_states, *args, **kwargs)
    key = (hidden_states.device, hidden_states.dtype, tuple(hidden_states.shape))
    graphs = self._exact_graph_cache
    if key not in graphs:
        graphs[key] = ExactShapeGraph(original, hidden_states)
    return graphs[key](hidden_states)


def install(model, rows=(32,)):
    """Call only after installing the graph-safe small MoE backend."""
    patched = 0
    for layer in model.model.layers:
        module = layer.mlp
        if hasattr(module, "moe_infer") and not hasattr(module, "_versioned_moe_original"):
            raise ValueError("routed MoE requires small_small_moe before graph capture")
        if not hasattr(module, "_exact_graph_original"):
            module._exact_graph_original = module.forward
            module._exact_graph_cache = {}
            module.forward = types.MethodType(_forward, module)
        module._exact_graph_rows = tuple(rows)
        module._exact_graph_enabled = True
        patched += 1
    return patched


def enable(model, enabled):
    for layer in model.model.layers:
        if hasattr(layer.mlp, "_exact_graph_enabled"):
            layer.mlp._exact_graph_enabled = enabled


def stats(model):
    graphs = [graph for layer in model.model.layers
              for graph in getattr(layer.mlp, "_exact_graph_cache", {}).values()]
    return dict(graphs=len(graphs), calls=sum(g.calls for g in graphs),
                capture_seconds=sum(g.capture_seconds for g in graphs))

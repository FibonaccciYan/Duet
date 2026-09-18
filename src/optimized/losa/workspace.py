"""Per-layer/block scratch buffers; selectors retain PyTorch ordering and ties.

Buffers must not escape their owning layer state or be used concurrently.
Graph capture owns separate state/workspaces. No global workspace cache.
"""
import torch


class Workspace:
    def __init__(self):
        self.buffers = {}
        self.allocations = 0

    def get(self, name, shape, reference, dtype=None):
        dtype = reference.dtype if dtype is None else dtype
        key = (name, tuple(shape), reference.device, dtype)
        value = self.buffers.get(key)
        if value is None:
            value = torch.empty(shape, device=reference.device, dtype=dtype)
            self.buffers[key] = value
            self.allocations += 1
        return value

    def topk(self, x, k, name, sorted=True):
        shape = (*x.shape[:-1], k)
        values = self.get(name + "_values", shape, x)
        indices = self.get(name + "_indices", shape, x, torch.int64)
        torch.topk(x, k, dim=-1, sorted=sorted, out=(values, indices))
        return indices

    def active(self, locality, count):
        indices = self.topk(locality, count, "active", sorted=False)
        result = self.get("active_sorted", (count,), indices)
        order = self.get("active_order", (count,), indices)
        torch.sort(indices, out=(result, order))
        return result

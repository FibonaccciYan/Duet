"""Inference-only wrapper around the existing FlashAttention RMSNorm kernel.

No new reduction implementation, precision change, or epsilon change.
"""
import types
import torch
from .sdar_norm_launch import norm


def inference_forward(self, hidden_states):
    if self.training or torch.is_grad_enabled() or not hidden_states.is_cuda:
        return self._versioned_norm_original(hidden_states)
    return norm(hidden_states, self.weight, self.variance_epsilon)


def install(model):
    count = 0
    for module in model.modules():
        if module.__class__.__name__ != "SDARRMSNorm":
            continue
        if not hasattr(module, "_versioned_norm_original"):
            module._versioned_norm_original = module.forward
            module.forward = types.MethodType(inference_forward, module)
        count += 1
    return count

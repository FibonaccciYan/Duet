"""Dense-only efficient SDPA, without modifying shared Sparse functions."""
import functools
import types
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


class _DenseFunctional:
    def __getattr__(self, name):
        return getattr(F, name)

    def scaled_dot_product_attention(self, q, k, v, *args, **kwargs):
        if kwargs.pop("enable_gqa", False):
            # Efficient SDPA needs matching head counts in this torch version.
            groups = q.shape[-3] // k.shape[-3]
            k = k.repeat_interleave(groups, dim=-3)
            v = v.repeat_interleave(groups, dim=-3)
        return F.scaled_dot_product_attention(q, k, v, *args, **kwargs)


def install(model, family):
    for layer in model.model.layers:
        attention = getattr(layer, "self_attn" if family == "sdar" else "attention")
        if hasattr(attention, "_dense_efficient_original"):
            continue
        original = attention.forward
        dense_forward = original
        if family == "sdar":
            function = original.__func__
            namespace = {**function.__globals__, "F": _DenseFunctional()}
            clone = types.FunctionType(function.__code__, namespace, function.__name__,
                                       function.__defaults__, function.__closure__)
            clone.__kwdefaults__ = function.__kwdefaults__
            dense_forward = types.MethodType(clone, attention)

        def make_forward(original, dense_forward):
            @functools.wraps(original)
            def forward(*args, **kwargs):
                config = model.config
                # Repatching the same model to Sparse must restore its original
                # execution behavior without requiring any Sparse source edits.
                sparse = any(getattr(config, family + "_" + name, False)
                             for name in ("query_sparse", "prefix_sparse", "losa"))
                if sparse:
                    return original(*args, **kwargs)
                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    return dense_forward(*args, **kwargs)
            return forward

        attention._dense_efficient_original = original
        attention.forward = make_forward(original, dense_forward)
    model.config.dense_sdpa_backend = "efficient_attention"

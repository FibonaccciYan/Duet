"""Reuse the packed SDAR QKV projection already used by the dense backend."""
import torch
import torch.nn.functional as F


def prepare(model):
    if getattr(model.config, "model_type", None) != "sdar":
        return 0
    count = 0
    for layer in model.model.layers:
        attention = layer.self_attn
        projections = (attention.q_proj, attention.k_proj, attention.v_proj)
        if any(p.bias is not None for p in projections):
            continue
        if not hasattr(attention, "_sdar_qkv_weight"):
            attention.register_buffer(
                "_sdar_qkv_weight",
                torch.cat(tuple(p.weight.detach() for p in projections), dim=0),
                persistent=False,
            )
        count += 1
    return count


def project(attention, hidden):
    batch, length, _ = hidden.shape
    sizes = (attention.q_proj.out_features, attention.k_proj.out_features,
             attention.v_proj.out_features)
    q, k, v = F.linear(hidden, attention._sdar_qkv_weight).split(sizes, dim=-1)
    q = attention.q_norm(q.view(batch, length, -1, attention.head_dim)).transpose(1, 2)
    k = attention.k_norm(k.view(batch, length, -1, attention.head_dim)).transpose(1, 2)
    v = v.view(batch, length, -1, attention.head_dim).transpose(1, 2)
    return q, k, v

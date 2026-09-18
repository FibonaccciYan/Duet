"""Portable Hugging Face model path for FOCUS row eviction."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache

from .algorithm import attention_importance, select_retained_positions


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q = torch.cat((q_rot * cos + _rotate_half(q_rot) * sin, q_pass), dim=-1)
    k = torch.cat((k_rot * cos + _rotate_half(k_rot) * sin, k_pass), dim=-1)
    return q, k


def _project_qkv(attention, hidden_states, position_embeddings, family: str):
    batch, length, _ = hidden_states.shape
    if family == "llada":
        qkv = attention.query_key_value(hidden_states).view(
            batch,
            length,
            attention.num_heads + 2 * attention.num_key_value_heads,
            attention.head_dim,
        )
        q, k, v = qkv.split(
            [attention.num_heads, attention.num_key_value_heads, attention.num_key_value_heads],
            dim=-2,
        )
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if attention.config.use_qk_norm:
            q, k = attention.query_layernorm(q), attention.key_layernorm(k)
    else:
        shape = (batch, length, -1, attention.head_dim)
        q = attention.q_norm(attention.q_proj(hidden_states).view(shape)).transpose(1, 2)
        k = attention.k_norm(attention.k_proj(hidden_states).view(shape)).transpose(1, 2)
        v = attention.v_proj(hidden_states).view(shape).transpose(1, 2)
    q, k = _apply_rotary(q, k, *position_embeddings)
    return q, k, v


def _dense_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float):
    groups = q.shape[1] // k.shape[1]
    if groups != 1:
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        is_causal=False,
        scale=float(scale),
    )


def _attention_mask(family: str, query_length: int, key_length: int, reference: torch.Tensor):
    if family == "llada":
        return reference.new_zeros((1, 1, query_length, key_length))
    return torch.ones((1, query_length, key_length), dtype=torch.bool, device=reference.device)


def focus_forward(
    model,
    *,
    family: str,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    prefix_cache=(),
    mask_id: int,
    alpha: float,
    average_decoded_tokens: float,
    block_progress: int,
):
    """Run one denoising forward and evict mask rows after layer one."""

    base = model.model
    if family == "llada":
        hidden_states = base.word_embeddings(input_ids)
        attention_attr = "attention"
    else:
        hidden_states = base.embed_tokens(input_ids)
        attention_attr = "self_attn"
    position_embeddings = base.rotary_emb(hidden_states, position_ids)
    cache = DynamicCache.from_legacy_cache(prefix_cache) if prefix_cache else DynamicCache()
    prefix_length = prefix_cache[0][0].shape[-2] if prefix_cache else 0
    full_positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    mask_positions = torch.where(input_ids[0] == int(mask_id))[0]
    selected_positions = full_positions
    importance = []

    for layer_idx, layer in enumerate(base.layers):
        if layer_idx <= 1:
            layer_hidden = hidden_states
            layer_positions = position_ids
            layer_position_embeddings = position_embeddings
        else:
            layer_hidden = hidden_states
            layer_positions = position_ids.index_select(1, selected_positions)
            layer_position_embeddings = (
                position_embeddings[0].index_select(1, selected_positions),
                position_embeddings[1].index_select(1, selected_positions),
            )

        if layer_idx <= 1 and mask_positions.numel():
            attention = getattr(layer, attention_attr)
            normalized = layer.input_layernorm(layer_hidden)
            q, k, v = _project_qkv(attention, normalized, layer_position_embeddings, family)
            importance.append(
                attention_importance(
                    q,
                    k,
                    mask_positions,
                    scale=getattr(attention, "scaling", None),
                )
            )

        mask = _attention_mask(
            family,
            layer_hidden.shape[1],
            prefix_length + layer_hidden.shape[1],
            layer_hidden,
        )
        kwargs = dict(
            attention_mask=mask,
            position_ids=layer_positions,
            past_key_value=cache,
            output_attentions=False,
            use_cache=True,
            position_embeddings=layer_position_embeddings,
        )
        if family == "llada":
            kwargs["output_router_logits"] = False
        else:
            kwargs["store_kv"] = True
        hidden_states = layer(layer_hidden, **kwargs)[0]

        if layer_idx == 1 and len(base.layers) > 2 and mask_positions.numel():
            selected_positions = select_retained_positions(
                input_ids[0],
                mask_id,
                importance[0],
                importance[1],
                alpha=alpha,
                average_decoded_tokens=average_decoded_tokens,
                block_progress=block_progress,
            )
            # Layer one must run on the complete block.  Eviction applies to
            # its output and therefore only affects layers two and later.
            hidden_states = hidden_states.index_select(1, selected_positions)

    hidden_states = base.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    if family == "llada":
        logits = logits.float()
    return SimpleNamespace(logits=logits, positions=selected_positions, cache=cache)


__all__ = ["focus_forward"]

"""Single-projection model path for FOCUS v2.

The old portable path projects Q/K/V once to measure FOCUS importance and then
lets the decoder layer project them again.  This module performs one QKV
projection for each executed layer and reuses the post-RoPE Q/K tensors for
importance.  Except for removing this duplicate projection, the layer and cache
semantics match :mod:`src.focus.model`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache

from .algorithm import attention_importance, select_retained_positions


@dataclass
class FocusV2Result:
    logits: torch.Tensor
    positions: torch.Tensor
    cache: DynamicCache
    trace: list[dict]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary embeddings in the HF remote-code layout used by both models."""

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
            q = attention.query_layernorm(q)
            k = attention.key_layernorm(k)
    else:
        shape = (batch, length, -1, attention.head_dim)
        q = attention.q_norm(attention.q_proj(hidden_states).view(shape)).transpose(1, 2)
        k = attention.k_norm(attention.k_proj(hidden_states).view(shape)).transpose(1, 2)
        v = attention.v_proj(hidden_states).view(shape).transpose(1, 2)

    q, k = _apply_rotary(q, k, *position_embeddings)
    return q, k, v


def _attention_mask(family: str, query_length: int, key_length: int, reference: torch.Tensor):
    if family == "llada":
        return reference.new_zeros((1, 1, query_length, key_length))
    return torch.ones((1, query_length, key_length), dtype=torch.bool, device=reference.device)


def _attention_once(
    attention,
    normalized_hidden: torch.Tensor,
    *,
    family: str,
    position_embeddings,
    attention_mask: torch.Tensor | None,
    past_key_value: DynamicCache | None,
    store_kv: bool,
):
    """Project Q/K/V once, cache K/V, run attention, and return post-RoPE Q/K."""

    q, k, v = _project_qkv(attention, normalized_hidden, position_embeddings, family)
    if past_key_value is not None and store_kv:
        cache_kwargs = {"sin": position_embeddings[1], "cos": position_embeddings[0]}
        if family == "llada":
            k, v = past_key_value.update(k, v, attention.layer_idx, cache_kwargs)
        else:
            k, v = past_key_value.update(k, v, attention.layer_idx)

    # Repeat GQA keys/values explicitly: the portable reference used this
    # representation, and the CUDA SDPA GQA path is less portable across the
    # torch builds on n33.
    groups = q.shape[1] // k.shape[1]
    if groups != 1:
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    attn_output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=float(attention.scaling),
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*normalized_hidden.shape[:-1], -1)
    output_projection = attention.dense if family == "llada" else attention.o_proj
    attn_output = output_projection(attn_output)
    return q, k, attn_output


def _compact_rows(hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Keep only FOCUS-retained rows after layer one."""

    return hidden_states.index_select(1, positions.contiguous())


def _finish_layer(layer, normalized_hidden: torch.Tensor, attention_output: torch.Tensor, family: str):
    """Attention residual followed by the decoder layer's post-attention MLP."""

    hidden_states = normalized_hidden + attention_output
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    if isinstance(hidden_states, tuple):
        # LLaDA sparse-MoE blocks return (states, router_logits).
        hidden_states = hidden_states[0]
    return residual + hidden_states.to(residual.device)


def focus_v2_forward(
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
) -> FocusV2Result:
    """Run one denoising forward with FOCUS eviction after layer one."""

    if input_ids.shape[0] != 1:
        raise ValueError("FOCUS v2 currently supports batch_size=1")
    if alpha < 1.0:
        raise ValueError("FOCUS v2 alpha must be at least 1")

    base = model.model
    attention_attr = "attention" if family == "llada" else "self_attn"
    if family == "llada":
        hidden_states = base.word_embeddings(input_ids)
    else:
        hidden_states = base.embed_tokens(input_ids)

    position_embeddings = base.rotary_emb(hidden_states, position_ids)
    cache = DynamicCache.from_legacy_cache(prefix_cache) if prefix_cache else DynamicCache()
    prefix_length = prefix_cache[0][0].shape[-2] if prefix_cache else 0
    full_positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    mask_positions = torch.where(input_ids[0] == int(mask_id))[0]
    selected_positions = full_positions
    importance: list[torch.Tensor] = []
    trace: list[dict] = []

    for layer_idx, layer in enumerate(base.layers):
        is_importance_layer = layer_idx <= 1 and bool(mask_positions.numel())
        # Selection changes only the output of layer one; layers zero and one
        # therefore always see the complete current block.
        positions = full_positions if layer_idx <= 1 else selected_positions
        layer_hidden = hidden_states
        layer_positions = (
            position_ids if layer_idx <= 1 else position_ids.index_select(1, positions)
        )
        layer_position_embeddings = (
            position_embeddings
            if layer_idx <= 1
            else (
                position_embeddings[0].index_select(1, positions),
                position_embeddings[1].index_select(1, positions),
            )
        )
        attention = getattr(layer, attention_attr)
        normalized = layer.input_layernorm(layer_hidden)
        mask = _attention_mask(
            family,
            layer_hidden.shape[1],
            prefix_length + layer_hidden.shape[1],
            layer_hidden,
        )
        q, k, attention_output = _attention_once(
            attention,
            normalized,
            family=family,
            position_embeddings=layer_position_embeddings,
            attention_mask=mask,
            past_key_value=cache,
            store_kv=True,
        )

        if is_importance_layer:
            importance.append(
                attention_importance(
                    q,
                    k,
                    mask_positions,
                    scale=getattr(attention, "scaling", None),
                )
            )

        hidden_states = _finish_layer(layer, layer_hidden, attention_output, family)

        if layer_idx == 1 and len(base.layers) > 2 and bool(mask_positions.numel()):
            selected_positions = select_retained_positions(
                input_ids[0],
                mask_id,
                importance[0],
                importance[1],
                alpha=alpha,
                average_decoded_tokens=average_decoded_tokens,
                block_progress=block_progress,
            )
            # As in the reference implementation, eviction applies to layer
            # one's output and therefore only affects layers two and later.
            hidden_states = _compact_rows(hidden_states, selected_positions)
            trace.append(
                {
                    "stage": "selection",
                    "mask_positions": int(mask_positions.numel()),
                    "retained": int(selected_positions.numel()),
                    "evicted": int(input_ids.shape[1] - selected_positions.numel()),
                }
            )

    hidden_states = base.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    if family == "llada":
        logits = logits.float()

    # The old reference records no selection event when every position remains.
    if not trace and bool(mask_positions.numel()) and len(base.layers) > 2:
        trace.append(
            {
                "stage": "selection",
                "mask_positions": int(mask_positions.numel()),
                "retained": int(selected_positions.numel()),
                "evicted": 0,
            }
        )

    return FocusV2Result(
        logits=logits,
        positions=selected_positions,
        cache=cache,
        trace=trace,
    )


__all__ = ["FocusV2Result", "focus_v2_forward"]

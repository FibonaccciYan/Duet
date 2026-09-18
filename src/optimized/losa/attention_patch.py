"""optimized LoSA attention patch.

This module patches only the loaded model object in memory and delegates the
LoSA semantics to the operators vendored in :mod:`src.reference.losa.operators`.
"""

from __future__ import annotations

import types

import torch


from .attention_ops import losa_optimized_attention_step


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rotary_local(query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = query[..., :rotary_dim], query[..., rotary_dim:]
    k_rot, k_pass = key[..., :rotary_dim], key[..., rotary_dim:]
    q_rot = q_rot * cos + _rotate_half(q_rot) * sin
    k_rot = k_rot * cos + _rotate_half(k_rot) * sin
    return torch.cat((q_rot, q_pass), dim=-1), torch.cat((k_rot, k_pass), dim=-1)


def _apply_rotary_from_attention(attention, query, key, cos, sin):
    if getattr(attention, "_versioned_rope_enabled", False):
        from src.kernels.optimized.rope_runtime import eligible
        if eligible(query,key,cos,sin):
            from src.kernels.optimized.rope_exact import apply
            return apply(query,key,cos,sin)
    original = getattr(attention, "_paper_losa_dense_forward", None)
    fn = None if original is None else original.__globals__.get("apply_rotary_pos_emb")
    if fn is not None:
        return fn(query, key, cos, sin)
    return _apply_rotary_local(query, key, cos, sin)


def _state_key(attention) -> tuple[str, int]:
    return (attention.__class__.__name__, int(attention.layer_idx))


def _tensor_to_losa_layout(x: torch.Tensor, *, contiguous: bool = True) -> torch.Tensor:
    """[1, H, T, D] -> [T, H, D]."""

    if x.shape[0] != 1:
        raise ValueError("paper LoSA runtime currently supports batch_size=1")
    out = x[0].transpose(0, 1)
    return out.contiguous() if contiguous else out


def _tensor_from_losa_layout(x: torch.Tensor) -> torch.Tensor:
    """[T, H, D] -> [1, H, T, D]."""

    return x.transpose(0, 1).unsqueeze(0).contiguous()


def _run_losa_attention(
    *,
    attention,
    model,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    output_projection,
):
    context = model._paper_losa_context
    prefix_length = int(context["prefix_length"])
    if prefix_length <= 0:
        return None
    query_length = query_states.shape[-2]
    if query_length <= 0:
        return None
    key_total = key_states.shape[-2]
    if key_total < prefix_length + query_length:
        raise ValueError(
            f"KV length {key_total} is smaller than prefix_length + query_length "
            f"({prefix_length} + {query_length})"
        )

    key = _state_key(attention)
    previous_state = context["states"].get(key)

    q = _tensor_to_losa_layout(query_states)
    k_prefix = _tensor_to_losa_layout(key_states[:, :, :prefix_length, :], contiguous=False)
    v_prefix = _tensor_to_losa_layout(value_states[:, :, :prefix_length, :], contiguous=False)
    k_block = _tensor_to_losa_layout(key_states[:, :, prefix_length : prefix_length + query_length, :])
    v_block = _tensor_to_losa_layout(value_states[:, :, prefix_length : prefix_length + query_length, :])

    result = losa_optimized_attention_step(
        q,
        k_prefix,
        v_prefix,
        k_block,
        v_block,
        page_size=int(context["page_size"]),
        token_budget=int(context["token_budget"]),
        active_count=int(context["active_count"]),
        mode=context["gqa_mode"],
        backend=context["backend"],
        previous_state=previous_state,
        fused_state=bool(context.get("fused_state", True)),
        shared_kv=bool(getattr(model, "_losa_optimized_shared_kv", False)),
    )
    context["states"][key] = result.next_state
    # The prefix is immutable for the lifetime of this block's states.
    # Keep the first concatenation and overwrite only the current-block tail.
    result.next_state.combined_key = key_states
    result.next_state.combined_value = value_states
    trace_item = {
        "layer": int(attention.layer_idx),
        "initialized_dense": bool(result.initialized_dense),
        "compact_page_table": True,
    }
    if context.get("trace_detail", False):
        trace_item["active_rows"] = [int(i) for i in result.active_rows.detach().cpu().tolist()]
    context.setdefault("trace", []).append(trace_item)
    output = _tensor_from_losa_layout(result.output.to(query_states.dtype))
    output = output.transpose(1, 2).reshape(query_states.shape[0], query_length, -1).contiguous()
    return output_projection(output)


def _append_current_block(attention, model, cache, key, value, cache_kwargs=None):
    context = model._paper_losa_context
    state = context["states"].get(_state_key(attention))
    prefix_length = int(context["prefix_length"])
    combined_key = getattr(state, "combined_key", None)
    combined_value = getattr(state, "combined_value", None)
    expected = prefix_length + key.shape[-2]
    if combined_key is not None and combined_key.shape[-2] == expected:
        combined_key[:, :, prefix_length:].copy_(key)
        combined_value[:, :, prefix_length:].copy_(value)
        # Match DynamicLayer.update's visible cache result without recopying
        # the prefix. The generation driver creates a fresh cache view per step.
        if hasattr(cache, "layers"):
            cache.layers[attention.layer_idx].keys = combined_key
            cache.layers[attention.layer_idx].values = combined_value
        else:
            cache.key_cache[attention.layer_idx] = combined_key
            cache.value_cache[attention.layer_idx] = combined_value
        return combined_key, combined_value
    return cache.update(key, value, attention.layer_idx, cache_kwargs)


def _llada_attention_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    position_embeddings=None,
    **kwargs,
):
    model = self._paper_losa_model
    if getattr(model, "_paper_losa_context", None) is None:
        return self._paper_losa_dense_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    bsz, q_len, _ = hidden_states.shape
    qkv = self.query_key_value(hidden_states).view(
        bsz,
        q_len,
        self.num_heads + 2 * self.num_key_value_heads,
        self.head_dim,
    )
    query_states, key_states, value_states = qkv.split(
        [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
    )
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    if self.config.use_qk_norm:
        query_states = self.query_layernorm(query_states)
        key_states = self.key_layernorm(key_states)
    cos, sin = position_embeddings
    query_states, key_states = _apply_rotary_from_attention(self, query_states, key_states, cos, sin)
    if past_key_value is not None:
        key_states, value_states = _append_current_block(
            self, model, past_key_value, key_states, value_states,
            {"sin": sin, "cos": cos}
        )

    output = _run_losa_attention(
        attention=self,
        model=model,
        query_states=query_states,
        key_states=key_states,
        value_states=value_states,
        output_projection=self.dense,
    )
    if output is None:
        return self._paper_losa_dense_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
    return output, None, past_key_value


def _sdar_attention_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask,
    past_key_value=None,
    cache_position=None,
    **kwargs,
):
    model = self._paper_losa_model
    if getattr(model, "_paper_losa_context", None) is None:
        return self._paper_losa_dense_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )

    bsz, q_len, _ = hidden_states.shape
    hidden_shape = (bsz, q_len, -1, self.head_dim)
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = _apply_rotary_from_attention(self, query_states, key_states, cos, sin)
    if past_key_value is not None and kwargs.get("store_kv", False):
        key_states, value_states = _append_current_block(
            self, model, past_key_value, key_states, value_states)
    elif past_key_value is not None and not kwargs.get("store_kv", False) and len(past_key_value) > self.layer_idx:
        # This cache is a disposable per-step view; the immutable prefix tuple
        # remains untouched. Use the same cache result as the store-KV path.
        key_states, value_states = _append_current_block(
            self, model, past_key_value, key_states, value_states)

    output = _run_losa_attention(
        attention=self,
        model=model,
        query_states=query_states,
        key_states=key_states,
        value_states=value_states,
        output_projection=self.o_proj,
    )
    if output is None:
        return self._paper_losa_dense_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )
    return output, None


def install_losa_optimized_attention(model, family: str) -> None:
    """Install paper-LoSA attention forwards on a loaded model object."""

    if family == "llada":
        layers = model.model.layers
        attr = "attention"
        forward = _llada_attention_forward
    elif family == "sdar":
        layers = model.model.layers
        attr = "self_attn"
        forward = _sdar_attention_forward
    else:
        raise ValueError(f"unsupported family: {family}")

    for layer in layers:
        attention = getattr(layer, attr)
        if not hasattr(attention, "_paper_losa_dense_forward"):
            attention._paper_losa_dense_forward = attention.forward
            attention._paper_losa_model = model
            attention.forward = types.MethodType(forward, attention)

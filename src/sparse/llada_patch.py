import math
import types
import weakref
from types import SimpleNamespace

import torch
from transformers.cache_utils import DynamicCache

from .sparse_ops import (
    _BlockDualCache,
    _dual_cache_from_dense,
    _prefix_from_dynamic_cache,
    _sample_with_confidence,
    _apply_rotary,
    _attention_output_lse,
    _block_attention_output_lse,
    _compact_prefix_cache,
    _losa_active_indices,
    _losa_key_energy,
    _merge_attention_states,
    _new_losa_state,
    _queue_losa_active_update,
)
from .triton_kernels import _triton_moe_infer, block_causal_prefill


def _prefill_attention_forward(
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
    """Exact block-causal attention without materializing the quadratic mask."""
    model = self._llada_prefill_model_ref()
    if not getattr(model, "_llada_triton_prefill", False):
        return self._llada_prefill_dense_forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
    batch_size, query_length, _ = hidden_states.shape
    qkv = self.query_key_value(hidden_states).view(
        batch_size,
        query_length,
        self.num_heads + 2 * self.num_key_value_heads,
        self.head_dim,
    )
    query, key, value = qkv.split(
        [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
    )
    query, key, value = (
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
    )
    if self.config.use_qk_norm:
        query, key = self.query_layernorm(query), self.key_layernorm(key)
    cos, sin = position_embeddings
    query, key = _apply_rotary(query, cos, sin), _apply_rotary(key, cos, sin)
    if past_key_value is not None:
        key, value = past_key_value.update(
            key, value, self.layer_idx, {"sin": sin, "cos": cos}
        )
    output = block_causal_prefill(query, key, value)
    output = output.transpose(1, 2).reshape(batch_size, query_length, -1)
    return self.dense(output), None, past_key_value


def _dense_block_prefill(model, input_ids, position_ids, cache=None):
    """Run LLaDA's first block pass using implicit block-causal metadata."""
    if not input_ids.is_cuda:
        length = input_ids.shape[1]
        blocks = torch.arange(length, device=input_ids.device) // 32
        allowed = blocks[:, None] >= blocks[None, :]
        dtype = next(model.parameters()).dtype
        mask = torch.zeros((1, 1, length, length), dtype=dtype).masked_fill(
            ~allowed, torch.finfo(dtype).min
        )
        return model._llada_block_cache_dense_forward(
            input_ids,
            attention_mask=mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )
    base = model.model
    hidden_states = base.word_embeddings(input_ids)
    position_embeddings = base.rotary_emb(hidden_states, position_ids)
    cache = DynamicCache() if cache is None else cache
    model._llada_triton_prefill = True
    try:
        for layer in base.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=cache,
                output_attentions=False,
                output_router_logits=False,
                use_cache=True,
                position_embeddings=position_embeddings,
            )[0]
    finally:
        model._llada_triton_prefill = False
    return SimpleNamespace(
        logits=model.lm_head(base.norm(hidden_states)), past_key_values=cache
    )




def _losa_attention_forward(
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
    model = self._llada_losa_model
    context = getattr(model, "_llada_losa_context", None)
    if context is None:
        return self._llada_losa_dense_forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
    input_shape = hidden_states.shape[:-1]
    batch_size, query_length, _ = hidden_states.shape
    qkv = self.query_key_value(hidden_states).view(
        batch_size,
        query_length,
        self.num_heads + 2 * self.num_key_value_heads,
        self.head_dim,
    )
    query, key, value = qkv.split(
        [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
    )
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    if self.config.use_qk_norm:
        query = self.query_layernorm(query)
        key = self.key_layernorm(key)
    cos, sin = position_embeddings
    query = _apply_rotary(query, cos, sin)
    key = _apply_rotary(key, cos, sin)

    state = context["selection_state"]["losa_states"].get(self.layer_idx)
    if state is None:
        # LoSA initializes every layer from dense attention. Besides matching
        # the reference semantics, this avoids introducing a numerical delta
        # before any sparse reuse has occurred.
        dense_outputs = self._llada_losa_dense_forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        cache = dense_outputs[2] if dense_outputs[2] is not None else past_key_value
        if hasattr(cache, "key_cache"):
            key, value = cache.key_cache[self.layer_idx], cache.value_cache[self.layer_idx]
        else:
            key, value = cache[self.layer_idx]
        prefix_length = int(context["prefix_cache_length"])
        prefix_key, prefix_value = key[:, :, :prefix_length], value[:, :, :prefix_length]
        query_positions = context["query_positions"]
        positions = query_positions.to(device=query.device, dtype=torch.long)
        collector = getattr(model, "_llada_query_losa_collector", None)
        if collector is not None:
            collector.record_losa(
                self.layer_idx,
                positions,
                torch.zeros_like(positions, dtype=torch.bool),
                torch.full(
                    (positions.numel(),), torch.nan, device=query.device
                ),
                torch.arange(positions.numel(), device=query.device),
            )
        if prefix_length:
            prefix_mask = attention_mask[..., :prefix_length]
            prefix_output, prefix_lse = _attention_output_lse(
                query,
                prefix_key,
                prefix_value,
                prefix_mask,
                self.num_key_value_groups,
            )
        else:
            prefix_output = query.new_zeros(
                batch_size, self.num_heads, query_length, self.head_dim
            ).float()
            prefix_lse = torch.full(
                (batch_size, self.num_heads, query_length),
                -torch.inf,
                dtype=torch.float32,
                device=query.device,
            )
        context["pending_losa_queries"].append((self.layer_idx, positions, query))
        context["pending_losa"].append(
            (self.layer_idx, positions, prefix_output.float(), prefix_lse)
        )
        return dense_outputs

    cache_kwargs = {"sin": sin, "cos": cos}
    if past_key_value is not None:
        key, value = past_key_value.update(key, value, self.layer_idx, cache_kwargs)

    prefix_length = int(context["prefix_cache_length"])
    prefix_key, block_key = key[:, :, :prefix_length], key[:, :, prefix_length:]
    prefix_value, block_value = value[:, :, :prefix_length], value[:, :, prefix_length:]
    prefix_mask = attention_mask[..., :prefix_length]
    block_mask = attention_mask[..., prefix_length:]
    block_output, block_lse = _block_attention_output_lse(
        query,
        block_key,
        block_value,
        block_mask,
        self.num_key_value_groups,
    )

    query_positions = context["query_positions"]
    score_mode = context.get("score_mode", "query")
    if score_mode.startswith("key_diag") and "key_energy" not in state:
        if prefix_length:
            state["key_energy"] = _losa_key_energy(
                prefix_key,
                self.num_key_value_groups,
                context.get("key_samples", 32),
            )
        else:
            state["key_energy"] = torch.ones(
                self.num_heads, self.head_dim, device=query.device
            )
    collector = getattr(model, "_llada_query_losa_collector", None)
    expose_priority = (
        context.get("unify_query_losa", False)
        and self.layer_idx == context.get("selection_layer")
    )
    active_result = _losa_active_indices(
        state,
        query,
        query_positions,
        context["active_topk"],
        return_metadata=collector is not None or expose_priority,
        score_mode=score_mode,
    )
    if collector is None and not expose_priority:
        active_indices = active_result
    else:
        active_indices, valid, delta = active_result
        if collector is not None:
            collector.record_losa(
                self.layer_idx,
                query_positions,
                valid,
                delta,
                active_indices,
            )
        if expose_priority:
            ranked = active_indices[valid.index_select(0, active_indices)]
            context["query_priority_positions"] = query_positions.index_select(
                0, ranked
            )
    if prefix_length:
        active_query = query.index_select(2, active_indices)
        active_prefix_mask = prefix_mask.index_select(2, active_indices)
        active_prefix_output, active_prefix_lse = _attention_output_lse(
            active_query,
            prefix_key,
            prefix_value,
            active_prefix_mask,
            self.num_key_value_groups,
        )
    else:
        active_prefix_output = query.new_zeros(
            batch_size, self.num_heads, active_indices.numel(), self.head_dim
        ).float()
        active_prefix_lse = torch.full(
            (batch_size, self.num_heads, active_indices.numel()),
            -torch.inf,
            dtype=torch.float32,
            device=query.device,
        )

    positions = query_positions.to(device=query.device, dtype=torch.long)
    prefix_output = state["prefix_output"].index_select(2, positions)
    prefix_lse = state["prefix_lse"].index_select(2, positions)
    if active_indices.numel():
        active_prefix_output = active_prefix_output.float()
        prefix_output.index_copy_(2, active_indices, active_prefix_output)
        prefix_lse.index_copy_(2, active_indices, active_prefix_lse)
        _queue_losa_active_update(
            context,
            self.layer_idx,
            positions,
            query,
            active_indices,
            active_prefix_output,
            active_prefix_lse,
        )
    output, _ = _merge_attention_states(
        prefix_output, prefix_lse, block_output, block_lse
    )
    output = output.to(query.dtype).transpose(1, 2).reshape(*input_shape, -1).contiguous()
    output = self.dense(output)
    return output, None, past_key_value


def _select_positions(
    model,
    hidden_states,
    input_ids,
    mask_id,
    ratio,
    top_k,
    temperature=0.0,
    top_p=None,
    cached_positions=None,
    selection_step=0,
    selection_interval=1,
    query_dense_threshold=0,
    threshold=0.95,
):
    """Apply LLaDA's final confidence selector to shallow Query logits."""
    mask = input_ids[0] == mask_id
    mask_count = int(mask.sum().item())
    collector = getattr(model, "_llada_query_losa_collector", None)
    if ratio >= 1.0 or mask_count <= query_dense_threshold:
        if collector is not None:
            collector.record_query(None, None, "dense")
        return None

    candidate_count = min(max(1, math.ceil(mask_count * ratio)), mask_count)
    decoded = torch.where(~mask)[0]
    if (
        cached_positions is not None
        and selection_interval > 1
        and selection_step % selection_interval != 0
    ):
        old_masks = cached_positions[mask[cached_positions]]
        if old_masks.numel() >= candidate_count:
            selected = torch.cat((decoded, old_masks))
            if collector is not None:
                collector.record_query(None, selected, "reuse")
            return selected

    mask_positions = torch.where(mask)[0]
    mask_logits = model.lm_head(hidden_states[:, mask_positions]).float()
    _, confidence = _sample_with_confidence(
        model,
        mask_logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    high_confidence = confidence[0] > threshold
    if int(high_confidence.sum().item()) >= candidate_count:
        selected_masks = mask_positions[high_confidence]
    else:
        selected_masks = mask_positions[
            torch.topk(confidence[0], k=candidate_count).indices
        ]
    selected = torch.cat((decoded, selected_masks))
    context = getattr(model, "_llada_losa_context", None)
    priority = None if context is None else context.get("query_priority_positions")
    if priority is not None:
        priority = priority[mask.index_select(0, priority)]
        priority = priority[~torch.isin(priority, selected)]
        selected = torch.cat((selected, priority))
    if collector is not None:
        collector.record_query(
            (mask_positions, confidence[0]), selected, "fresh"
        )
    return selected


def _transfer_tokens(
    model,
    block_tokens,
    old_block_tokens,
    prompt_mask,
    active_mask,
    active_logits,
    temperature,
    top_p,
    top_k,
    threshold,
    editing_threshold,
    num_to_transfer,
    logit_positions=None,
):
    x0, x0_p = _sample_with_confidence(
        model,
        active_logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    if logit_positions is None:
        prediction_tokens = x0
        mask_confidence = torch.where(active_mask, x0_p, -torch.inf)
    else:
        prediction_tokens = block_tokens.clone()
        prediction_tokens.index_copy_(1, logit_positions, x0)
        mask_confidence = torch.full_like(active_mask, -torch.inf, dtype=x0_p.dtype)
        mask_confidence.index_copy_(1, logit_positions, x0_p)
    if logit_positions is None:
        mask_candidates = active_mask
    else:
        # Query-sparse logits exist only for selected masks recomputed through
        # the remaining layers, so only those positions may transfer.
        mask_candidates = torch.zeros_like(active_mask)
        mask_candidates.index_fill_(1, logit_positions, True)
        mask_candidates &= active_mask
    high_confidence = (mask_confidence[0] > threshold) & mask_candidates[0]
    mask_transfer = torch.zeros_like(active_mask)
    if int(high_confidence.sum().item()) >= num_to_transfer:
        mask_transfer[0] = high_confidence
    else:
        available = int(mask_candidates.sum().item())
        if available:
            indices = torch.topk(
                mask_confidence[0], k=min(num_to_transfer, available)
            ).indices
            mask_transfer[0, indices] = True

    if logit_positions is None:
        editable = (~active_mask) & (~prompt_mask.unsqueeze(0))
        editing_confidence = torch.where(editable, x0_p, -torch.inf)
        editing = (editing_confidence[0] > editing_threshold) & editable[0]
        editing &= x0[0] != old_block_tokens[0]
        transfer = mask_transfer | editing.unsqueeze(0)
    else:
        transfer = mask_transfer
    if transfer.any():
        block_tokens[transfer] = prediction_tokens[transfer]
    return block_tokens, transfer




def _capture_block_queries(model, block_start):
    captured = [None] * len(model.model.layers)
    handles = []

    for layer_idx, layer in enumerate(model.model.layers):
        attention = layer.attention

        def capture(_module, _inputs, output, idx=layer_idx, attn=attention):
            qkv = output[:, block_start:]
            qkv = qkv.view(
                qkv.shape[0],
                qkv.shape[1],
                attn.num_heads + 2 * attn.num_key_value_heads,
                attn.head_dim,
            )
            query = qkv[:, :, : attn.num_heads].transpose(1, 2)
            if attn.config.use_qk_norm:
                query = attn.query_layernorm(query)
            captured[idx] = query.contiguous()

        handles.append(attention.query_key_value.register_forward_hook(capture))
    return captured, handles




def _layer_attention_mask(
    attention_mask,
    query_positions,
    key_positions,
    prefix_positions,
    original_prefix_length,
    cache=None,
):
    # The cache contains only valid prefix and current-block keys. Every
    # current-block query may attend to every cached key, so the old gather/
    # cat path rebuilt an all-zero mask on every layer.
    cache_key = (query_positions.numel(), prefix_positions.numel())
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    result = attention_mask.new_zeros(
        (
            1,
            1,
            query_positions.numel(),
            prefix_positions.numel() + key_positions.numel(),
        )
    )
    if cache is not None:
        cache[cache_key] = result
    return result


def _cached_forward(
    model,
    input_ids,
    attention_mask,
    position_ids,
    prefix_cache,
    selection_state,
    mask_id,
    ratio,
    top_k,
    selection_interval,
    query_dense_threshold,
    temperature=0.0,
    top_p=None,
    query_sparse=True,
    selection_layer=1,
    threshold=0.95,
    prefix_indices=None,
    original_prefix_length=None,
):
    base = model.model
    inputs_embeds = base.word_embeddings(input_ids)
    position_embeddings = base.rotary_emb(inputs_embeds, position_ids)
    full_cache = DynamicCache.from_legacy_cache(prefix_cache)
    sparse_cache = selection_state.get("sparse_cache")
    hidden_states = inputs_embeds
    selected_positions = None
    selected_position_ids = None
    selected_position_embeddings = None
    full_hidden_base = None
    compressed_hidden_states = False
    zero_attention_masks = {}

    losa_context = None
    if model.config.llada_losa:
        losa_states = selection_state.setdefault("losa_states", {})
        losa_context = {
            "selection_state": selection_state,
            "prefix_cache_length": prefix_cache[0][0].shape[-2] if prefix_cache else 0,
            "block_length": input_ids.shape[1],
            "active_topk": model.config.llada_losa_active_topk,
            "score_mode": model.config.llada_losa_score_mode,
            "key_samples": model.config.llada_losa_key_samples,
            "pending_losa": [],
            "pending_losa_queries": [],
            "query_positions": None,
            "selection_layer": selection_layer,
            "unify_query_losa": model.config.llada_query_losa_union,
            "query_priority_positions": None,
        }
        model._llada_losa_context = losa_context

    compact_prefix_length = prefix_cache[0][0].shape[-2] if prefix_cache else 0
    if original_prefix_length is None:
        original_prefix_length = compact_prefix_length
    all_positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    try:
        for layer_idx, decoder_layer in enumerate(base.layers):
            if selected_positions is None:
                layer_hidden = hidden_states
                layer_position_ids = position_ids
                layer_position_embeddings = position_embeddings
                layer_cache = full_cache
                layer_query_positions = all_positions
            else:
                layer_hidden = (
                    hidden_states
                    if compressed_hidden_states
                    else hidden_states.index_select(1, selected_positions)
                )
                layer_position_ids = selected_position_ids
                layer_position_embeddings = selected_position_embeddings
                if sparse_cache is None:
                    raise RuntimeError("Sparse cache is missing after dense refresh.")
                sparse_cache.set_positions(selected_positions)
                layer_cache = sparse_cache
                layer_query_positions = selected_positions

            if losa_context is not None:
                losa_context["query_positions"] = layer_query_positions

            layer_attention_mask = None
            if not query_sparse or losa_context is not None:
                layer_prefix_positions = (
                    prefix_indices[layer_idx]
                    if prefix_indices is not None
                    else torch.arange(original_prefix_length, device=input_ids.device)
                )
                layer_attention_mask = _layer_attention_mask(
                    attention_mask,
                    layer_query_positions,
                    all_positions,
                    layer_prefix_positions,
                    original_prefix_length,
                    cache=zero_attention_masks,
                )

            layer_outputs = decoder_layer(
                layer_hidden,
                attention_mask=layer_attention_mask,
                position_ids=layer_position_ids,
                past_key_value=layer_cache,
                output_attentions=False,
                output_router_logits=False,
                use_cache=True,
                position_embeddings=layer_position_embeddings,
            )
            hidden_states = layer_outputs[0]
            compressed_hidden_states = selected_positions is not None

            if (
                query_sparse
                and layer_idx == selection_layer
                and selection_layer < len(base.layers) - 1
            ):
                selected_positions = _select_positions(
                    model,
                    hidden_states,
                    input_ids,
                    mask_id=mask_id,
                    ratio=ratio,
                    top_k=top_k,
                    temperature=temperature,
                    top_p=top_p,
                    cached_positions=selection_state.get("positions"),
                    selection_step=selection_state["step"],
                    selection_interval=selection_interval,
                    query_dense_threshold=query_dense_threshold,
                    threshold=threshold,
                )
                selection_state["positions"] = selected_positions
                if selected_positions is not None:
                    full_hidden_base = hidden_states.clone()
                    selected_position_ids = position_ids.index_select(
                        1, selected_positions
                    )
                    selected_position_embeddings = (
                        position_embeddings[0].index_select(1, selected_positions),
                        position_embeddings[1].index_select(1, selected_positions),
                    )

        if losa_context is not None:
            for layer_idx, positions, query in losa_context["pending_losa_queries"]:
                state = selection_state["losa_states"].get(layer_idx)
                if state is None:
                    state = _new_losa_state(query, input_ids.shape[1])
                    selection_state["losa_states"][layer_idx] = state
                state["previous_query"].index_copy_(2, positions, query)
            for layer_idx, positions, prefix_output, prefix_lse in losa_context["pending_losa"]:
                state = selection_state["losa_states"][layer_idx]
                state["prefix_output"].index_copy_(2, positions, prefix_output)
                state["prefix_lse"].index_copy_(2, positions, prefix_lse)
                state["valid"][0, positions] = True
                if positions.numel() == state["valid"].shape[1]:
                    state["fully_valid"] = True
    finally:
        if losa_context is not None:
            model._llada_losa_context = None

    if selected_positions is not None:
        full_hidden_base[:, selected_positions] = hidden_states
        hidden_states = full_hidden_base

    hidden_states = base.norm(hidden_states)
    if selected_positions is None:
        return model.lm_head(hidden_states).float(), selected_positions, None
    selected_mask_positions = selected_positions[
        input_ids[0, selected_positions] == mask_id
    ]
    mask_hidden = hidden_states.index_select(1, selected_mask_positions)
    return (
        model.lm_head(mask_hidden).float(),
        selected_positions,
        selected_mask_positions,
    )


def _trim_to_first_eos(generated: torch.Tensor, eos_id: int) -> torch.Tensor:
    positions = (generated[0] == eos_id).nonzero(as_tuple=True)[0]
    return generated[:, : positions[0].item() + 1] if len(positions) else generated


@torch.inference_mode()
def _block_cache_generate(self, *args, **kwargs):
    inputs = kwargs.pop("inputs", args[0] if args else None)
    if inputs is None:
        raise ValueError("block-cache generation requires `inputs`")
    if inputs.shape[0] != 1:
        raise AssertionError("block-cache sparse generation currently assumes batch_size == 1")

    temperature = float(kwargs.pop("temperature", 0.0))
    block_length = int(kwargs.pop("block_length", 32))
    steps = int(kwargs.pop("steps", 32))
    gen_length = int(kwargs.pop("gen_length", 2048))
    top_p = kwargs.pop("top_p", None)
    top_k = kwargs.pop("top_k", None)
    eos_early_stop = bool(kwargs.pop("eos_early_stop", False))
    minimal_topk = int(kwargs.pop("minimal_topk", 1))
    threshold = float(kwargs.pop("threshold", 0.95))
    editing_threshold = float(kwargs.pop("editing_threshold", 0.9))
    max_post_steps = int(kwargs.pop("max_post_steps", 16))
    eos_id = int(kwargs.pop("eos_id", 156892))
    mask_id = int(kwargs.pop("mask_id", 156895))
    num_to_transfer = int(kwargs.pop("num_to_transfer", 1))
    if kwargs:
        raise TypeError(f"Unsupported block-cache generation arguments: {sorted(kwargs)}")
    if block_length <= 0 or gen_length < 0 or num_to_transfer <= 0:
        raise ValueError("block_length and num_to_transfer must be positive; gen_length must be non-negative")

    steps = min(steps, max(1, gen_length // max(1, minimal_topk)))
    input_ids = inputs.to(self.device)
    prompt_length = input_ids.shape[1]
    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length
    mask_dtype = next(self.parameters()).dtype
    mask_template = torch.empty((), dtype=mask_dtype, device=self.device)
    position_ids = torch.arange(total_length, device=self.device).unsqueeze(0)
    x = torch.full((1, total_length), mask_id, dtype=torch.long, device=self.device)
    x[:, :prompt_length] = input_ids

    ratio = self.config.llada_sparse_dlm_ratio
    selection_interval = self.config.llada_sparse_dlm_selection_interval
    selection_top_k = self.config.llada_sparse_dlm_top_k
    selection_layer = self.config.llada_query_selection_layer
    query_dense_threshold = self.config.llada_query_dense_threshold
    query_sparse = self.config.llada_query_sparse
    prefix_sparse = self.config.llada_prefix_sparse
    prefix_token_budget = self.config.llada_prefix_token_budget
    prefix_chunk_size = self.config.llada_prefix_chunk_size
    prefill_blocks = prompt_length // block_length
    fixed_prefix_length = prefill_blocks * block_length
    prefix_sparse = (
        prefix_sparse
        and fixed_prefix_length >= self.config.llada_prefix_min_prefix_length
    )
    query_sparse = (
        query_sparse
        and fixed_prefix_length >= self.config.llada_query_min_prefix_length
    )
    fixed_prefix_cache = None
    if input_ids.is_cuda and fixed_prefix_length:
        fixed_prefix_cache = DynamicCache()
        prefill_chunk = 8192
        for start in range(0, fixed_prefix_length, prefill_chunk):
            end = min(start + prefill_chunk, fixed_prefix_length)
            _dense_block_prefill(
                self,
                x[:, start:end],
                position_ids[:, start:end],
                fixed_prefix_cache,
            )

    previous_prefix_indices = None
    previous_prefix_length = 0
    for block_idx in range(prefill_blocks, num_blocks):
        block_start = block_idx * block_length
        block_end = min((block_idx + 1) * block_length, total_length)
        current_window_end = block_end
        cur_x = x[:, :current_window_end]
        cur_positions = position_ids[:, :current_window_end]
        old_block_tokens = cur_x[:, -block_length:].clone()
        active_block_mask = cur_x[:, -block_length:] == mask_id
        prompt_mask = torch.zeros(block_length, dtype=torch.bool, device=self.device)
        if block_start < prompt_length:
            prompt_mask[: min(prompt_length - block_start, block_length)] = True

        captured_queries = handles = None
        if prefix_sparse and block_start:
            capture_start = (
                block_start - fixed_prefix_length
                if fixed_prefix_cache is not None
                else block_start
            )
            captured_queries, handles = _capture_block_queries(self, capture_start)
        try:
            if fixed_prefix_cache is None:
                dense_outputs = _dense_block_prefill(self, cur_x, cur_positions)
            else:
                refresh_cache = DynamicCache.from_legacy_cache(
                    fixed_prefix_cache.to_legacy_cache()
                )
                dense_outputs = _dense_block_prefill(
                    self,
                    cur_x[:, fixed_prefix_length:],
                    cur_positions[:, fixed_prefix_length:],
                    refresh_cache,
                )
        finally:
            if handles is not None:
                for handle in handles:
                    handle.remove()
        active_logits = dense_outputs.logits[:, -block_length:, :].float()
        block_tokens = cur_x[:, -block_length:]
        block_tokens, _ = _transfer_tokens(
            self,
            block_tokens,
            old_block_tokens,
            prompt_mask,
            active_block_mask,
            active_logits,
            temperature,
            top_p,
            top_k,
            threshold,
            editing_threshold,
            num_to_transfer,
        )
        cur_x[:, -block_length:] = block_tokens
        x[:, :current_window_end] = cur_x

        prefix_indices = None
        if prefix_sparse and block_start:
            prefix_cache, prefix_indices = _compact_prefix_cache(
                self,
                dense_outputs.past_key_values,
                block_start,
                captured_queries,
                cur_positions[:, block_start:block_end],
                prefix_token_budget,
                prefix_chunk_size,
                previous_prefix_indices,
                previous_prefix_length,
            )
            previous_prefix_indices = prefix_indices
            previous_prefix_length = block_start
        else:
            prefix_cache = _prefix_from_dynamic_cache(
                dense_outputs.past_key_values, block_start
            )
        sparse_cache = (
            _dual_cache_from_dense(
                dense_outputs.past_key_values,
                prefix_cache,
                block_start,
                block_end,
            )
            if query_sparse
            else None
        )
        del dense_outputs
        selection_state = {"positions": None, "step": 0, "sparse_cache": sparse_cache}
        post_steps = 0
        max_iterations = max(steps, block_length) + max_post_steps
        for _ in range(1, max_iterations):
            old_block_tokens = x[:, block_start:block_end].clone()
            active_block_mask = old_block_tokens == mask_id
            if not active_block_mask.any():
                post_steps += 1
                if post_steps > max_post_steps:
                    break
            block_input = x[:, block_start:block_end]
            step_mask = mask_template
            step_positions = position_ids[:, block_start:block_end]
            collector = getattr(self, "_llada_query_losa_collector", None)
            if collector is not None:
                collector.begin_step(
                    block_idx,
                    selection_state["step"],
                    old_block_tokens[0],
                    mask_id,
                )
            logits, selected_positions, logit_positions = _cached_forward(
                self,
                block_input,
                step_mask,
                step_positions,
                prefix_cache,
                selection_state,
                mask_id,
                ratio,
                selection_top_k,
                selection_interval,
                query_dense_threshold,
                temperature=temperature,
                top_p=top_p,
                query_sparse=query_sparse,
                selection_layer=selection_layer,
                threshold=threshold,
                prefix_indices=prefix_indices,
                original_prefix_length=block_start,
            )
            selection_state["step"] += 1
            active_logits = logits
            block_tokens = x[:, block_start:block_end]
            block_tokens, transfer = _transfer_tokens(
                self,
                block_tokens,
                old_block_tokens,
                prompt_mask,
                active_block_mask,
                active_logits,
                temperature,
                top_p,
                top_k,
                threshold,
                editing_threshold,
                num_to_transfer,
                logit_positions=logit_positions,
            )
            if query_sparse and logit_positions is not None:
                changed = block_tokens != old_block_tokens
                allowed = torch.zeros_like(changed)
                allowed.index_fill_(1, logit_positions, True)
                if torch.any(changed & ~allowed):
                    raise RuntimeError(
                        "Query-sparse transfer changed a position without a corresponding logit"
                    )
            if collector is not None:
                collector.end_step(transfer[0])
            x[:, block_start:block_end] = block_tokens
            if not active_block_mask.any() and not transfer.any():
                break

        if torch.any(x[:, block_start:block_end] == mask_id):
            raise RuntimeError(f"Block {block_idx} still contains mask tokens after block-cache steps")
        if eos_early_stop:
            generated_part = x[0, prompt_length:block_end]
            eos_positions = (generated_part == eos_id).nonzero(as_tuple=True)[0]
            if len(eos_positions):
                break

    generated = x[:, prompt_length : prompt_length + gen_length]
    return _trim_to_first_eos(generated, eos_id)


def patch_llada_model(
    model,
    ratio=0.5,
    top_k=64,
    selection_interval=4,
    query_dense_threshold=4,
    query_min_prefix_length=24576,
    selection_layer=1,
    query_sparse=True,
    prefix_sparse=True,
    prefix_min_prefix_length=4096,
    prefix_token_budget=256,
    prefix_chunk_size=1024,
    losa=False,
    losa_active_topk=5,
    losa_score_mode="query",
    losa_key_samples=32,
    query_losa_union=False,
):
    if (
        top_k <= 0
        or selection_interval <= 0
        or prefix_token_budget <= 0
        or prefix_chunk_size <= 0
        or prefix_min_prefix_length < 0
        or query_min_prefix_length < 0
        or losa_active_topk <= 0
        or losa_score_mode not in {"query", "key_diag"}
        or losa_key_samples <= 0
        or (
            query_sparse
            and (
                selection_layer < 0
                or selection_layer >= len(model.model.layers) - 1
            )
        )
    ):
        raise ValueError(
            "positive sparse parameters and 0 <= selection_layer < num_hidden_layers - 1 are required"
        )
    model.config.llada_sparse_dlm_ratio = min(max(float(ratio), 0.0), 1.0)
    model.config.llada_sparse_dlm_top_k = int(top_k)
    model.config.llada_sparse_dlm_selection_interval = max(1, int(selection_interval))
    model.config.llada_query_dense_threshold = int(query_dense_threshold)
    model.config.llada_query_min_prefix_length = int(query_min_prefix_length)
    model.config.llada_query_selection_layer = int(selection_layer)
    model.config.llada_query_sparse = bool(query_sparse)
    model.config.llada_prefix_sparse = bool(prefix_sparse)
    model.config.llada_prefix_min_prefix_length = int(prefix_min_prefix_length)
    model.config.llada_prefix_token_budget = int(prefix_token_budget)
    model.config.llada_prefix_chunk_size = int(prefix_chunk_size)
    model.config.llada_losa = bool(losa)
    model.config.llada_losa_active_topk = int(losa_active_topk)
    model.config.llada_losa_score_mode = losa_score_mode
    model.config.llada_losa_key_samples = int(losa_key_samples)
    model.config.llada_query_losa_union = bool(query_losa_union)
    if not hasattr(model, "_llada_block_cache_dense_forward"):
        model._llada_block_cache_dense_forward = model.forward
    for layer_idx, layer in enumerate(model.model.layers):
        attention = layer.attention
        if not hasattr(attention, "_llada_prefill_dense_forward"):
            attention._llada_prefill_dense_forward = attention.forward
            attention._llada_prefill_model_ref = weakref.ref(model)
            attention.layer_idx = layer_idx
            attention.forward = types.MethodType(
                _prefill_attention_forward, attention
            )
    if losa:
        for layer_idx, layer in enumerate(model.model.layers):
            attention = layer.attention
            if not hasattr(attention, "_llada_losa_dense_forward"):
                attention._llada_losa_dense_forward = attention.forward
                attention._llada_losa_model = model
                attention.layer_idx = layer_idx
                attention.forward = types.MethodType(_losa_attention_forward, attention)
    model.generate = types.MethodType(_block_cache_generate, model)
    return model




def _packed_weights(block):
    experts = list(block.experts)
    if not experts:
        return None
    projections = []
    for name in ("gate_proj", "up_proj", "down_proj"):
        weights = [getattr(expert, name).weight for expert in experts]
        if any(weight is None for weight in weights):
            return None
        projections.append(
            torch.stack([weight.detach() for weight in weights], dim=0).contiguous()
        )
    return tuple(projections)


def _pack_block(block):
    if getattr(block, "_llada_moe_expert_patched", False):
        return True
    if not hasattr(block, "experts") or not hasattr(block, "moe_infer"):
        return False

    weights = _packed_weights(block)
    if weights is None:
        return False
    if any(weight.device.type != "cuda" for weight in weights):
        return False
    gate_weight, up_weight, down_weight = weights
    if gate_weight.ndim != 3 or up_weight.shape != gate_weight.shape:
        return False
    if down_weight.ndim != 3 or down_weight.shape[0] != gate_weight.shape[0]:
        return False

    block.register_buffer("_llada_moe_gate_weight", gate_weight, persistent=False)
    block.register_buffer("_llada_moe_up_weight", up_weight, persistent=False)
    block.register_buffer("_llada_moe_down_weight", down_weight, persistent=False)

    # The packed buffers replace the per-expert parameters in memory. The
    # patched inference method is installed immediately after this point.
    for expert in block.experts:
        expert.gate_proj.weight = None
        expert.up_proj.weight = None
        expert.down_proj.weight = None
    block.moe_infer = types.MethodType(_triton_moe_infer, block)
    block._llada_moe_expert_patched = True
    return True


def patch_moe_experts(model):
    """Patch all compatible routed MoE blocks in ``model`` in-place."""
    patched = 0
    for module in model.modules():
        if _pack_block(module):
            patched += 1
    model._llada_moe_expert_patch_count = patched
    return patched

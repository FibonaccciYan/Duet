"""Runtime block-diffusion generation patch for SDAR checkpoints.

The checkpoint contains the model forward pass but not SDAR's generation
routine. This keeps the official block/KV-cache algorithm plus the optional
late-layer query selector in this repository and leaves the model directory
untouched.
"""

import math
import types
import weakref

import torch
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache

from .sparse_ops import (
    _BlockDualCache,
    _apply_rotary,
    _attention_output_lse,
    _block_attention_output_lse,
    _compact_prefix_cache,
    _dual_cache_from_dense,
    _losa_active_indices,
    _losa_key_energy,
    _merge_attention_states,
    _new_losa_state,
    _queue_losa_active_update,
    _prefix_from_dynamic_cache,
)
from .triton_kernels import block_causal_prefill
# Zero-based decoder layer after which Query Sparse chooses mask candidates.
# Layer 4 is too early for SDAR-b32: its candidate ranking diverges sharply
# from the final-layer transfer positions.
QUERY_SELECTION_LAYER = 5


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
    minimum_mask_candidates=1,
    strategy="low_confidence_static",
    threshold=1.0,
    entropy_budget=None,
    decoded_count=None,
):
    if strategy == "sequential" and decoded_count is not None:
        mask_count = input_ids.shape[1] - decoded_count
        if ratio >= 1.0 or mask_count <= query_dense_threshold:
            return None
        candidate_count = min(
            max(int(minimum_mask_candidates), math.ceil(mask_count * ratio)),
            mask_count,
        )
        if (
            cached_positions is not None
            and selection_interval > 1
            and selection_step % selection_interval != 0
            and cached_positions.numel() - decoded_count >= candidate_count
        ):
            return torch.arange(cached_positions.numel(), device=input_ids.device)
        return torch.arange(
            decoded_count + candidate_count,
            device=input_ids.device,
        )

    mask = input_ids[0] == mask_id
    mask_count = int(mask.sum().item())
    if ratio >= 1.0 or mask_count <= query_dense_threshold:
        return None

    candidate_count = min(
        max(int(minimum_mask_candidates), math.ceil(mask_count * ratio)),
        mask_count,
    )
    decoded = torch.where(~mask)[0]
    if (
        cached_positions is not None
        and selection_interval > 1
        and selection_step % selection_interval != 0
    ):
        old_masks = cached_positions[mask[cached_positions]]
        if old_masks.numel() >= candidate_count:
            return torch.cat((decoded, old_masks))
    mask_positions = torch.where(mask)[0]
    if strategy == "sequential":
        return torch.cat((decoded, mask_positions[:candidate_count]))

    mask_logits = model.lm_head(hidden_states[:, mask_positions, :])
    _, confidence = model._sample_with_temperature_topk_topp(
        mask_logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    entropy = (
        entropy_from_logits(mask_logits, temperature, top_k, top_p)
        if strategy == "entropy_bounded"
        else None
    )
    selected = select_transfer(
        torch.ones_like(confidence, dtype=torch.bool),
        confidence,
        candidate_count,
        strategy,
        threshold,
        entropy=entropy,
        entropy_budget=entropy_budget,
    )
    return torch.cat((decoded, mask_positions[torch.where(selected[0])[0]]))


def _capture_sdar_block_queries(model):
    """Capture unrotated, normalized queries from SDAR's dense first step."""
    captured = [None] * len(model.model.layers)
    handles = []
    for layer_idx, layer in enumerate(model.model.layers):
        attention = layer.self_attn

        def capture(_module, _inputs, output, idx=layer_idx, attn=attention):
            query = output.view(
                output.shape[0],
                output.shape[1],
                attn.num_attention_heads,
                attn.head_dim,
            )
            captured[idx] = attn.q_norm(query).transpose(1, 2).contiguous()

        handles.append(attention.q_proj.register_forward_hook(capture))
    return captured, handles


def _sdar_attention_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask,
    past_key_value=None,
    **kwargs,
):
    """Run block prefill and known-unmasked decode without mask inspection."""
    model = self._sdar_prefill_model_ref()
    prefill = getattr(model, "_sdar_triton_prefill", False)
    decode = getattr(model, "_sdar_decode_attention", False)
    if not prefill and not decode:
        return self._sdar_prefill_dense_forward(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            **kwargs,
        )

    batch_size, query_length, _ = hidden_states.shape
    query = self.q_norm(
        self.q_proj(hidden_states).view(
            batch_size, query_length, self.num_attention_heads, self.head_dim
        )
    ).transpose(1, 2)
    key = self.k_norm(
        self.k_proj(hidden_states).view(
            batch_size, query_length, self.num_key_value_heads, self.head_dim
        )
    ).transpose(1, 2)
    value = self.v_proj(hidden_states).view(
        batch_size, query_length, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    cos, sin = position_embeddings
    query = _apply_rotary(query, cos, sin)
    key = _apply_rotary(key, cos, sin)
    if past_key_value is not None and kwargs.get("store_kv", False):
        key, value = past_key_value.update(key, value, self.layer_idx)
    elif past_key_value is not None and len(past_key_value) > self.layer_idx:
        prefix_key, prefix_value = past_key_value[self.layer_idx]
        key = torch.cat((prefix_key, key), dim=2)
        value = torch.cat((prefix_value, value), dim=2)

    if prefill:
        output = block_causal_prefill(query, key, value)
    else:
        output = F.scaled_dot_product_attention(
            query, key, value, scale=self.scaling, enable_gqa=True
        )
    output = output.transpose(1, 2).reshape(batch_size, query_length, -1)
    return self.o_proj(output.contiguous()), None


def _sdar_losa_attention_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask,
    past_key_value=None,
    **kwargs,
):
    """SDAR attention with LoSA prefix-state reuse during denoising only."""
    model = self._sdar_losa_model_ref()
    context = getattr(model, "_sdar_losa_context", None)
    if context is None:
        return self._sdar_losa_dense_forward(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            **kwargs,
        )

    batch_size, query_length, _ = hidden_states.shape
    query = self.q_norm(
        self.q_proj(hidden_states).view(
            batch_size, query_length, self.num_attention_heads, self.head_dim
        )
    ).transpose(1, 2)
    key = self.k_norm(
        self.k_proj(hidden_states).view(
            batch_size, query_length, self.num_key_value_heads, self.head_dim
        )
    ).transpose(1, 2)
    value = self.v_proj(hidden_states).view(
        batch_size, query_length, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    cos, sin = position_embeddings
    query = _apply_rotary(query, cos, sin)
    key = _apply_rotary(key, cos, sin)
    mask = attention_mask.unsqueeze(1) if attention_mask.ndim == 3 else attention_mask
    prefix_length = int(context["prefix_cache_length"])
    positions = context["query_positions"].to(device=query.device, dtype=torch.long)
    state = context["selection_state"]["losa_states"].get(self.layer_idx)

    if state is None:
        dense_outputs = self._sdar_losa_dense_forward(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            **kwargs,
        )
        if (
            past_key_value is not None
            and hasattr(past_key_value, "key_cache")
            and len(past_key_value.key_cache) > self.layer_idx
        ):
            cached_key = past_key_value.key_cache[self.layer_idx]
            cached_value = past_key_value.value_cache[self.layer_idx]
        elif past_key_value is not None and len(past_key_value) > self.layer_idx:
            cached_key, cached_value = past_key_value[self.layer_idx]
        else:
            cached_key, cached_value = key, value
        if prefix_length:
            prefix_output, prefix_lse = _attention_output_lse(
                query,
                cached_key[:, :, :prefix_length],
                cached_value[:, :, :prefix_length],
                mask[..., :prefix_length],
                self.num_key_value_groups,
            )
        else:
            prefix_output = query.new_zeros(
                batch_size, self.num_attention_heads, query_length, self.head_dim
            ).float()
            prefix_lse = torch.full(
                (batch_size, self.num_attention_heads, query_length),
                -torch.inf,
                dtype=torch.float32,
                device=query.device,
            )
        context["pending_losa_queries"].append((self.layer_idx, positions, query))
        context["pending_losa"].append(
            (self.layer_idx, positions, prefix_output.float(), prefix_lse)
        )
        return dense_outputs

    if past_key_value is not None:
        if kwargs.get("store_kv", False):
            key, value = past_key_value.update(key, value, self.layer_idx)
        elif len(past_key_value) > self.layer_idx:
            prefix_key, prefix_value = past_key_value[self.layer_idx]
            key = torch.cat((prefix_key, key), dim=-2)
            value = torch.cat((prefix_value, value), dim=-2)

    prefix_key, block_key = key[:, :, :prefix_length], key[:, :, prefix_length:]
    prefix_value, block_value = (
        value[:, :, :prefix_length],
        value[:, :, prefix_length:],
    )
    prefix_mask, block_mask = mask[..., :prefix_length], mask[..., prefix_length:]
    block_output, block_lse = _block_attention_output_lse(
        query,
        block_key,
        block_value,
        block_mask,
        self.num_key_value_groups,
    )
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
                self.num_attention_heads, self.head_dim, device=query.device
            )
    active_indices = _losa_active_indices(
        state,
        query,
        positions,
        context["active_topk"],
        score_mode=score_mode,
    )
    if prefix_length:
        active_prefix_output, active_prefix_lse = _attention_output_lse(
            query.index_select(2, active_indices),
            prefix_key,
            prefix_value,
            prefix_mask.index_select(2, active_indices),
            self.num_key_value_groups,
        )
    else:
        active_prefix_output = query.new_zeros(
            batch_size,
            self.num_attention_heads,
            active_indices.numel(),
            self.head_dim,
        ).float()
        active_prefix_lse = torch.full(
            (batch_size, self.num_attention_heads, active_indices.numel()),
            -torch.inf,
            dtype=torch.float32,
            device=query.device,
        )

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
    output = output.to(query.dtype).transpose(1, 2).reshape(
        batch_size, query_length, -1
    )
    return self.o_proj(output.contiguous()), None


def _stop_ids(model, eos_id):
    if eos_id is None:
        eos_id = getattr(model.generation_config, "eos_token_id", None)
    if eos_id is None:
        return ()
    if isinstance(eos_id, int):
        return (eos_id,)
    return tuple(int(token_id) for token_id in eos_id)


def _sparse_cached_forward(
    model,
    input_ids,
    position_ids,
    prefix_cache,
    selection_state,
    mask_id,
    temperature,
    top_k,
    top_p,
    minimum_mask_candidates=1,
    strategy="sequential",
    threshold=0.85,
    entropy_budget=None,
    refresh_late_kv=False,
    deep_only_transfer=False,
    query_sparse=True,
):
    base = model.model
    hidden_states = base.embed_tokens(input_ids)
    position_embeddings = base.rotary_emb(hidden_states, position_ids)
    sparse_cache = selection_state.get("sparse_cache")
    if query_sparse and sparse_cache is None:
        raise RuntimeError("Query Sparse requires an initialized block cache")
    full_cache = (
        None if query_sparse else DynamicCache.from_legacy_cache(prefix_cache)
    )
    all_positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    selected_positions = None
    full_hidden_base = None
    compressed = False
    prefix_lengths = [key.shape[-2] for key, _ in prefix_cache]
    prefix_length = prefix_lengths[0] if prefix_lengths else 0
    attention_masks = {}
    sparse_position_ids = None
    sparse_position_embeddings = None
    selection_layer = min(
        model.config.sdar_query_selection_layer,
        len(base.layers) - 2,
    )
    losa_context = None
    losa_active_topk = model.config.sdar_losa_active_topk
    if (
        prefix_cache
        and model.config.sdar_losa
        and losa_active_topk < input_ids.shape[1]
    ):
        selection_state.setdefault("losa_states", {})
        losa_context = {
            "selection_state": selection_state,
            "prefix_cache_length": 0,
            "active_topk": losa_active_topk,
            "score_mode": model.config.sdar_losa_score_mode,
            "key_samples": model.config.sdar_losa_key_samples,
            "pending_losa": [],
            "pending_losa_queries": [],
            "query_positions": None,
        }
        model._sdar_losa_context = losa_context

    try:
        for layer_idx, decoder_layer in enumerate(base.layers):
            if (
                not query_sparse
                or layer_idx < selection_layer
                or selected_positions is None
                or refresh_late_kv
            ):
                layer_hidden = hidden_states
                layer_positions = position_ids
                layer_position_embeddings = position_embeddings
                if query_sparse:
                    sparse_cache.set_positions(all_positions)
                    layer_cache = sparse_cache
                else:
                    layer_cache = full_cache
                layer_query_positions = all_positions
            else:
                layer_hidden = (
                    hidden_states
                    if compressed
                    else hidden_states.index_select(1, selected_positions)
                )
                layer_positions = sparse_position_ids
                layer_position_embeddings = sparse_position_embeddings
                layer_cache = sparse_cache
                layer_query_positions = selected_positions

            layer_prefix_length = (
                prefix_lengths[layer_idx] if prefix_lengths else 0
            )
            layer_attention = None
            if losa_context is not None:
                mask_key = (layer_hidden.shape[1], layer_prefix_length)
                layer_attention = attention_masks.get(mask_key)
                if layer_attention is None:
                    layer_attention = torch.ones(
                        input_ids.shape[0],
                        layer_hidden.shape[1],
                        layer_prefix_length + input_ids.shape[1],
                        dtype=torch.bool,
                        device=input_ids.device,
                    )
                    attention_masks[mask_key] = layer_attention
            if losa_context is not None:
                losa_context["prefix_cache_length"] = layer_prefix_length
                losa_context["query_positions"] = layer_query_positions

            hidden_states = decoder_layer(
                layer_hidden,
                attention_mask=layer_attention,
                position_ids=layer_positions,
                past_key_value=layer_cache,
                output_attentions=False,
                use_cache=True,
                store_kv=True,
                position_embeddings=layer_position_embeddings,
            )[0]
            compressed = (
                query_sparse
                and selected_positions is not None
                and layer_idx >= selection_layer
                and not refresh_late_kv
            )

            if query_sparse and layer_idx == selection_layer:
                selected_positions = _select_positions(
                    model,
                    base.norm(hidden_states),
                    input_ids,
                    mask_id=mask_id,
                    ratio=model.config.sdar_sparse_dlm_ratio,
                    top_k=top_k,
                    temperature=temperature,
                    top_p=top_p,
                    cached_positions=selection_state.get("positions"),
                    selection_step=selection_state["step"],
                    selection_interval=model.config.sdar_sparse_dlm_selection_interval,
                    query_dense_threshold=(
                        model.config.sdar_query_dense_threshold
                    ),
                    minimum_mask_candidates=minimum_mask_candidates,
                    strategy=strategy,
                    threshold=threshold,
                    entropy_budget=entropy_budget,
                    decoded_count=selection_state.get("sequential_decoded"),
                )
                selection_state["positions"] = selected_positions
                if selected_positions is not None:
                    full_hidden_base = hidden_states.clone()
                    sparse_position_ids = position_ids.index_select(
                        1, selected_positions
                    )
                    sparse_position_embeddings = (
                        position_embeddings[0].index_select(1, selected_positions),
                        position_embeddings[1].index_select(1, selected_positions),
                    )
                    sparse_cache.set_positions(selected_positions)

        if losa_context is not None:
            states = selection_state["losa_states"]
            for layer_idx, positions, query in losa_context["pending_losa_queries"]:
                state = states.get(layer_idx)
                if state is None:
                    state = _new_losa_state(query, input_ids.shape[1])
                    states[layer_idx] = state
                state["previous_query"].index_copy_(2, positions, query)
            for layer_idx, positions, prefix_output, prefix_lse in losa_context[
                "pending_losa"
            ]:
                state = states[layer_idx]
                state["prefix_output"].index_copy_(2, positions, prefix_output)
                state["prefix_lse"].index_copy_(2, positions, prefix_lse)
                state["valid"][0, positions] = True
                if positions.numel() == state["valid"].shape[1]:
                    state["fully_valid"] = True
    finally:
        if losa_context is not None:
            model._sdar_losa_context = None

    if selected_positions is not None:
        selected_hidden = (
            hidden_states.index_select(1, selected_positions)
            if refresh_late_kv
            else hidden_states
        )
        full_hidden_base[:, selected_positions] = selected_hidden
        hidden_states = full_hidden_base
    hidden_states = base.norm(hidden_states)
    if selected_positions is None:
        return model.lm_head(hidden_states), None
    if deep_only_transfer:
        mask_positions = selected_positions[
            input_ids[0].index_select(0, selected_positions) == mask_id
        ]
    else:
        if strategy == "sequential" and "sequential_decoded" in selection_state:
            mask_positions = torch.arange(
                selection_state["sequential_decoded"],
                input_ids.shape[1],
                device=input_ids.device,
            )
        else:
            mask_positions = torch.where(input_ids[0] == mask_id)[0]
    return model.lm_head(hidden_states.index_select(1, mask_positions)), mask_positions


@torch.inference_mode()
def _block_diffusion_generate(self, *args, **kwargs):
    inputs = kwargs.pop("inputs", args[0] if args else None)
    if inputs is None:
        raise ValueError("SDAR generation requires `inputs`")
    if inputs.shape[0] != 1:
        raise ValueError("SDAR runtime patch currently requires batch_size=1")

    gen_length = int(kwargs.pop("gen_length", 128))
    block_length = int(kwargs.pop("block_length", 32))
    steps = int(kwargs.pop("steps", block_length))
    temperature = float(kwargs.pop("temperature", 0.0))
    top_k = kwargs.pop("top_k", 0) or 0
    top_p = kwargs.pop("top_p", 1.0)
    strategy = kwargs.pop("remasking_strategy", "sequential")
    threshold = float(kwargs.pop("threshold", 0.85))
    eb_threshold = kwargs.pop("eb_threshold", None)
    mask_id = int(kwargs.pop("mask_id", 151669))
    eos_id = kwargs.pop("eos_id", None)
    eos_early_stop = bool(kwargs.pop("eos_early_stop", True))
    kwargs.pop("attention_mask", None)
    if kwargs:
        raise TypeError(f"Unsupported SDAR generation arguments: {sorted(kwargs)}")
    if gen_length < 0:
        raise ValueError("gen_length must be non-negative")
    if gen_length == 0:
        return inputs[:, :0]
    if strategy == "entropy_bounded" and eb_threshold is None:
        raise ValueError("eb_threshold is required for entropy_bounded transfer")

    input_ids = inputs.to(self.device)
    prompt_length = input_ids.shape[1]
    stop_ids = _stop_ids(self, eos_id)
    query_sparse = self.config.sdar_query_sparse
    prefix_sparse = self.config.sdar_prefix_sparse
    losa = self.config.sdar_losa
    prefix_token_budget = self.config.sdar_prefix_token_budget
    prefix_chunk_size = self.config.sdar_prefix_chunk_size
    losa_active_topk = self.config.sdar_losa_active_topk
    active_losa = losa and losa_active_topk < block_length
    selection_top_k = self.config.sdar_sparse_dlm_top_k
    refresh_step = self.config.sdar_sparse_dlm_refresh_step
    deep_only_transfer = self.config.sdar_sparse_dlm_deep_only_transfer
    selection_state = {}
    previous_prefix_indices = None
    previous_prefix_length = 0

    def sparse_denoise(
        model,
        block_tokens,
        attention_mask,
        position_ids,
        past_key_values,
        block_start,
        block_end,
        step,
        minimum,
    ):
        nonlocal previous_prefix_indices, previous_prefix_length
        if step == 0:
            dense_cache = DynamicCache.from_legacy_cache(
                past_key_values.to_legacy_cache()
            )
            captured_queries = handles = None
            if prefix_sparse and block_start:
                captured_queries, handles = _capture_sdar_block_queries(model)
            try:
                outputs = model(
                    block_tokens,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=dense_cache,
                    use_cache=True,
                    store_kv=True,
                )
            finally:
                if handles is not None:
                    for handle in handles:
                        handle.remove()
            if prefix_sparse and block_start:
                prefix_cache, previous_prefix_indices = _compact_prefix_cache(
                    model,
                    dense_cache,
                    block_start,
                    captured_queries,
                    position_ids,
                    prefix_token_budget,
                    prefix_chunk_size,
                    previous_prefix_indices,
                    previous_prefix_length,
                )
                previous_prefix_length = block_start
            else:
                prefix_cache = _prefix_from_dynamic_cache(dense_cache, block_start)
            selection_state.clear()
            selection_state.update(
                positions=None,
                step=step,
                sequential_decoded=(
                    min(block_tokens.shape[1], max(0, prompt_length - block_start) + minimum)
                    if strategy == "sequential" else None
                ),
                sparse_cache=(
                    _dual_cache_from_dense(
                        dense_cache, prefix_cache, block_start, block_end
                    )
                    if query_sparse
                    else None
                ),
                prefix_cache=prefix_cache,
                losa_states={},
            )
            return outputs.logits, None

        logits, logit_positions = _sparse_cached_forward(
            model,
            block_tokens,
            position_ids,
            selection_state["prefix_cache"],
            selection_state,
            mask_id,
            temperature,
            selection_top_k,
            top_p,
            minimum_mask_candidates=minimum,
            strategy=strategy,
            threshold=threshold,
            entropy_budget=eb_threshold,
            # Large blocks are most sensitive immediately after the first few
            # transfers. selection_state["step"] is the zero-based completed
            # sparse-call index; -2 is the diagnostic every-step upper bound.
            refresh_late_kv=(
                query_sparse
                and block_length >= 16
                and (
                    refresh_step == -2
                    or selection_state["step"] == refresh_step
                )
            ),
            deep_only_transfer=deep_only_transfer,
            query_sparse=query_sparse,
        )
        selection_state["step"] = step
        if strategy == "sequential":
            selection_state["sequential_decoded"] += minimum
        return logits, logit_positions

    self._sdar_decode_attention = True
    try:
        tokens = block_diffusion_generate(
            self,
            prompt={"input_ids": input_ids},
            mask_id=mask_id,
            gen_length=gen_length,
            block_length=block_length,
            denoising_steps=steps,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            remasking_strategy=strategy,
            confidence_threshold=threshold,
            eb_threshold=eb_threshold,
            stopping_criteria_idx=(stop_ids if eos_early_stop else None),
            denoise_fn=(
                sparse_denoise
                if query_sparse or prefix_sparse or active_losa
                else None
            ),
        )
    finally:
        self._sdar_decode_attention = False
    generated = tokens[:, prompt_length : prompt_length + gen_length]
    if eos_early_stop and stop_ids:
        stop_positions = torch.cat(
            [(generated[0] == token_id).nonzero() for token_id in stop_ids]
        )
        if stop_positions.numel():
            generated = generated[:, : int(stop_positions.min()) + 1]
    return generated


def patch_sdar_model(
    model,
    ratio=0.5,
    top_k=64,
    selection_interval=1,
    query_dense_threshold=0,
    refresh_step=-1,
    selection_layer=QUERY_SELECTION_LAYER,
    deep_only_transfer=False,
    query_sparse=True,
    prefix_sparse=False,
    prefix_token_budget=256,
    prefix_chunk_size=1024,
    losa=False,
    losa_active_topk=5,
    losa_score_mode="query",
    losa_key_samples=32,
):
    if getattr(model.config, "model_type", None) != "sdar":
        raise TypeError("SDAR patch requires a model with config.model_type == 'sdar'")
    if (
        top_k <= 0
        or selection_interval <= 0
        or refresh_step < -2
        or selection_layer < 0
        or prefix_token_budget <= 0
        or prefix_chunk_size <= 0
        or losa_active_topk <= 0
        or losa_score_mode not in {"query", "key_diag"}
        or losa_key_samples <= 0
    ):
        raise ValueError(
            "top_k, intervals, prefix budget, chunk size, and LoSA active "
            "top-k must be positive; "
            "refresh_step must be >= -2"
        )
    model.config.sdar_sparse_dlm_ratio = min(max(float(ratio), 0.0), 1.0)
    model.config.sdar_sparse_dlm_top_k = int(top_k)
    model.config.sdar_sparse_dlm_selection_interval = int(selection_interval)
    model.config.sdar_query_dense_threshold = int(
        query_dense_threshold
    )
    model.config.sdar_sparse_dlm_refresh_step = int(refresh_step)
    model.config.sdar_query_selection_layer = int(selection_layer)
    model.config.sdar_sparse_dlm_deep_only_transfer = bool(deep_only_transfer)
    model.config.sdar_query_sparse = bool(query_sparse)
    model.config.sdar_prefix_sparse = bool(prefix_sparse)
    model.config.sdar_prefix_token_budget = int(prefix_token_budget)
    model.config.sdar_prefix_chunk_size = int(prefix_chunk_size)
    model.config.sdar_losa = bool(losa)
    model.config.sdar_losa_active_topk = int(losa_active_topk)
    model.config.sdar_losa_score_mode = losa_score_mode
    model.config.sdar_losa_key_samples = int(losa_key_samples)
    for layer in model.model.layers:
        attention = layer.self_attn
        if not hasattr(attention, "_sdar_prefill_dense_forward"):
            attention._sdar_prefill_dense_forward = attention.forward
            attention._sdar_prefill_model_ref = weakref.ref(model)
            attention.forward = types.MethodType(
                _sdar_attention_forward, attention
            )
    if losa:
        for layer in model.model.layers:
            attention = layer.self_attn
            if not hasattr(attention, "_sdar_losa_dense_forward"):
                attention._sdar_losa_dense_forward = attention.forward
                attention._sdar_losa_model_ref = weakref.ref(model)
                attention.forward = types.MethodType(
                    _sdar_losa_attention_forward, attention
                )
    if not hasattr(model, "_sample_with_temperature_topk_topp"):
        model._sample_with_temperature_topk_topp = types.MethodType(
            lambda _self, logits, temperature=0.0, top_k=0, top_p=1.0: (
                sample_with_temperature_topk_topp(logits, temperature, top_k, top_p)
            ),
            model,
        )
    if not hasattr(model, "_sdar_hf_generate"):
        model._sdar_hf_generate = model.generate
        model.generate = types.MethodType(_block_diffusion_generate, model)
    return model




def top_k_logits(logits, k):
    if k is None or k <= 0:
        return logits
    else:
        values, _ = torch.topk(logits, min(int(k), logits.shape[-1]))
        min_values = values[..., -1, None]
        return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)


def top_p_logits(logits, p):
    if p is None or p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask_indices = torch.scatter(torch.full_like(logits, False, dtype=torch.bool),
                                 -1, sorted_indices, sorted_mask)
    logits = logits.masked_fill(mask_indices, float('-inf'))
    return logits


def sample_with_temperature_topk_topp(logits, temperature=1.0, top_k=0, top_p=1.0):
    orig_shape = logits.shape[:-1]    # [batch, block]
    vocab_size = logits.shape[-1]

    logits = logits.reshape(-1, vocab_size)  # [batch*block, vocab]

    if temperature is None or temperature <= 0:
        token = logits.argmax(dim=-1)
        probs = F.softmax(logits, dim=-1)
        token_prob = torch.gather(probs, -1, token.unsqueeze(-1)).squeeze(-1)
        return token.view(*orig_shape), token_prob.view(*orig_shape)

    if temperature != 1.0:
        logits = logits / temperature
    if top_k is not None and top_k > 0:
        logits = top_k_logits(logits, top_k)
    if top_p is not None and top_p < 1.0:
        logits = top_p_logits(logits, top_p)
    probs = F.softmax(logits, dim=-1)  # shape: [batch*block, vocab]
    assert probs.dim() == 2
    token = torch.multinomial(probs, num_samples=1)  # [batch*block, 1]
    token_prob = torch.gather(probs, -1, token)     # [batch*block, 1]

    return token.view(*orig_shape), token_prob.view(*orig_shape)


def entropy_from_logits(logits, temperature=1.0, top_k=0, top_p=1.0):
    """Return categorical entropy for every token position."""
    logits = logits.float()
    if temperature is not None and temperature > 0:
        logits = logits / temperature
        logits = top_k_logits(logits, top_k)
        logits = top_p_logits(logits, top_p)
    log_probs = F.log_softmax(logits, dim=-1)
    terms = torch.where(
        torch.isfinite(log_probs), log_probs.exp() * log_probs, 0.0
    )
    return -terms.sum(dim=-1)


def get_num_transfer_tokens(block_length, steps):
    if block_length <= 0 or steps <= 0 or steps > block_length:
        raise ValueError("SDAR requires 1 <= denoising_steps <= block_length")
    base = block_length // steps
    remainder = block_length % steps
    num_transfer_tokens = torch.zeros(steps, dtype=torch.int64) + base
    num_transfer_tokens[:remainder] += 1
    return num_transfer_tokens


def select_transfer(
    mask,
    confidence,
    minimum,
    strategy,
    threshold,
    entropy=None,
    entropy_budget=None,
):
    """Apply SDAR's transfer rule to positions with available predictions."""
    if strategy == "entropy_bounded" and (
        entropy is None or entropy_budget is None
    ):
        raise ValueError(
            "entropy and entropy_budget are required for entropy_bounded"
        )
    transfer = torch.zeros_like(mask)
    for batch_idx in range(mask.shape[0]):
        available = torch.isfinite(
            entropy[batch_idx] if strategy == "entropy_bounded" else confidence[batch_idx]
        )
        positions = torch.where(mask[batch_idx] & available)[0]
        count = min(int(minimum), positions.numel())
        if not count:
            continue
        if strategy == "entropy_bounded":
            values, order = torch.sort(entropy[batch_idx, positions])
            budget_count = int(
                torch.searchsorted(
                    torch.cumsum(values, dim=0),
                    values.new_tensor(float(entropy_budget)),
                    right=False,
                ).item()
            )
            count = min(max(count, budget_count, 1), positions.numel())
            selected = positions[order[:count]]
        elif strategy == "sequential":
            selected = positions[:count]
        else:
            scores = confidence[batch_idx, positions]
            if strategy == "low_confidence_dynamic":
                high = positions[scores > threshold]
                selected = high if high.numel() >= count else positions[
                    torch.topk(scores, count).indices
                ]
            elif strategy == "low_confidence_static":
                selected = positions[torch.topk(scores, count).indices]
            else:
                raise ValueError(f"Unknown remasking strategy: {strategy}")
        transfer[batch_idx, selected] = True
    return transfer


@torch.inference_mode()
def block_diffusion_generate(
        model,
        prompt,
        mask_id,
        gen_length=128,
        block_length=32,
        denoising_steps=32,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        remasking_strategy='sequential',
        confidence_threshold=0.85,
        eb_threshold=None,
        stopping_criteria_idx=None,
        denoise_fn=None,
    ):

    model.eval()
    if remasking_strategy == "entropy_bounded" and eb_threshold is None:
        raise ValueError("eb_threshold is required for entropy_bounded transfer")
    input_ids = prompt['input_ids']
    prompt_length = input_ids.shape[1]
    past_key_values = DynamicCache()

    num_blocks = (prompt_length + gen_length +
                  block_length - 1) // block_length
    total_length = num_blocks * block_length

    position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)

    x = torch.full((1, total_length), mask_id,
                   dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length

    # Prefill stage.  The block-causal mask lets aligned prompt chunks be
    # written to the same KV cache independently.  Keeping the query side
    # bounded avoids SDPA materializing an O(prompt_length**2) score tensor.
    if prefill_length > 0:
        prefill_chunk_length = (
            prefill_length if prefill_length <= 256 * block_length
            else 128 * block_length
        )
        model._sdar_triton_prefill = True
        try:
            for chunk_start in range(0, prefill_length, prefill_chunk_length):
                chunk_end = min(chunk_start + prefill_chunk_length, prefill_length)
                cur_x = x[:, chunk_start:chunk_end]
                cur_position_ids = position_ids[:, chunk_start:chunk_end]
                model(cur_x,
                      attention_mask=None,
                      position_ids=cur_position_ids,
                      past_key_values=past_key_values,
                      use_cache=True,
                      store_kv=True)
        finally:
            model._sdar_triton_prefill = False

    num_transfer_tokens = get_num_transfer_tokens(
        block_length, denoising_steps)

    # Decode stage
    for num_block in range(prefill_blocks, num_blocks):
        cur_x = x[:, num_block*block_length:(num_block+1)*block_length].clone()
        cur_attn_mask = torch.ones(
            1,
            block_length,
            (num_block + 1) * block_length,
            dtype=torch.bool,
            device=model.device,
        )
        cur_position_ids = position_ids[:, num_block *
                                        block_length:(num_block+1)*block_length]
        for step in range(denoising_steps + 1):
            mask_index = (cur_x == mask_id)
            if mask_index.sum() == 0:
                # Store kv cache
                model(cur_x,
                      attention_mask=cur_attn_mask,
                      position_ids=cur_position_ids,
                      past_key_values=past_key_values,
                      use_cache=True,
                      store_kv=True)
                break

            if step == denoising_steps:
                raise RuntimeError(
                    f"SDAR block {num_block} still contains masks after "
                    f"{denoising_steps} steps"
                )

            # Denosing
            logit_positions = None
            if denoise_fn is None:
                logits = model(cur_x,
                               attention_mask=cur_attn_mask,
                               position_ids=cur_position_ids,
                               past_key_values=past_key_values,
                               use_cache=True,
                               store_kv=False).logits
            else:
                logits, logit_positions = denoise_fn(
                    model=model,
                    block_tokens=cur_x,
                    attention_mask=cur_attn_mask,
                    position_ids=cur_position_ids,
                    past_key_values=past_key_values,
                    block_start=num_block * block_length,
                    block_end=(num_block + 1) * block_length,
                    step=step,
                    minimum=int(num_transfer_tokens[step]),
                )

            # Sampling
            x0, x0_p = sample_with_temperature_topk_topp(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            x0_entropy = (
                entropy_from_logits(logits, temperature, top_k, top_p)
                if remasking_strategy == "entropy_bounded"
                else None
            )

            if logit_positions is not None:
                full_x0 = cur_x.clone()
                full_x0.index_copy_(1, logit_positions, x0)
                full_confidence = torch.full_like(
                    cur_x, -torch.inf, dtype=x0_p.dtype
                )
                full_confidence.index_copy_(1, logit_positions, x0_p)
                x0, x0_p = full_x0, full_confidence
                if x0_entropy is not None:
                    full_entropy = torch.full_like(
                        cur_x, torch.inf, dtype=x0_entropy.dtype
                    )
                    full_entropy.index_copy_(1, logit_positions, x0_entropy)
                    x0_entropy = full_entropy

            # Sampling strategy
            confidence = torch.where(mask_index, x0_p, -torch.inf)
            transfer_index = select_transfer(
                mask_index,
                confidence,
                num_transfer_tokens[step],
                remasking_strategy,
                confidence_threshold,
                entropy=x0_entropy,
                entropy_budget=eb_threshold,
            )

            cur_x[transfer_index] = x0[transfer_index]

        x[:, num_block*block_length:(num_block+1)*block_length] = cur_x
        if stopping_criteria_idx is not None and any(
            torch.any(x[:, prompt_length:] == stop_idx)
            for stop_idx in stopping_criteria_idx
        ):
            break

    return x

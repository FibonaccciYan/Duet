"""Runtime block-diffusion generation patch for SDAR checkpoints.

The checkpoint contains the model forward pass but not SDAR's generation
routine. This keeps the official block/KV-cache algorithm plus the optional
late-layer query selector in this repository and leaves the model directory
untouched.
"""

import types
import weakref

import torch
from transformers.cache_utils import DynamicCache

from .core import (
    _BlockDualCache,
    _dual_cache_from_dense,
    _legacy_prefix_cache,
    _select_positions,
)
from .block_cache_sparse_dlm_patch import (
    _apply_rotary,
    _attention_output_lse,
    _compact_prefix_cache,
    _losa_active_indices,
    _merge_attention_states,
    _new_losa_state,
)
from .sdar_generate import (
    block_diffusion_generate,
    get_num_transfer_tokens as _transfer_counts,
    sample_with_temperature_topk_topp as _sample_with_confidence,
    select_transfer as _select_transfer,
)


# Zero-based decoder layer after which Query Sparse chooses mask candidates.
# Layer 4 is too early for SDAR-b32: its candidate ranking diverges sharply
# from the final-layer transfer positions.
QUERY_SELECTION_LAYER = 5


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
    block_output, block_lse = _attention_output_lse(
        query, block_key, block_value, block_mask, self.num_key_value_groups
    )
    active_indices = _losa_active_indices(
        state, query, positions, context["active_topk"]
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
        active_positions = positions.index_select(0, active_indices)
        active_prefix_output = active_prefix_output.float()
        prefix_output.index_copy_(2, active_indices, active_prefix_output)
        prefix_lse.index_copy_(2, active_indices, active_prefix_lse)
        context["pending_losa"].append(
            (
                self.layer_idx,
                active_positions,
                active_prefix_output,
                active_prefix_lse,
            )
        )
    context["pending_losa_queries"].append((self.layer_idx, positions, query))
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
    full_cache = DynamicCache.from_legacy_cache(prefix_cache)
    sparse_cache = selection_state.get("sparse_cache")
    selected_positions = None
    full_hidden_base = None
    compressed = False
    prefix_lengths = [key.shape[-2] for key, _ in prefix_cache]
    prefix_length = prefix_lengths[0] if prefix_lengths else 0
    attention_masks = {}
    sparse_position_ids = None
    sparse_position_embeddings = None
    selection_layer = min(
        int(getattr(model.config, "sdar_query_selection_layer", QUERY_SELECTION_LAYER)),
        len(base.layers) - 2,
    )
    losa_context = None
    losa_active_topk = int(getattr(model.config, "sdar_losa_active_topk", 5))
    if (
        prefix_cache
        and bool(getattr(model.config, "sdar_losa", False))
        and losa_active_topk < input_ids.shape[1]
    ):
        selection_state.setdefault("losa_states", {})
        losa_context = {
            "selection_state": selection_state,
            "prefix_cache_length": 0,
            "active_topk": losa_active_topk,
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
                layer_cache = full_cache
                layer_query_positions = torch.arange(
                    input_ids.shape[1], device=input_ids.device
                )
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
                    dense_fallback_mask_count=(
                        model.config.sdar_sparse_dlm_dense_fallback_mask_count
                    ),
                    minimum_mask_candidates=minimum_mask_candidates,
                    strategy=strategy,
                    threshold=threshold,
                    entropy_budget=entropy_budget,
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
    finally:
        if losa_context is not None:
            model._sdar_losa_context = None

    if query_sparse and refresh_late_kv:
        selection_state["sparse_cache"] = _BlockDualCache(
            full_cache.to_legacy_cache(), prefix_lengths
        )
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
        mask_positions = torch.where(input_ids[0] == mask_id)[0]
    return model.lm_head(hidden_states.index_select(1, mask_positions)), mask_positions


@torch.no_grad()
def _block_diffusion_generate(self, *args, **kwargs):
    inputs = kwargs.pop("inputs", kwargs.pop("input_ids", args[0] if args else None))
    if inputs is None:
        raise ValueError("SDAR generation requires `inputs`")
    if inputs.shape[0] != 1:
        raise ValueError("SDAR runtime patch currently requires batch_size=1")

    gen_length = int(kwargs.pop("gen_length", kwargs.pop("max_new_tokens", 128)))
    block_length = int(kwargs.pop("block_length", 32))
    steps = int(kwargs.pop("denoising_steps", kwargs.pop("steps", block_length)))
    temperature = float(kwargs.pop("temperature", 0.0))
    top_k = kwargs.pop("top_k", 0) or 0
    top_p = kwargs.pop("top_p", 1.0)
    strategy = kwargs.pop("remasking_strategy", "sequential")
    threshold = float(
        kwargs.pop("confidence_threshold", kwargs.pop("threshold", 0.85))
    )
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
    query_sparse = bool(getattr(self.config, "sdar_query_sparse", False))
    prefix_sparse = bool(getattr(self.config, "sdar_prefix_sparse", False))
    losa = bool(getattr(self.config, "sdar_losa", False))
    prefix_token_budget = int(
        getattr(self.config, "sdar_prefix_token_budget", 256)
    )
    prefix_chunk_size = int(getattr(self.config, "sdar_prefix_chunk_size", 256))
    losa_active_topk = int(getattr(self.config, "sdar_losa_active_topk", 5))
    active_losa = losa and losa_active_topk < block_length
    selection_top_k = int(getattr(self.config, "sdar_sparse_dlm_top_k", 64))
    refresh_step = int(getattr(self.config, "sdar_sparse_dlm_refresh_step", -1))
    deep_only_transfer = bool(
        getattr(self.config, "sdar_sparse_dlm_deep_only_transfer", False)
    )
    selection_state = {}

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
                prefix_cache, _ = _compact_prefix_cache(
                    model,
                    dense_cache,
                    block_start,
                    captured_queries,
                    position_ids,
                    prefix_token_budget,
                    prefix_chunk_size,
                )
            else:
                prefix_cache = _legacy_prefix_cache(dense_cache, block_start)
            selection_state.clear()
            selection_state.update(
                positions=None,
                step=step,
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
        return logits, logit_positions

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
            sparse_denoise if query_sparse or prefix_sparse or active_losa else None
        ),
    )
    generated = tokens[:, prompt_length : prompt_length + gen_length]
    if stop_ids:
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
    dense_fallback_mask_count=0,
    refresh_step=-1,
    selection_layer=QUERY_SELECTION_LAYER,
    deep_only_transfer=False,
    query_sparse=True,
    prefix_sparse=False,
    prefix_token_budget=256,
    prefix_chunk_size=256,
    losa=False,
    losa_active_topk=5,
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
    ):
        raise ValueError(
            "top_k, intervals, prefix budget, chunk size, and LoSA active "
            "top-k must be positive; "
            "refresh_step must be >= -2"
        )
    model.config.sdar_sparse_dlm_ratio = min(max(float(ratio), 0.0), 1.0)
    model.config.sdar_sparse_dlm_top_k = int(top_k)
    model.config.sdar_sparse_dlm_selection_interval = int(selection_interval)
    model.config.sdar_sparse_dlm_dense_fallback_mask_count = int(
        dense_fallback_mask_count
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
                _sample_with_confidence(logits, temperature, top_k, top_p)
            ),
            model,
        )
    if not hasattr(model, "_sdar_hf_generate"):
        model._sdar_hf_generate = model.generate
        model.generate = types.MethodType(_block_diffusion_generate, model)
    return model


# Backward-compatible model-local entry point; new callers use sparse.core.
patch_model = patch_sdar_model

"""Runtime block-diffusion generation patch for SDAR checkpoints.

The checkpoint contains the model forward pass but not SDAR's generation
routine. This keeps the official block/KV-cache algorithm plus the optional
late-layer query selector in this repository and leaves the model directory
untouched.
"""

import types

import torch
from transformers.cache_utils import DynamicCache

from .core import (
    _dual_cache_from_dense,
    _legacy_prefix_cache,
    _select_positions,
)
from .sdar_generate import (
    block_diffusion_generate,
    get_num_transfer_tokens as _transfer_counts,
    sample_with_temperature_topk_topp as _sample_with_confidence,
    select_transfer as _select_transfer,
)


# 1-based decoder layer after which Query Sparse chooses mask candidates.
# Layer 4 is too early for SDAR-b32: its candidate ranking diverges sharply
# from the final-layer transfer positions.
QUERY_SELECTION_LAYER = 6


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
    strategy="low_confidence_dynamic",
    threshold=0.85,
    entropy_budget=None,
    refresh_late_kv=False,
):
    base = model.model
    hidden_states = base.embed_tokens(input_ids)
    position_embeddings = base.rotary_emb(hidden_states, position_ids)
    full_cache = DynamicCache.from_legacy_cache(prefix_cache)
    sparse_cache = selection_state["sparse_cache"]
    selected_positions = None
    full_hidden_base = None
    compressed = False
    prefix_length = prefix_cache[0][0].shape[-2] if prefix_cache else 0
    selection_layer = min(QUERY_SELECTION_LAYER, len(base.layers) - 1)

    for layer_idx, decoder_layer in enumerate(base.layers):
        if (
            layer_idx < selection_layer
            or selected_positions is None
            or refresh_late_kv
        ):
            layer_hidden = hidden_states
            layer_positions = position_ids
            layer_position_embeddings = position_embeddings
            layer_cache = full_cache
        else:
            layer_hidden = (
                hidden_states
                if compressed
                else hidden_states.index_select(1, selected_positions)
            )
            layer_positions = position_ids.index_select(1, selected_positions)
            layer_position_embeddings = (
                position_embeddings[0].index_select(1, selected_positions),
                position_embeddings[1].index_select(1, selected_positions),
            )
            sparse_cache.set_positions(selected_positions)
            layer_cache = sparse_cache

        layer_attention = torch.ones(
            input_ids.shape[0],
            layer_hidden.shape[1],
            prefix_length + input_ids.shape[1],
            dtype=torch.bool,
            device=input_ids.device,
        )
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
            selected_positions is not None
            and layer_idx >= selection_layer
            and not refresh_late_kv
        )

        if layer_idx + 1 == selection_layer:
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

    if refresh_late_kv:
        selection_state["sparse_cache"] = _dual_cache_from_dense(
            full_cache,
            prefix_cache,
            prefix_length,
            prefix_length + input_ids.shape[1],
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
    block_length = int(kwargs.pop("block_length", 4))
    steps = int(kwargs.pop("denoising_steps", kwargs.pop("steps", block_length)))
    temperature = float(kwargs.pop("temperature", 0.0))
    top_k = kwargs.pop("top_k", 0) or 0
    top_p = kwargs.pop("top_p", 1.0)
    strategy = kwargs.pop("remasking_strategy", "low_confidence_dynamic")
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
    selection_top_k = int(getattr(self.config, "sdar_sparse_dlm_top_k", 64))
    refresh_step = int(getattr(self.config, "sdar_sparse_dlm_refresh_step", 2))
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
            outputs = model(
                block_tokens,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=dense_cache,
                use_cache=True,
                store_kv=True,
            )
            prefix_cache = _legacy_prefix_cache(dense_cache, block_start)
            selection_state.clear()
            selection_state.update(
                positions=None,
                step=step,
                sparse_cache=_dual_cache_from_dense(
                    dense_cache, prefix_cache, block_start, block_end
                ),
                prefix_cache=prefix_cache,
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
            # transfers. Refresh every late-layer KV once, then stay sparse.
            refresh_late_kv=(
                block_length >= 16 and selection_state["step"] == refresh_step
            ),
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
        denoise_fn=(sparse_denoise if query_sparse else None),
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
    refresh_step=2,
    query_sparse=True,
):
    if getattr(model.config, "model_type", None) != "sdar":
        raise TypeError("SDAR patch requires a model with config.model_type == 'sdar'")
    if top_k <= 0 or selection_interval <= 0 or refresh_step < -1:
        raise ValueError(
            "top_k and selection_interval must be positive; refresh_step must be >= -1"
        )
    model.config.sdar_sparse_dlm_ratio = min(max(float(ratio), 0.0), 1.0)
    model.config.sdar_sparse_dlm_top_k = int(top_k)
    model.config.sdar_sparse_dlm_selection_interval = int(selection_interval)
    model.config.sdar_sparse_dlm_dense_fallback_mask_count = int(
        dense_fallback_mask_count
    )
    model.config.sdar_sparse_dlm_refresh_step = int(refresh_step)
    model.config.sdar_query_sparse = bool(query_sparse)
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

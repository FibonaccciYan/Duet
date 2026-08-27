"""Runtime block-diffusion generation patch for SDAR checkpoints.

The checkpoint contains the model forward pass but not SDAR's generation
routine. This keeps the official block/KV-cache algorithm plus the optional
late-layer query selector in this repository and leaves the model directory
untouched.
"""

import types

import torch
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache

from .core import (
    _dual_cache_from_dense,
    _legacy_prefix_cache,
    _select_positions,
)


def _top_k_logits(logits, top_k):
    if not top_k:
        return logits
    cutoff = torch.topk(logits, min(int(top_k), logits.shape[-1]), dim=-1).values[
        ..., -1, None
    ]
    return logits.masked_fill(logits < cutoff, -torch.inf)


def _top_p_logits(logits, top_p):
    if top_p is None or float(top_p) >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    remove = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1) > float(top_p)
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    remove = torch.zeros_like(remove).scatter(-1, sorted_indices, remove)
    return logits.masked_fill(remove, -torch.inf)


def _sample_with_confidence(logits, temperature=0.0, top_k=0, top_p=1.0):
    if temperature is None or float(temperature) <= 0:
        tokens = logits.argmax(dim=-1)
        confidence = F.softmax(logits, dim=-1).gather(
            -1, tokens.unsqueeze(-1)
        ).squeeze(-1)
        return tokens, confidence

    scaled = logits / float(temperature)
    confidence_probabilities = F.softmax(scaled, dim=-1)
    filtered = _top_k_logits(scaled, top_k)
    filtered = _top_p_logits(filtered, top_p)
    probabilities = F.softmax(filtered, dim=-1)
    shape = probabilities.shape[:-1]
    sampled = torch.multinomial(
        probabilities.reshape(-1, probabilities.shape[-1]), 1
    ).reshape(shape)
    confidence = confidence_probabilities.gather(
        -1, sampled.unsqueeze(-1)
    ).squeeze(-1)
    return sampled, confidence


def _transfer_counts(block_length, steps):
    if block_length <= 0 or steps <= 0 or steps > block_length:
        raise ValueError("SDAR requires 1 <= denoising_steps <= block_length")
    base, remainder = divmod(block_length, steps)
    counts = torch.full((steps,), base, dtype=torch.long)
    counts[:remainder] += 1
    return counts


def _select_transfer(mask, confidence, minimum, strategy, threshold):
    transfer = torch.zeros_like(mask)
    for batch_idx in range(mask.shape[0]):
        positions = torch.where(
            mask[batch_idx] & torch.isfinite(confidence[batch_idx])
        )[0]
        count = min(int(minimum), positions.numel())
        if not count:
            continue
        if strategy == "sequential":
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
                raise ValueError(f"Unsupported SDAR remasking strategy: {strategy}")
        transfer[batch_idx, selected] = True
    return transfer


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

    for layer_idx, decoder_layer in enumerate(base.layers):
        if layer_idx < 2 or selected_positions is None:
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
        compressed = selected_positions is not None and layer_idx >= 2

        if layer_idx == 1 and len(base.layers) > 2:
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
            )
            selection_state["positions"] = selected_positions
            if selected_positions is not None:
                full_hidden_base = hidden_states.clone()

    if selected_positions is not None:
        full_hidden_base[:, selected_positions] = hidden_states
        hidden_states = full_hidden_base
    hidden_states = base.norm(hidden_states)
    if selected_positions is None:
        return model.lm_head(hidden_states), None
    mask_positions = selected_positions[
        input_ids[0].index_select(0, selected_positions) == mask_id
    ]
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

    transfer_counts = _transfer_counts(block_length, steps)
    input_ids = inputs.to(self.device)
    prompt_length = input_ids.shape[1]
    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length
    block_mask = torch.tril(
        torch.ones(num_blocks, num_blocks, dtype=torch.bool, device=self.device)
    )
    attention_mask = (
        block_mask.repeat_interleave(block_length, 0)
        .repeat_interleave(block_length, 1)
        .unsqueeze(0)
    )
    position_ids = torch.arange(total_length, device=self.device).unsqueeze(0)
    tokens = torch.full(
        (1, total_length), mask_id, dtype=torch.long, device=self.device
    )
    tokens[:, :prompt_length] = input_ids
    cache = DynamicCache()

    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length
    if prefill_length:
        self(
            tokens[:, :prefill_length],
            attention_mask=attention_mask[:, :prefill_length, :prefill_length],
            position_ids=position_ids[:, :prefill_length],
            past_key_values=cache,
            use_cache=True,
            store_kv=True,
        )

    stop_ids = _stop_ids(self, eos_id)
    query_sparse = bool(getattr(self.config, "sdar_query_sparse", False))
    for block_idx in range(prefill_blocks, num_blocks):
        start = block_idx * block_length
        end = start + block_length
        block_tokens = tokens[:, start:end].clone()
        block_attention = attention_mask[:, start:end, :end]
        block_positions = position_ids[:, start:end]
        selection_state = None

        for step, minimum in enumerate(transfer_counts):
            mask = block_tokens == mask_id
            if not mask.any():
                break
            logit_positions = None
            if query_sparse and step == 0:
                dense_cache = DynamicCache.from_legacy_cache(
                    cache.to_legacy_cache()
                )
                dense_outputs = self(
                    block_tokens,
                    attention_mask=block_attention,
                    position_ids=block_positions,
                    past_key_values=dense_cache,
                    use_cache=True,
                    store_kv=True,
                )
                logits = dense_outputs.logits
                prefix_cache = _legacy_prefix_cache(dense_cache, start)
                selection_state = {
                    "positions": None,
                    "step": 0,
                    "sparse_cache": _dual_cache_from_dense(
                        dense_cache, prefix_cache, start, end
                    ),
                    "prefix_cache": prefix_cache,
                }
            elif query_sparse:
                logits, logit_positions = _sparse_cached_forward(
                    self,
                    block_tokens,
                    block_positions,
                    selection_state["prefix_cache"],
                    selection_state,
                    mask_id,
                    temperature,
                    top_k,
                    top_p,
                )
                selection_state["step"] += 1
            else:
                logits = self(
                    block_tokens,
                    attention_mask=block_attention,
                    position_ids=block_positions,
                    past_key_values=cache,
                    use_cache=True,
                    store_kv=False,
                ).logits
            prediction, confidence = _sample_with_confidence(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            if logit_positions is not None:
                full_prediction = block_tokens.clone()
                full_prediction.index_copy_(1, logit_positions, prediction)
                full_confidence = torch.full_like(
                    block_tokens, -torch.inf, dtype=confidence.dtype
                )
                full_confidence.index_copy_(1, logit_positions, confidence)
                prediction, confidence = full_prediction, full_confidence
            transfer = _select_transfer(
                mask, confidence, minimum, strategy, threshold
            )
            block_tokens[transfer] = prediction[transfer]

        if torch.any(block_tokens == mask_id):
            raise RuntimeError(
                f"SDAR block {block_idx} still contains masks after {steps} steps"
            )
        self(
            block_tokens,
            attention_mask=block_attention,
            position_ids=block_positions,
            past_key_values=cache,
            use_cache=True,
            store_kv=True,
        )
        tokens[:, start:end] = block_tokens
        if eos_early_stop and stop_ids:
            generated = tokens[0, prompt_length:end]
            if any(torch.any(generated == token_id) for token_id in stop_ids):
                break

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
    query_sparse=True,
):
    if getattr(model.config, "model_type", None) != "sdar":
        raise TypeError("SDAR patch requires a model with config.model_type == 'sdar'")
    if top_k <= 0 or selection_interval <= 0:
        raise ValueError("top_k and selection_interval must be positive")
    model.config.sdar_sparse_dlm_ratio = min(max(float(ratio), 0.0), 1.0)
    model.config.sdar_sparse_dlm_top_k = int(top_k)
    model.config.sdar_sparse_dlm_selection_interval = int(selection_interval)
    model.config.sdar_sparse_dlm_dense_fallback_mask_count = int(
        dense_fallback_mask_count
    )
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

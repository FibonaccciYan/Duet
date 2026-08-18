import math
import types

import torch
from transformers.cache_utils import DynamicCache


def _sample_with_confidence(model, logits, temperature, top_p, top_k):
    return model._sample_with_temperature_topk_topp(
        logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )


def _select_positions(
    model,
    hidden_states,
    input_ids,
    mask_id,
    ratio,
    top_k,
    cached_positions=None,
    selection_step=0,
    selection_interval=1,
    dense_fallback_mask_count=0,
):
    mask = input_ids[0] == mask_id
    mask_count = int(mask.sum().item())
    if ratio >= 1.0 or mask_count <= dense_fallback_mask_count:
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
            return torch.cat((decoded, old_masks[:candidate_count])).sort().values

    mask_positions = torch.where(mask)[0]
    mask_hidden = hidden_states[:, mask_positions, :]
    mask_logits = model.lm_head(mask_hidden).float()
    vocab_top_k = min(max(int(top_k), 1), mask_logits.shape[-1])
    topk_logits = torch.topk(mask_logits, vocab_top_k, dim=-1).values
    probs = torch.softmax(topk_logits, dim=-1)
    confidence = torch.sum(probs * torch.log(probs.clamp_min(1e-12)), dim=-1)
    _, top_indices = torch.topk(confidence[0], k=candidate_count)
    selected_masks = mask_positions[top_indices]
    return torch.cat((decoded, selected_masks)).sort().values


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
    allowed_mask=None,
):
    x0, x0_p = _sample_with_confidence(
        model,
        active_logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    mask_candidates = active_mask if allowed_mask is None else active_mask & allowed_mask
    mask_confidence = torch.where(mask_candidates, x0_p, -torch.inf)
    high_conf_mask = (mask_confidence[0] > threshold) & mask_candidates[0]
    mask_transfer = torch.zeros_like(active_mask)
    if int(high_conf_mask.sum().item()) >= num_to_transfer:
        mask_transfer[0] = high_conf_mask
    else:
        available = int(mask_candidates.sum().item())
        if available:
            _, indices = torch.topk(mask_confidence[0], k=min(num_to_transfer, available))
            mask_transfer[0, indices] = True

    editable = (~active_mask) & (~prompt_mask.unsqueeze(0))
    editing_confidence = torch.where(editable, x0_p, -torch.inf)
    editing = (editing_confidence[0] > editing_threshold) & editable[0]
    editing &= x0[0] != old_block_tokens[0]
    transfer = mask_transfer | editing.unsqueeze(0)
    if transfer.any():
        block_tokens[transfer] = x0[transfer]
    return block_tokens, transfer


def _legacy_prefix_cache(cache, prefix_length):
    legacy = cache.to_legacy_cache()
    return tuple(
        (
            key[:, :, :prefix_length, :].contiguous(),
            value[:, :, :prefix_length, :].contiguous(),
        )
        for key, value in legacy
    )


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
    dense_fallback_mask_count,
):
    base = model.model
    inputs_embeds = base.word_embeddings(input_ids)
    position_embeddings = base.rotary_emb(inputs_embeds, position_ids)
    full_cache = DynamicCache.from_legacy_cache(prefix_cache)
    sparse_cache = DynamicCache.from_legacy_cache(prefix_cache)
    hidden_states = inputs_embeds
    selected_positions = None
    full_hidden_base = None
    compressed_hidden_states = False

    prefix_length = prefix_cache[0][0].shape[-2] if prefix_cache else 0
    for layer_idx, decoder_layer in enumerate(base.layers):
        if layer_idx < 2 or selected_positions is None:
            layer_hidden = hidden_states
            layer_position_ids = position_ids
            layer_position_embeddings = position_embeddings
            layer_attention_mask = attention_mask
            layer_cache = full_cache
        else:
            layer_hidden = (
                hidden_states
                if compressed_hidden_states
                else hidden_states.index_select(1, selected_positions)
            )
            layer_position_ids = position_ids.index_select(1, selected_positions)
            layer_position_embeddings = (
                position_embeddings[0].index_select(1, selected_positions),
                position_embeddings[1].index_select(1, selected_positions),
            )
            selected_mask = attention_mask.index_select(2, selected_positions)
            selected_columns = selected_mask.index_select(3, selected_positions + prefix_length)
            layer_attention_mask = torch.cat(
                (selected_mask[:, :, :, :prefix_length], selected_columns), dim=-1
            )
            layer_cache = sparse_cache

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
        compressed_hidden_states = selected_positions is not None and layer_idx >= 2

        if layer_idx == 1 and len(base.layers) > 2:
            selected_positions = _select_positions(
                model,
                hidden_states,
                input_ids,
                mask_id=mask_id,
                ratio=ratio,
                top_k=top_k,
                cached_positions=selection_state.get("positions"),
                selection_step=selection_state["step"],
                selection_interval=selection_interval,
                dense_fallback_mask_count=dense_fallback_mask_count,
            )
            selection_state["positions"] = selected_positions
            if selected_positions is not None:
                full_hidden_base = hidden_states.clone()

    if selected_positions is not None:
        full_hidden_base[:, selected_positions] = hidden_states
        hidden_states = full_hidden_base

    hidden_states = base.norm(hidden_states)
    return model.lm_head(hidden_states).float(), selected_positions


@torch.no_grad()
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
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=self.device, dtype=torch.bool))
    allowed_attention = (
        block_mask.repeat_interleave(block_length, dim=0)
        .repeat_interleave(block_length, dim=1)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    mask_dtype = next(self.parameters()).dtype
    full_attention_mask = torch.zeros(
        allowed_attention.shape,
        dtype=mask_dtype,
        device=self.device,
    ).masked_fill(~allowed_attention, torch.finfo(mask_dtype).min)
    position_ids = torch.arange(total_length, device=self.device).unsqueeze(0)
    x = torch.full((1, total_length), mask_id, dtype=torch.long, device=self.device)
    x[:, :prompt_length] = input_ids

    ratio = min(max(float(getattr(self.config, "llada_sparse_dlm_ratio", 0.5)), 0.0), 1.0)
    selection_interval = max(1, int(getattr(self.config, "llada_sparse_dlm_selection_interval", 4)))
    selection_top_k = int(getattr(self.config, "llada_sparse_dlm_top_k", 64))
    fallback_count = int(getattr(self.config, "llada_sparse_dlm_dense_fallback_mask_count", 4))
    dense_forward = self._llada_block_cache_dense_forward
    prefill_blocks = prompt_length // block_length

    for block_idx in range(prefill_blocks, num_blocks):
        block_start = block_idx * block_length
        block_end = min((block_idx + 1) * block_length, total_length)
        current_window_end = block_end
        cur_x = x[:, :current_window_end]
        cur_mask = full_attention_mask[:, :, :current_window_end, :current_window_end]
        cur_positions = position_ids[:, :current_window_end]
        old_block_tokens = cur_x[:, -block_length:].clone()
        active_block_mask = cur_x[:, -block_length:] == mask_id
        prompt_mask = torch.zeros(block_length, dtype=torch.bool, device=self.device)
        if block_start < prompt_length:
            prompt_mask[: min(prompt_length - block_start, block_length)] = True

        dense_outputs = dense_forward(
            cur_x,
            attention_mask=cur_mask,
            position_ids=cur_positions,
            use_cache=True,
            return_dict=True,
        )
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

        prefix_cache = _legacy_prefix_cache(dense_outputs.past_key_values, block_start)
        del dense_outputs
        selection_state = {"positions": None, "step": 0}
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
            step_mask = full_attention_mask[:, :, block_start:block_end, :block_end]
            step_positions = position_ids[:, block_start:block_end]
            logits, selected_positions = _cached_forward(
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
                fallback_count,
            )
            selection_state["step"] += 1
            active_logits = logits
            selected_mask = None
            if selected_positions is not None:
                selected_mask = torch.zeros_like(active_block_mask)
                selected_mask[:, selected_positions] = True
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
                allowed_mask=selected_mask,
            )
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
    eos_positions = (generated[0] == eos_id).nonzero(as_tuple=True)[0]
    if len(eos_positions):
        generated = generated[:, : eos_positions[0].item() + 1]
    return generated


def patch_model(
    model,
    ratio=0.5,
    top_k=64,
    selection_interval=4,
    dense_fallback_mask_count=4,
):
    if top_k <= 0 or selection_interval <= 0:
        raise ValueError("top_k and selection_interval must be positive")
    model.config.llada_sparse_dlm_ratio = float(ratio)
    model.config.llada_sparse_dlm_top_k = int(top_k)
    model.config.llada_sparse_dlm_selection_interval = max(1, int(selection_interval))
    model.config.llada_sparse_dlm_dense_fallback_mask_count = int(dense_fallback_mask_count)
    if not hasattr(model, "_llada_block_cache_dense_forward"):
        model._llada_block_cache_dense_forward = model.forward
    model.generate = types.MethodType(_block_cache_generate, model)

import math
import types
from typing import Optional, Tuple

import torch
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import repeat_kv


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_llada_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


def _select_positions_from_layer1(
    hidden_states: torch.Tensor,
    input_ids: torch.LongTensor,
    lm_head,
    model,
    mask_token_id: int,
    block_length: int,
    prompt_length: int,
    ratio: float,
    temperature: float,
    top_p,
    top_k,
    threshold: float,
    editing_threshold: float,
    num_to_transfer: int,
    dense_fallback_mask_count: int,
) -> Optional[torch.LongTensor]:
    input_ids = input_ids.to(hidden_states.device)
    decoded_index = input_ids != int(mask_token_id)
    active_block_mask = input_ids[:, -block_length:] == int(mask_token_id)
    active_mask_count = int(active_block_mask[0].sum().item())
    if float(ratio) >= 1.0 or active_mask_count <= int(dense_fallback_mask_count):
        return None

    logits = lm_head(hidden_states).float()
    input_ids = input_ids.to(logits.device)
    decoded_index = input_ids != int(mask_token_id)
    active_block_mask = input_ids[:, -block_length:] == int(mask_token_id)
    active_mask_count = int(active_block_mask[0].sum().item())
    if float(ratio) >= 1.0 or active_mask_count <= int(dense_fallback_mask_count):
        return None

    full_score = torch.full(
        input_ids.shape,
        -torch.inf,
        dtype=logits.dtype,
        device=logits.device,
    )
    full_score[decoded_index] = torch.inf

    active_logits = logits[:, -block_length:, :]
    _, x0_p = model._sample_with_temperature_topk_topp(
        active_logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    # Confidence ranking predictor: keep all decoded tokens, then select the
    # top-ratio active masks by early-layer token confidence.
    candidate_count = min(
        max(1, math.ceil(active_mask_count * float(ratio))),
        active_mask_count,
    )
    if candidate_count > 0:
        mask_scores = torch.where(active_block_mask, x0_p, -torch.inf)
        _, idx = torch.topk(mask_scores[0], k=candidate_count)
        block_scores = full_score[:, -block_length:]
        block_scores[0, idx] = mask_scores[0, idx]

    # Prompt tokens and already decoded tokens are preserved by decoded_index.
    decoded_count = int(decoded_index[0].sum().item())
    total_count = decoded_count + candidate_count
    if total_count <= 0:
        return torch.empty(input_ids.shape[0], 0, dtype=torch.long, device=input_ids.device)
    _, positions = torch.topk(full_score, k=total_count, dim=-1)
    positions, _ = torch.sort(positions, dim=-1)
    return positions


def _gather_4d_attention_mask(attention_mask, positions):
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 4:
        return attention_mask
    bsz, _, _, seq_len = attention_mask.shape
    q_len = positions.shape[-1]
    row_index = positions.view(bsz, 1, q_len, 1).expand(bsz, 1, q_len, seq_len)
    reduced = attention_mask.gather(2, row_index)
    col_index = positions.view(bsz, 1, 1, q_len).expand(bsz, 1, q_len, q_len)
    return reduced.gather(3, col_index)


def _gather_4d_attention_mask_columns(attention_mask, positions):
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 4:
        return attention_mask
    bsz, _, q_len, _ = attention_mask.shape
    sparse_len = positions.shape[-1]
    col_index = positions.view(bsz, 1, 1, sparse_len).expand(
        bsz,
        1,
        q_len,
        sparse_len,
    )
    return attention_mask.gather(3, col_index)


def _gather_4d_attention_mask_rows(attention_mask, positions):
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 4:
        return attention_mask
    bsz, _, _, seq_len = attention_mask.shape
    sparse_len = positions.shape[-1]
    row_index = positions.view(bsz, 1, sparse_len, 1).expand(
        bsz,
        1,
        sparse_len,
        seq_len,
    )
    return attention_mask.gather(2, row_index)


def _llada_sparse_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    sparse_state = getattr(self.config, "_llada_demo_sparse_state", None)
    layer_idx = getattr(self, "layer_idx", None)
    use_sparse = (
        hidden_states.shape[1] > 1
        and sparse_state is not None
        and layer_idx is not None
        and layer_idx >= 2
        and sparse_state.get("query_positions") is not None
        and sparse_state["query_positions"].shape[-1] > 0
    )
    if not use_sparse:
        return self._llada_demo_sparse_dense_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    input_shape = hidden_states.shape[:-1]
    bsz, q_len, _ = hidden_states.size()

    qkv = self.query_key_value(hidden_states)
    qkv = qkv.view(
        bsz,
        q_len,
        self.num_heads + 2 * self.num_key_value_heads,
        self.head_dim,
    )
    query_states, key_states, value_states = qkv.split(
        [self.num_heads, self.num_key_value_heads, self.num_key_value_heads],
        dim=-2,
    )
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    if self.config.use_qk_norm:
        query_states = self.query_layernorm(query_states)
        key_states = self.key_layernorm(key_states)

    cos, sin = position_embeddings
    cos = cos.to(query_states.device)
    sin = sin.to(query_states.device)
    query_states, key_states = _apply_llada_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    positions = sparse_state["query_positions"].to(query_states.device)
    sparse_len = positions.shape[-1]
    sparse_mode = sparse_state.get("sparse_mode", "kv")
    if sparse_mode == "kv":
        # Full Q + sparse K/V.
        full_q = query_states
        gather_kv = positions.view(bsz, 1, sparse_len, 1).expand(
            bsz, self.num_key_value_heads, sparse_len, self.head_dim
        )
        sparse_k = key_states.gather(2, gather_kv)
        sparse_v = value_states.gather(2, gather_kv)

        sparse_k = repeat_kv(sparse_k, self.num_key_value_groups)
        sparse_v = repeat_kv(sparse_v, self.num_key_value_groups)
        sparse_mask = _gather_4d_attention_mask_columns(attention_mask, positions)

        attn_weights = torch.matmul(full_q, sparse_k.transpose(2, 3)) * self.scaling
        if sparse_mask is not None:
            attn_weights = attn_weights + sparse_mask[:, :, :, : sparse_k.shape[-2]].to(attn_weights.device)
        probs = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(full_q.dtype)
        attn_output = torch.matmul(probs, sparse_v)
    elif sparse_mode == "q":
        # Sparse Q + full K/V.
        gather_q = positions.view(bsz, 1, sparse_len, 1).expand(
            bsz, self.num_heads, sparse_len, self.head_dim
        )
        sparse_q = query_states.gather(2, gather_q)
        full_k = repeat_kv(key_states, self.num_key_value_groups)
        full_v = repeat_kv(value_states, self.num_key_value_groups)
        sparse_mask = _gather_4d_attention_mask_rows(attention_mask, positions)

        attn_weights = torch.matmul(sparse_q, full_k.transpose(2, 3)) * self.scaling
        if sparse_mask is not None:
            attn_weights = attn_weights + sparse_mask[:, :, :, : full_k.shape[-2]].to(attn_weights.device)
        probs = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(sparse_q.dtype)
        sparse_output = torch.matmul(probs, full_v)

        attn_output = torch.zeros_like(query_states)
        scatter_index = positions.view(bsz, 1, sparse_len, 1).expand(
            bsz, self.num_heads, sparse_len, self.head_dim
        )
        attn_output.scatter_(2, scatter_index, sparse_output)
    else:
        raise ValueError(f"Unsupported llada demo sparse mode: {sparse_mode}")

    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = self.dense(attn_output)
    return attn_output, None, past_key_value


def _llada_sparse_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    output_router_logits: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, self_attn_weights, present_key_value = self.attention(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        position_embeddings=position_embeddings,
        use_cache=use_cache,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    if isinstance(hidden_states, tuple):
        hidden_states, router_logits = hidden_states
    else:
        router_logits = None
    hidden_states = residual + hidden_states.to(residual.device)

    sparse_state = getattr(self.attention.config, "_llada_demo_sparse_state", None)
    layer_idx = getattr(self.attention, "layer_idx", None)
    if (
        sparse_state is not None
        and layer_idx == 1
        and sparse_state.get("query_positions") is None
    ):
        sparse_state["query_positions"] = _select_positions_from_layer1(
            hidden_states=hidden_states,
            input_ids=sparse_state["input_ids"],
            lm_head=sparse_state["lm_head"],
            model=sparse_state["model"],
            mask_token_id=sparse_state["mask_token_id"],
            block_length=sparse_state["block_length"],
            prompt_length=sparse_state["prompt_length"],
            ratio=sparse_state["ratio"],
            temperature=sparse_state["temperature"],
            top_p=sparse_state["top_p"],
            top_k=sparse_state["top_k"],
            threshold=sparse_state["threshold"],
            editing_threshold=sparse_state["editing_threshold"],
            num_to_transfer=sparse_state["num_to_transfer"],
            dense_fallback_mask_count=sparse_state["dense_fallback_mask_count"],
        )

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)
    if use_cache:
        outputs += (present_key_value,)
    if output_router_logits:
        outputs += (router_logits,)
    return outputs


def _llada_sparse_model_forward(self, *args, **kwargs):
    input_ids = kwargs.get("input_ids")
    if input_ids is None and args:
        input_ids = args[0]
    if input_ids is None:
        return self._llada_demo_sparse_dense_forward(*args, **kwargs)

    self.config._llada_demo_sparse_state = {
        "input_ids": input_ids,
        "lm_head": self.lm_head,
        "model": self,
        "mask_token_id": int(getattr(self.config, "llada_demo_sparse_mask_id", 156895)),
        "block_length": int(getattr(self.config, "llada_demo_sparse_block_length", 32)),
        "prompt_length": int(getattr(self.config, "llada_demo_sparse_prompt_length", 0)),
        "ratio": float(getattr(self.config, "llada_demo_sparse_ratio", 0.5)),
        "temperature": float(getattr(self.config, "llada_demo_sparse_temperature", 0.0)),
        "top_p": getattr(self.config, "llada_demo_sparse_top_p", None),
        "top_k": getattr(self.config, "llada_demo_sparse_top_k", None),
        "sparse_mode": getattr(self.config, "llada_demo_sparse_mode", "kv"),
        "threshold": float(getattr(self.config, "llada_demo_sparse_threshold", 0.5)),
        "editing_threshold": float(
            getattr(self.config, "llada_demo_sparse_editing_threshold", 0.0)
        ),
        "num_to_transfer": int(getattr(self.config, "llada_demo_sparse_num_to_transfer", 1)),
        "dense_fallback_mask_count": int(
            getattr(self.config, "llada_demo_sparse_dense_fallback_mask_count", 8)
        ),
        "query_positions": None,
    }
    try:
        return self._llada_demo_sparse_dense_forward(*args, **kwargs)
    finally:
        sparse_state = getattr(self.config, "_llada_demo_sparse_state", None)
        query_positions = (
            sparse_state.get("query_positions")
            if isinstance(sparse_state, dict)
            else None
        )
        self.config.llada_demo_sparse_last_query_positions = (
            query_positions.detach().cpu() if query_positions is not None else None
        )
        self.config._llada_demo_sparse_state = None


def patch_model(
    model,
    ratio: float = 0.5,
    block_length: int = 32,
    mask_id: int = 156895,
    temperature: float = 0.0,
    top_p=None,
    top_k=None,
    sparse_mode: str = "kv",
    threshold: float = 0.5,
    editing_threshold: float = 0.0,
    num_to_transfer: int = 1,
    dense_fallback_mask_count: int = 8,
) -> None:
    if sparse_mode not in {"kv", "q"}:
        raise ValueError(f"Unsupported llada demo sparse mode: {sparse_mode}")
    model.config.llada_demo_sparse_ratio = float(ratio)
    model.config.llada_demo_sparse_block_length = int(block_length)
    model.config.llada_demo_sparse_mask_id = int(mask_id)
    model.config.llada_demo_sparse_temperature = float(temperature)
    model.config.llada_demo_sparse_top_p = top_p
    model.config.llada_demo_sparse_top_k = top_k
    model.config.llada_demo_sparse_mode = sparse_mode
    model.config.llada_demo_sparse_threshold = float(threshold)
    model.config.llada_demo_sparse_editing_threshold = float(editing_threshold)
    model.config.llada_demo_sparse_num_to_transfer = int(num_to_transfer)
    model.config.llada_demo_sparse_dense_fallback_mask_count = int(
        dense_fallback_mask_count
    )

    if not hasattr(model, "_llada_demo_sparse_dense_forward"):
        model._llada_demo_sparse_dense_forward = model.forward
    model.forward = types.MethodType(_llada_sparse_model_forward, model)

    for module in model.modules():
        if type(module).__name__ == "LLaDA2MoeAttention":
            if not hasattr(module, "_llada_demo_sparse_dense_forward"):
                module._llada_demo_sparse_dense_forward = module.forward
            module.forward = types.MethodType(_llada_sparse_attention_forward, module)
        elif type(module).__name__ == "LLaDA2MoeDecoderLayer":
            module.forward = types.MethodType(_llada_sparse_decoder_layer_forward, module)

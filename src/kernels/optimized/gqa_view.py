"""Represent GQA sharing using batch/head strides, without duplicating KV."""
import types
import torch.nn.functional as F


def attention(q, k, v, mask=None, scale=None):
    batch, hq, nq, dim = q.shape
    hkv, nk = k.shape[1:3]
    if hq % hkv:
        raise ValueError("Hq must be divisible by Hkv")
    if batch != 1:
        raise ValueError("GQA view currently supports batch size one")
    group = hq // hkv
    grouped_q = q.reshape(hkv, group, nq, dim)
    grouped_k = k.reshape(hkv, 1, nk, dim).expand(hkv, group, nk, dim)
    grouped_v = v.reshape(hkv, 1, nk, dim).expand(hkv, group, nk, dim)
    return F.scaled_dot_product_attention(
        grouped_q, grouped_k, grouped_v, attn_mask=mask,
        dropout_p=0.0, is_causal=False, scale=scale,
    ).reshape(batch, hq, nq, dim)


def _llada_forward(self, hidden_states, attention_mask=None, position_ids=None,
                   past_key_value=None, output_attentions=False, use_cache=False,
                   position_embeddings=None, **kwargs):
    original = self._versioned_gqa_original
    rotary = original.__globals__.get("apply_rotary_pos_emb")
    if output_attentions or rotary is None or hidden_states.shape[0] != 1:
        return original(hidden_states=hidden_states, attention_mask=attention_mask,
                        position_ids=position_ids, past_key_value=past_key_value,
                        output_attentions=output_attentions, use_cache=use_cache,
                        position_embeddings=position_embeddings, **kwargs)
    batch, length, _ = hidden_states.shape
    qkv = self.query_key_value(hidden_states).view(
        batch, length, self.num_heads + 2*self.num_key_value_heads, self.head_dim)
    q, k, v = qkv.split(
        [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2)
    q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
    if self.config.use_qk_norm:
        q, k = self.query_layernorm(q), self.key_layernorm(k)
    cos, sin = position_embeddings
    q, k = rotary(q, k, cos, sin)
    if past_key_value is not None:
        k, v = past_key_value.update(k, v, self.layer_idx, {"sin":sin, "cos":cos})
    out = attention(q,k,v,attention_mask,float(self.scaling))
    out = out.transpose(1,2).reshape(batch,length,-1).contiguous()
    return self.dense(out), None, past_key_value


def install_llada(model):
    """Opt-in dense SDPA adapter; retain the existing specialized prefill path."""
    count = 0
    for layer in model.model.layers:
        module = getattr(layer, "attention", None)
        if module is None or getattr(module.config, "_attn_implementation", None) != "sdpa":
            continue
        if not hasattr(module, "_versioned_gqa_original"):
            target = "_llada_prefill_dense_forward" if hasattr(module, "_llada_prefill_dense_forward") else "forward"
            original = getattr(module, target)
            if "apply_rotary_pos_emb" not in original.__globals__:
                continue
            module._versioned_gqa_original = original
            module._versioned_gqa_target = target
        setattr(module, module._versioned_gqa_target, types.MethodType(_llada_forward, module))
        count += 1
    return count


def restore_llada(model):
    for layer in model.model.layers:
        module = getattr(layer, "attention", None)
        if module is not None and hasattr(module, "_versioned_gqa_original"):
            setattr(module, module._versioned_gqa_target, module._versioned_gqa_original)

"""Batched block-causal prefill; no FOCUS selection occurs in prompt blocks.

Uses the same attention kernel as the repository's matched dense/Sparse
prefill. Visibility is block-causal, not token-causal. Floating-point batching
can change rounding; validate quality separately from mathematical support.
"""
import torch
from transformers.cache_utils import DynamicCache
from src.reference.sparse.triton_kernels import block_causal_prefill



@torch.inference_mode()
def build_prefix(model, family, inputs, positions, *, block_length=32, chunk_length=4096):
    if inputs.shape[-1] % block_length or chunk_length % block_length:
        raise ValueError("prefill inputs and chunk length must be block aligned")
    base=model.model
    cache=DynamicCache()
    if inputs.shape[-1]==0:
        return ()
    for start in range(0,inputs.shape[-1],chunk_length):
        end=min(start+chunk_length,inputs.shape[-1])
        tokens=inputs[:,start:end]
        hidden=base.word_embeddings(tokens) if family=="llada" else base.embed_tokens(tokens)
        rope=base.rotary_emb(hidden,positions[:,start:end])
        for index,layer in enumerate(base.layers):
            attn=layer.attention if family=="llada" else layer.self_attn
            normalized=layer.input_layernorm(hidden)
            q,k,v=_project_qkv(attn,normalized,rope,family)
            cache_kwargs={"sin":rope[1],"cos":rope[0]} if family=="llada" else None
            k,v=cache.update(k,v,index,cache_kwargs)
            if index==len(base.layers)-1:
                # All consumers need only this layer's KV, not its hidden output.
                continue
            if float(attn.scaling)!=q.shape[-1]**-0.5:
                raise ValueError("prefill kernel scale differs from model attention scale")
            output=block_causal_prefill(q,k,v,block_length)
            output=output.transpose(1,2).reshape(*hidden.shape[:-1],-1).contiguous()
            output=attn.dense(output) if family=="llada" else attn.o_proj(output)
            hidden=_finish_layer(layer,hidden,output,family)
    return cache.to_legacy_cache()


# Shared projection helpers preserved from the prior prefill implementation.
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

"""Batched block-causal prefill; no FOCUS selection occurs in prompt blocks.

Uses the same attention kernel as the repository's matched dense/Sparse
prefill. Visibility is block-causal, not token-causal. Floating-point batching
can change rounding; validate quality separately from mathematical support.
"""
import torch
from transformers.cache_utils import DynamicCache
from src.reference.sparse.triton_kernels import block_causal_prefill
from .model import _project_qkv, _finish_layer


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

"""Experimental dense GQA initialization, keeping natural-log softmax state."""
from flash_attn import flash_attn_func


def attention(q, k, v, *, softmax_scale=None):
    output, lse, _ = flash_attn_func(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
        dropout_p=0.0, softmax_scale=softmax_scale, causal=False,
        return_attn_probs=True,
    )
    return output[0], lse[0].transpose(0, 1).contiguous()

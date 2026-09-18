"""FP32 GQA reference algebra without materializing repeated K/V."""
import torch


def attention(q, k, v, *, softmax_scale=None):
    rows, heads, dim = q.shape
    kv_heads = k.shape[1]
    group = heads // kv_heads
    if heads % kv_heads:
        raise ValueError("invalid GQA group")
    scale = dim**-0.5 if softmax_scale is None else softmax_scale
    grouped_q = q.permute(1,0,2).reshape(kv_heads,group*rows,dim).float()
    grouped_k = k.permute(1,2,0).float()
    grouped_v = v.permute(1,0,2).float()
    scores = torch.bmm(grouped_q,grouped_k) * scale
    lse = torch.logsumexp(scores,dim=-1)
    probabilities = torch.softmax(scores,dim=-1)
    output = torch.bmm(probabilities,grouped_v)
    return (output.reshape(heads,rows,dim).permute(1,0,2).to(q.dtype),
            lse.reshape(heads,rows).transpose(0,1))

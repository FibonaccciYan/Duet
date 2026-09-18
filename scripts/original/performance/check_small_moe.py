"""Numerical and latency gate for unchanged routing, including expert skew."""
import json
import time
from types import SimpleNamespace
import torch
from src.reference.losa.moe_patch import _triton_moe_infer
from src.kernels.optimized.small_moe import small_moe


def latency(fn):
    for _ in range(3):
        fn()
    samples = []
    for _ in range(7):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter()-start)*100)
    return sorted(samples)[len(samples)//2]


torch.manual_seed(42)
for dtype in (torch.bfloat16, torch.float16):
    gate = torch.randn(256, 512, 2048, device="cuda", dtype=dtype) * 0.02
    up = torch.randn_like(gate) * 0.02
    down = torch.randn(256, 2048, 512, device="cuda", dtype=dtype) * 0.02
    block = SimpleNamespace(_paper_losa_moe_gate_weight=gate,
                            _paper_losa_moe_up_weight=up,
                            _paper_losa_moe_down_weight=down)
    for tokens in (1, 5, 16, 32):
        x = torch.randn(tokens, 2048, device="cuda", dtype=dtype)
        for skew in (False, True):
            ids = (torch.arange(8, device="cuda").expand(tokens, -1).contiguous()
                   if skew else torch.randn(tokens, 256, device="cuda").topk(8).indices)
            weights = torch.randn(tokens, 8, device="cuda").softmax(-1)
            old = lambda: _triton_moe_infer(block, x, ids, weights)
            new = lambda: small_moe(x, ids, weights, gate, up, down)
            a, b = old(), new()
            torch.testing.assert_close(a, b, atol=0, rtol=0)
            before, after = latency(old), latency(new)
            print(json.dumps(dict(dtype=str(dtype), tokens=tokens, skew=skew,
                                  bitwise_equal=torch.equal(a,b), old_ms=before,
                                  new_ms=after, speedup=before/after)), flush=True)

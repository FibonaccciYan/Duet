"""Small-token packed MoE without host reads of routing counts.

Expert choices, routing weights, GEMM tile/reduction and combination are unchanged.
Only models with unique top-k expert IDs and <=32 input tokens use this path.
"""
import types
import torch
import triton
import triton.language as tl

@triton.jit
def _routes(expert_ids, order, counts, TOTAL: tl.constexpr,
            CAPACITY: tl.constexpr, BLOCK: tl.constexpr):
    expert = tl.program_id(0)
    route = tl.arange(0, BLOCK)
    ids = tl.load(expert_ids + route, route < TOTAL, other=-1)
    match = (route < TOTAL) & (ids == expert)
    rank = tl.cumsum(match.to(tl.int32)) - 1
    count = tl.sum(match.to(tl.int32))
    tl.store(counts + expert, count)
    tl.store(order + expert * CAPACITY + rank, route,
             mask=match & (rank < CAPACITY))

@triton.jit
def _moe_gate_up_kernel(
    x_ptr,
    order_ptr,
    top_k: tl.constexpr,
    gate_weight_ptr,
    up_weight_ptr,
    activated_ptr,
    counts_ptr,
    hidden_size,
    intermediate_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    expert_id = pid_m
    local_tile_m = 0

    rows = local_tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    count = tl.load(counts_ptr + expert_id)
    if count > 0:
        start = expert_id * BLOCK_M
        row_ids = start + rows
        row_mask = rows < count
        col_mask = cols < intermediate_size

        source_rows = tl.load(order_ptr + row_ids, mask=row_mask, other=0) // top_k
        gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, tl.cdiv(hidden_size, BLOCK_K)):
            ks = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
            k_mask = ks < hidden_size
            x_tile = tl.load(
                x_ptr + source_rows[:, None] * hidden_size + ks[None, :],
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            gate_tile = tl.load(
                gate_weight_ptr
                + expert_id * intermediate_size * hidden_size
                + cols[:, None] * hidden_size
                + ks[None, :],
                mask=col_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            up_tile = tl.load(
                up_weight_ptr
                + expert_id * intermediate_size * hidden_size
                + cols[:, None] * hidden_size
                + ks[None, :],
                mask=col_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            gate_acc += tl.dot(x_tile, tl.trans(gate_tile), out_dtype=tl.float32)
            up_acc += tl.dot(x_tile, tl.trans(up_tile), out_dtype=tl.float32)

        gate_values = gate_acc.to(x_ptr.dtype.element_ty).to(tl.float32)
        up_values = up_acc.to(x_ptr.dtype.element_ty).to(tl.float32)
        activated = gate_values * tl.sigmoid(gate_values) * up_values
        tl.store(
            activated_ptr + row_ids[:, None] * intermediate_size + cols[None, :],
            activated,
            mask=row_mask[:, None] & col_mask[None, :],
        )


@triton.jit
def _moe_down_kernel(
    activated_ptr,
    down_weight_ptr,
    out_ptr,
    order_ptr,
    counts_ptr,
    hidden_size,
    intermediate_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    expert_id = pid_m
    local_tile_m = 0

    rows = local_tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    count = tl.load(counts_ptr + expert_id)
    if count > 0:
        start = expert_id * BLOCK_M
        row_ids = start + rows
        row_mask = rows < count
        col_mask = cols < hidden_size

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, tl.cdiv(intermediate_size, BLOCK_K)):
            ks = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
            k_mask = ks < intermediate_size
            activated = tl.load(
                activated_ptr + row_ids[:, None] * intermediate_size + ks[None, :],
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            weights = tl.load(
                down_weight_ptr
                + expert_id * hidden_size * intermediate_size
                + cols[:, None] * intermediate_size
                + ks[None, :],
                mask=col_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.dot(activated, tl.trans(weights), out_dtype=tl.float32)

        original_route = tl.load(order_ptr + row_ids, mask=row_mask, other=0)
        tl.store(
            out_ptr + original_route[:, None] * hidden_size + cols[None, :],
            acc,
            mask=row_mask[:, None] & col_mask[None, :],
        )


def small_moe(x, ids, weights, gate, up, down):
    """Requires <=32 rows and distinct expert IDs within each row (top-k)."""
    ids = ids.reshape(-1, ids.shape[-1]).contiguous()
    weights = weights.reshape_as(ids)
    tokens, top_k = ids.shape
    if tokens == 0:
        return torch.empty_like(x)
    if tokens > 32 or not x.is_contiguous():
        raise ValueError("small_moe requires contiguous x with at most 32 rows")
    experts, intermediate, hidden = gate.shape
    order = torch.empty((experts, 32), dtype=torch.int32, device=x.device)
    counts = torch.empty(experts, dtype=torch.int32, device=x.device)
    _routes[(experts,)](ids, order, counts, TOTAL=ids.numel(), CAPACITY=32,
                       BLOCK=triton.next_power_of_2(ids.numel()))
    activated = torch.empty((experts * 32, intermediate), dtype=torch.float32, device=x.device)
    _moe_gate_up_kernel[(experts, triton.cdiv(intermediate, 128))](
        x, order, top_k, gate, up, activated, counts, hidden, intermediate,
        BLOCK_M=32, BLOCK_N=128, BLOCK_K=64)
    routed = torch.empty((tokens * top_k, hidden), dtype=x.dtype, device=x.device)
    _moe_down_kernel[(experts, triton.cdiv(hidden, 128))](
        activated, down, routed, order, counts, hidden, intermediate,
        BLOCK_M=32, BLOCK_N=128, BLOCK_K=64, num_warps=8)
    # Preserve the baseline dtype conversion, weighting and reduction.
    return (routed.view(tokens, top_k, hidden).to(weights.dtype)
            .mul_(weights.unsqueeze(-1)).sum(dim=1).to(x.dtype))


def _infer(self, x, topk_ids, topk_weight):
    if x.shape[0] > 32 or not x.is_contiguous():
        return self._versioned_moe_original(x, topk_ids, topk_weight)
    prefix = self._versioned_moe_weight_prefix
    return small_moe(x, topk_ids, topk_weight,
                     getattr(self, prefix + "gate_weight"),
                     getattr(self, prefix + "up_weight"),
                     getattr(self, prefix + "down_weight"))


def install_small_moe(model):
    """Opt in after the existing packed-weight patch. Does not repack weights."""
    count = 0
    for module in model.modules():
        for prefix in ("_paper_losa_moe_", "_llada_moe_"):
            if hasattr(module, prefix + "gate_weight") and hasattr(module, "moe_infer"):
                if not hasattr(module, "_versioned_moe_original"):
                    module._versioned_moe_original = module.moe_infer
                module._versioned_moe_weight_prefix = prefix
                module.moe_infer = types.MethodType(_infer, module)
                count += 1
                break
    return count


def restore_small_moe(model):
    for module in model.modules():
        if hasattr(module, "_versioned_moe_original"):
            module.moe_infer = module._versioned_moe_original

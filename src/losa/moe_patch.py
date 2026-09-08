"""Self-owned routed-MoE expert inference patch for LLaDA2.x.

This module is based on the read-only audit of ysy's MoE acceleration idea, but
it does not import `/data0/ysy/sparse`.  It preserves LLaDA2.x routing
semantics and replaces only the routed expert MLP execution backend.
"""

from __future__ import annotations

import types
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


MOE_BLOCK_M = 32
MOE_BLOCK_N = 128
MOE_BLOCK_K = 64


@dataclass(frozen=True)
class MoEPatchReport:
    requested: bool
    patched_blocks: int
    backend: str


@triton.jit
def _route_small_kernel(
    expert_ids,
    counts,
    offsets,
    order,
    total_rows: tl.constexpr,
    num_experts: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.arange(0, BLOCK)
    tl.store(counts + row, 0, mask=row < num_experts)
    tl.debug_barrier()

    valid = row < total_rows
    expert = tl.load(expert_ids + row, mask=valid, other=0).to(tl.int32)
    rank = tl.atomic_add(counts + expert, 1, mask=valid)
    tl.debug_barrier()

    destination = rank
    prefix = tl.zeros((BLOCK,), dtype=tl.int32)
    for expert_idx in range(num_experts):
        count = tl.load(counts + expert_idx)
        destination += tl.where(expert_idx < expert, count, 0)
        prefix += tl.where(expert_idx < row, count, 0)
    tl.store(order + destination, row, mask=valid)
    tl.store(offsets + row, prefix, mask=row <= num_experts)


def _route_by_expert(expert_ids: torch.Tensor, num_experts: int):
    total_rows = int(expert_ids.numel())
    if total_rows <= 256:
        counts = torch.empty(num_experts, dtype=torch.int32, device=expert_ids.device)
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=expert_ids.device)
        order = torch.empty(total_rows, dtype=torch.long, device=expert_ids.device)
        _route_small_kernel[(1,)](
            expert_ids,
            counts,
            offsets,
            order,
            total_rows=total_rows,
            num_experts=int(num_experts),
            BLOCK=triton.next_power_of_2(max(total_rows, num_experts + 1)),
            num_warps=4,
        )
        return counts, offsets, order

    counts = torch.bincount(expert_ids, minlength=num_experts)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    order = torch.argsort(expert_ids)
    return counts, offsets, order


@triton.jit
def _moe_gate_up_kernel(
    x_ptr,
    gate_weight_ptr,
    up_weight_ptr,
    activated_ptr,
    tile_experts_ptr,
    tile_offsets_ptr,
    offsets_ptr,
    counts_ptr,
    total_rows,
    hidden_size,
    intermediate_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    expert_id = tl.load(tile_experts_ptr + pid_m)
    local_tile_m = pid_m - tl.load(tile_offsets_ptr + expert_id)

    rows = local_tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    count = tl.load(counts_ptr + expert_id)
    start = tl.load(offsets_ptr + expert_id)
    row_ids = start + rows
    row_mask = (rows < count) & (row_ids < total_rows)
    col_mask = cols < intermediate_size

    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, tl.cdiv(hidden_size, BLOCK_K)):
        ks = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = ks < hidden_size
        x_tile = tl.load(
            x_ptr + row_ids[:, None] * hidden_size + ks[None, :],
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
    tile_experts_ptr,
    tile_offsets_ptr,
    offsets_ptr,
    counts_ptr,
    total_rows,
    hidden_size,
    intermediate_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    expert_id = tl.load(tile_experts_ptr + pid_m)
    local_tile_m = pid_m - tl.load(tile_offsets_ptr + expert_id)

    rows = local_tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    count = tl.load(counts_ptr + expert_id)
    start = tl.load(offsets_ptr + expert_id)
    row_ids = start + rows
    row_mask = (rows < count) & (row_ids < total_rows)
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


def _triton_moe_infer(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor):
    topk_ids = topk_ids.reshape(-1, topk_ids.shape[-1])
    topk_weight = topk_weight.reshape_as(topk_ids)
    token_count, top_k = topk_ids.shape
    if token_count == 0:
        return x.new_empty((0, x.shape[-1]))

    gate_weight = self._paper_losa_moe_gate_weight
    up_weight = self._paper_losa_moe_up_weight
    down_weight = self._paper_losa_moe_down_weight
    num_experts, intermediate_size, hidden_size = gate_weight.shape

    flat_ids = topk_ids.reshape(-1)
    counts, offsets, order = _route_by_expert(flat_ids, int(num_experts))
    sorted_tokens = x.index_select(0, order // top_k).contiguous()
    total_rows = int(sorted_tokens.shape[0])

    tile_counts = torch.div(counts + MOE_BLOCK_M - 1, MOE_BLOCK_M, rounding_mode="floor")
    tile_offsets = torch.cat((tile_counts.new_zeros(1), tile_counts.cumsum(0)))
    total_tiles = int(tile_offsets[-1].item())
    if total_tiles == 0:
        return x.new_zeros((token_count, hidden_size))
    tile_experts = torch.repeat_interleave(tile_counts, output_size=total_tiles)

    activated = torch.empty((total_rows, intermediate_size), dtype=torch.float32, device=x.device)
    _moe_gate_up_kernel[
        (total_tiles, triton.cdiv(intermediate_size, MOE_BLOCK_N))
    ](
        sorted_tokens,
        gate_weight,
        up_weight,
        activated,
        tile_experts,
        tile_offsets,
        offsets,
        counts,
        total_rows,
        hidden_size,
        intermediate_size,
        BLOCK_M=MOE_BLOCK_M,
        BLOCK_N=MOE_BLOCK_N,
        BLOCK_K=MOE_BLOCK_K,
    )

    routed_out = torch.empty((total_rows, hidden_size), dtype=x.dtype, device=x.device)
    _moe_down_kernel[
        (total_tiles, triton.cdiv(hidden_size, MOE_BLOCK_N))
    ](
        activated,
        down_weight,
        routed_out,
        order,
        tile_experts,
        tile_offsets,
        offsets,
        counts,
        total_rows,
        hidden_size,
        intermediate_size,
        BLOCK_M=MOE_BLOCK_M,
        BLOCK_N=MOE_BLOCK_N,
        BLOCK_K=MOE_BLOCK_K,
        num_warps=8,
    )

    return (
        routed_out.view(token_count, top_k, hidden_size)
        .to(topk_weight.dtype)
        .mul_(topk_weight.unsqueeze(-1))
        .sum(dim=1)
        .to(x.dtype)
    )


def _packed_expert_weights(block):
    experts = list(getattr(block, "experts", ()))
    if not experts:
        return None
    packed = []
    for projection_name in ("gate_proj", "up_proj", "down_proj"):
        weights = [getattr(expert, projection_name).weight for expert in experts]
        if any(weight is None for weight in weights):
            return None
        packed.append(torch.stack([weight.detach() for weight in weights], dim=0).contiguous())
    return tuple(packed)


def _patch_one_block(block) -> bool:
    if getattr(block, "_paper_losa_moe_expert_patched", False):
        return True
    if not hasattr(block, "experts") or not hasattr(block, "moe_infer"):
        return False

    weights = _packed_expert_weights(block)
    if weights is None:
        return False
    if any(weight.device.type != "cuda" for weight in weights):
        return False
    gate_weight, up_weight, down_weight = weights
    if gate_weight.ndim != 3 or up_weight.shape != gate_weight.shape:
        return False
    if down_weight.ndim != 3 or down_weight.shape[0] != gate_weight.shape[0]:
        return False

    block.register_buffer("_paper_losa_moe_gate_weight", gate_weight, persistent=False)
    block.register_buffer("_paper_losa_moe_up_weight", up_weight, persistent=False)
    block.register_buffer("_paper_losa_moe_down_weight", down_weight, persistent=False)

    # Inference-only memory reduction: the packed buffers above become the
    # canonical runtime weights.  The loaded checkpoint on disk is untouched.
    for expert in block.experts:
        expert.gate_proj.weight = None
        expert.up_proj.weight = None
        expert.down_proj.weight = None
    block.moe_infer = types.MethodType(_triton_moe_infer, block)
    block._paper_losa_moe_expert_patched = True
    return True


def patch_moe_experts(model, *, family: str | None = None) -> MoEPatchReport:
    """Patch compatible routed MoE blocks in-place.

    For non-MoE models this is a no-op with `patched_blocks == 0`.
    """

    patched = 0
    for module in model.modules():
        if _patch_one_block(module):
            patched += 1
    model._paper_losa_moe_expert_patch_count = patched
    model._paper_losa_moe_expert_patch_backend = "triton_packed_routed_moe"
    model._paper_losa_moe_expert_patch_family = family
    return MoEPatchReport(
        requested=True,
        patched_blocks=patched,
        backend="triton_packed_routed_moe",
    )

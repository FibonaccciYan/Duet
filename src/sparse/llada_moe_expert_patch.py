"""Runtime MoE expert patch for the LLaDA2.1 remote-code model.

The remote model dispatches routed tokens by synchronizing expert counts to
the CPU and invoking one Python MLP per expert. This patch packs each layer's
expert weights in-place and uses GPU-side routing plus Triton grouped MLP
kernels. The model files are never modified on disk.
"""

import types

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in minimal installs
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


_BLOCK_M = 16
_BLOCK_N = 128
_BLOCK_K = 128


if _TRITON_AVAILABLE:

    @triton.jit
    def _moe_gate_up_kernel(
        x_ptr,
        gate_weight_ptr,
        up_weight_ptr,
        gate_out_ptr,
        up_out_ptr,
        active_experts_ptr,
        offsets_ptr,
        counts_ptr,
        total_rows,
        tiles_m,
        hidden_size,
        intermediate_size,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        expert_slot = pid // tiles_m
        expert_id = tl.load(active_experts_ptr + expert_slot)
        tile_m = pid % tiles_m
        pid_n = tl.program_id(1)

        rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
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
            x_ptrs = x_ptr + row_ids[:, None] * hidden_size + ks[None, :]
            x_tile = tl.load(x_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)

            gate_ptrs = (
                gate_weight_ptr
                + expert_id * intermediate_size * hidden_size
                + cols[:, None] * hidden_size
                + ks[None, :]
            )
            up_ptrs = (
                up_weight_ptr
                + expert_id * intermediate_size * hidden_size
                + cols[:, None] * hidden_size
                + ks[None, :]
            )
            gate_tile = tl.load(
                gate_ptrs,
                mask=col_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            up_tile = tl.load(
                up_ptrs,
                mask=col_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            gate_acc += tl.dot(x_tile, tl.trans(gate_tile), out_dtype=tl.float32)
            up_acc += tl.dot(x_tile, tl.trans(up_tile), out_dtype=tl.float32)

        out_ptrs = gate_out_ptr + row_ids[:, None] * intermediate_size + cols[None, :]
        tl.store(out_ptrs, gate_acc, mask=row_mask[:, None] & col_mask[None, :])
        out_ptrs = up_out_ptr + row_ids[:, None] * intermediate_size + cols[None, :]
        tl.store(out_ptrs, up_acc, mask=row_mask[:, None] & col_mask[None, :])


    @triton.jit
    def _moe_down_kernel(
        gate_ptr,
        up_ptr,
        down_weight_ptr,
        out_ptr,
        active_experts_ptr,
        offsets_ptr,
        counts_ptr,
        total_rows,
        tiles_m,
        hidden_size,
        intermediate_size,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        expert_slot = pid // tiles_m
        expert_id = tl.load(active_experts_ptr + expert_slot)
        tile_m = pid % tiles_m
        pid_n = tl.program_id(1)

        rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
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
            gate_values = tl.load(
                gate_ptr + row_ids[:, None] * intermediate_size + ks[None, :],
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            up_values = tl.load(
                up_ptr + row_ids[:, None] * intermediate_size + ks[None, :],
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            activated = gate_values * tl.sigmoid(gate_values) * up_values
            weight_ptrs = (
                down_weight_ptr
                + expert_id * hidden_size * intermediate_size
                + cols[:, None] * intermediate_size
                + ks[None, :]
            )
            weights = tl.load(
                weight_ptrs,
                mask=col_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.dot(activated, tl.trans(weights), out_dtype=tl.float32)

        tl.store(
            out_ptr + row_ids[:, None] * hidden_size + cols[None, :],
            acc,
            mask=row_mask[:, None] & col_mask[None, :],
        )


def _packed_weights(block):
    experts = list(block.experts)
    if not experts:
        return None
    projections = []
    for name in ("gate_proj", "up_proj", "down_proj"):
        weights = [getattr(expert, name).weight for expert in experts]
        if any(weight is None for weight in weights):
            return None
        projections.append(
            torch.stack([weight.detach() for weight in weights], dim=0).contiguous()
        )
    return tuple(projections)


def _triton_moe_infer(self, x, topk_ids, topk_weight):
    topk_ids = topk_ids.reshape(-1, topk_ids.shape[-1])
    topk_weight = topk_weight.reshape_as(topk_ids)
    token_count, top_k = topk_ids.shape
    if token_count == 0:
        return x.new_empty((0, x.shape[-1]))

    gate_weight = self._llada_moe_gate_weight
    up_weight = self._llada_moe_up_weight
    down_weight = self._llada_moe_down_weight
    num_experts, intermediate_size, hidden_size = gate_weight.shape
    flat_ids = topk_ids.reshape(-1)
    counts = torch.bincount(flat_ids, minlength=num_experts)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    order = torch.argsort(flat_ids)
    sorted_tokens = x.index_select(0, order // top_k).contiguous()
    total_rows = sorted_tokens.shape[0]
    active_experts = torch.nonzero(counts, as_tuple=False).flatten()
    # One scalar sync keeps the launch grid compact instead of padding every
    # active expert to the full routed-token count.
    max_tokens = max(1, int(counts.max().item()))
    tiles_m = triton.cdiv(max_tokens, _BLOCK_M)

    gate_out = torch.empty(
        (total_rows, intermediate_size), dtype=x.dtype, device=x.device
    )
    up_out = torch.empty_like(gate_out)
    _moe_gate_up_kernel[
        (active_experts.numel() * tiles_m, triton.cdiv(intermediate_size, _BLOCK_N))
    ](
        sorted_tokens,
        gate_weight,
        up_weight,
        gate_out,
        up_out,
        active_experts,
        offsets,
        counts,
        total_rows,
        tiles_m,
        hidden_size,
        intermediate_size,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
    )

    routed_out = torch.empty(
        (total_rows, hidden_size), dtype=x.dtype, device=x.device
    )
    _moe_down_kernel[
        (active_experts.numel() * tiles_m, triton.cdiv(hidden_size, _BLOCK_N))
    ](
        gate_out,
        up_out,
        down_weight,
        routed_out,
        active_experts,
        offsets,
        counts,
        total_rows,
        tiles_m,
        hidden_size,
        intermediate_size,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
    )

    restored = torch.empty_like(routed_out)
    restored.index_copy_(0, order, routed_out)
    return (
        restored.view(token_count, top_k, hidden_size)
        .to(topk_weight.dtype)
        .mul_(topk_weight.unsqueeze(-1))
        .sum(dim=1)
        .to(x.dtype)
    )


def _pack_block(block):
    if getattr(block, "_llada_moe_expert_patched", False):
        return True
    if not _TRITON_AVAILABLE or not torch.cuda.is_available():
        return False
    if not hasattr(block, "experts") or not hasattr(block, "moe_infer"):
        return False

    weights = _packed_weights(block)
    if weights is None:
        return False
    if any(weight.device.type != "cuda" for weight in weights):
        return False
    gate_weight, up_weight, down_weight = weights
    if gate_weight.ndim != 3 or up_weight.shape != gate_weight.shape:
        return False
    if down_weight.ndim != 3 or down_weight.shape[0] != gate_weight.shape[0]:
        return False

    block.register_buffer("_llada_moe_gate_weight", gate_weight, persistent=False)
    block.register_buffer("_llada_moe_up_weight", up_weight, persistent=False)
    block.register_buffer("_llada_moe_down_weight", down_weight, persistent=False)

    # The packed buffers replace the per-expert parameters in memory. The
    # patched inference method is installed immediately after this point.
    for expert in block.experts:
        expert.gate_proj.weight = None
        expert.up_proj.weight = None
        expert.down_proj.weight = None
    block.moe_infer = types.MethodType(_triton_moe_infer, block)
    block._llada_moe_expert_patched = True
    return True


def patch_moe_experts(model):
    """Patch all compatible routed MoE blocks in ``model`` in-place."""
    patched = 0
    for module in model.modules():
        if _pack_block(module):
            patched += 1
    model._llada_moe_expert_patch_count = patched
    return patched

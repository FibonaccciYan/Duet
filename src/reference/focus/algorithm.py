"""Framework-independent FOCUS importance and row-retention algorithm.

The implementation follows the FOCUS SGLang/LMDeploy model path: attention
importance is measured on unresolved mask rows in layers zero and one; the
positive importance delta determines which masks continue through the suffix
layers. Already materialized tokens are always retained.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass
class FocusDecodeState:
    """Small amount of state carried between denoising iterations."""

    decoded_tokens: int = 0
    measured_steps: int = 0
    block_progress: int = -1

    @property
    def average_decoded_tokens(self) -> float:
        if self.measured_steps == 0:
            return 1.0
        return max(1.0, self.decoded_tokens / self.measured_steps)

    def record_transfer(self, count: int) -> None:
        self.decoded_tokens += int(count)
        self.measured_steps += 1


def attention_importance(
    query: torch.Tensor,
    key: torch.Tensor,
    mask_positions: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Return FOCUS importance for each unresolved mask position.

    ``query`` is ``[batch, query_heads, block, dim]`` and ``key`` is
    ``[batch, kv_heads, block, dim]``. The portable runtime currently mirrors
    the project's batch-size-one patch contract.
    """

    if query.ndim != 4 or key.ndim != 4 or query.shape[0] != 1 or key.shape[0] != 1:
        raise ValueError("FOCUS importance expects batch-size-one rank-4 Q/K tensors")
    if query.shape[-1] != key.shape[-1] or query.shape[1] % key.shape[1]:
        raise ValueError("incompatible FOCUS query/key head layout")
    if mask_positions.ndim != 1:
        raise ValueError("mask_positions must be one-dimensional")
    if mask_positions.numel() == 0:
        return query.new_empty((0,))

    q = query[0].transpose(0, 1).index_select(0, mask_positions)
    k = key[0].transpose(0, 1).index_select(0, mask_positions)
    groups = query.shape[1] // key.shape[1]
    k = k.repeat_interleave(groups, dim=1)
    scores = torch.einsum("qhd,khd->qhk", q.float(), k.float())
    scores.mul_(float(scale if scale is not None else query.shape[-1] ** -0.5))

    neg_inf = torch.full_like(scores[..., :1], float("-inf"))
    previous = torch.cat((neg_inf, scores[..., :-1]), dim=-1)
    following = torch.cat((scores[..., 1:], neg_inf), dim=-1)
    pooled = torch.maximum(scores, previous)
    pooled = torch.maximum(pooled, following)
    return torch.softmax(pooled, dim=-1).sum(dim=(0, 1)).to(query.dtype)


def _select_delta(delta: torch.Tensor, target: int) -> torch.Tensor:
    if target >= delta.numel():
        return torch.ones_like(delta, dtype=torch.bool)
    order = torch.argsort(delta, descending=True)
    top = torch.zeros_like(delta, dtype=torch.bool)
    top[order[:target]] = True
    threshold = delta.float().mean() + delta.float().std(unbiased=False)
    above = delta.float() >= threshold
    return above if int(above.sum().item()) >= target else top


def select_retained_positions(
    block_tokens: torch.Tensor,
    mask_id: int,
    layer0_importance: torch.Tensor,
    layer1_importance: torch.Tensor,
    *,
    alpha: float,
    average_decoded_tokens: float,
    block_progress: int = -1,
) -> torch.Tensor:
    """Select full-block positions retained after FOCUS layer one."""

    if block_tokens.ndim != 1:
        raise ValueError("block_tokens must be one-dimensional")
    if alpha < 1.0:
        raise ValueError("FOCUS alpha must be at least 1")
    mask_positions = torch.where(block_tokens == int(mask_id))[0]
    if mask_positions.numel() == 0:
        return torch.arange(block_tokens.numel(), device=block_tokens.device)
    if layer0_importance.shape != mask_positions.shape or layer1_importance.shape != mask_positions.shape:
        raise ValueError("FOCUS importance length must match unresolved masks")

    target = min(
        int(mask_positions.numel()),
        max(1, math.ceil(max(1.0, float(average_decoded_tokens)) * float(alpha))),
    )
    retain_mask = _select_delta(layer1_importance.float() - layer0_importance.float(), target)

    # Preserve the two ordering rules from the engine implementation. A kept
    # row pulls in its immediate left neighbour, and rows before the rightmost
    # kept position cannot be skipped until progress has moved beyond them.
    if mask_positions.numel() > 1:
        adjacent = (mask_positions[1:] - mask_positions[:-1]) == 1
        retain_mask[:-1] |= adjacent & retain_mask[1:] & ~retain_mask[:-1]
    if not bool(retain_mask.any().item()):
        retain_mask[0] = True
    rightmost = int(mask_positions[retain_mask].max().item())
    retain_mask |= (mask_positions < rightmost) & (mask_positions > int(block_progress))

    flags = block_tokens != int(mask_id)
    flags = flags.clone()
    flags[mask_positions[retain_mask]] = True
    return torch.where(flags)[0]


__all__ = [
    "FocusDecodeState",
    "attention_importance",
    "select_retained_positions",
]

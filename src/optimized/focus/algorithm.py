"""optimized FOCUS retains the reference importance and row-selection rules.

Intermediate predicates remain on device; argsort and threshold arithmetic
are unchanged. Final compaction still has a data-dependent output shape.
"""

from src.reference.focus.algorithm import (
    FocusDecodeState,
    attention_importance,
)
import math
import torch


def select_retained_positions(
    block_tokens, mask_id, layer0_importance, layer1_importance, *,
    alpha, average_decoded_tokens, block_progress=-1,
):
    """Same argsort, threshold and constraints, without intermediate host reads."""
    if block_tokens.ndim != 1:
        raise ValueError("block_tokens must be one-dimensional")
    if alpha < 1.0:
        raise ValueError("FOCUS alpha must be at least 1")
    mask_positions = torch.where(block_tokens == int(mask_id))[0]
    count = mask_positions.numel()
    if count == 0:
        return torch.arange(block_tokens.numel(), device=block_tokens.device)
    if layer0_importance.shape != mask_positions.shape or layer1_importance.shape != mask_positions.shape:
        raise ValueError("FOCUS importance length must match unresolved masks")
    target = min(count, max(1, math.ceil(max(1.0, float(average_decoded_tokens))*float(alpha))))
    delta = layer1_importance.float()-layer0_importance.float()
    if target >= count:
        retained = torch.ones_like(delta,dtype=torch.bool)
    else:
        order = torch.argsort(delta,descending=True)
        top = torch.zeros_like(delta,dtype=torch.bool)
        top[order[:target]] = True
        threshold = delta.float().mean()+delta.float().std(unbiased=False)
        above = delta.float() >= threshold
        retained = torch.where(above.sum() >= target,above,top)
    if count > 1:
        adjacent = (mask_positions[1:]-mask_positions[:-1]) == 1
        retained[:-1] |= adjacent & retained[1:] & ~retained[:-1]
    # Retain the defensive fallback without synchronizing its predicate.
    fallback = ~retained.any()
    retained[0] |= fallback
    rightmost = torch.where(retained,mask_positions,-1).max()
    retained |= (mask_positions<rightmost) & (mask_positions>int(block_progress))
    flags = (block_tokens != int(mask_id)).clone()
    flags[mask_positions] = retained
    return torch.where(flags)[0]

__all__ = [
    "FocusDecodeState",
    "attention_importance",
    "select_retained_positions",
]

"""Small helpers retained for paper motivation diagnostics."""

from __future__ import annotations

import torch


def selector_mask(mask, scores, minimum, threshold):
    """Select the same confidence-constrained transfer set as the runtime."""
    confidence = torch.where(mask, scores, -torch.inf)
    high_confidence = (confidence[0] > threshold) & mask[0]
    transfer = torch.zeros_like(mask)
    minimum = min(int(minimum), int(mask.sum().item()))
    if int(high_confidence.sum().item()) >= minimum:
        transfer[0] = high_confidence
    elif minimum:
        indices = torch.topk(confidence[0], k=minimum).indices
        transfer[0, indices] = True
    return transfer


def overlap_metrics(predicted, actual):
    """Return set-overlap diagnostics for predicted and final transfers."""
    overlap = int((predicted & actual).sum().item())
    predicted_count = int(predicted.sum().item())
    actual_count = int(actual.sum().item())
    union = predicted_count + actual_count - overlap
    return {
        "predicted_count": predicted_count,
        "final_count": actual_count,
        "overlap_count": overlap,
        "precision": overlap / max(predicted_count, 1),
        "recall": overlap / max(actual_count, 1),
        "jaccard": overlap / max(union, 1),
    }

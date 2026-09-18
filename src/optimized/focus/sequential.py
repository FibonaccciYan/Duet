"""SDAR sequential adaptation; reference confidence-based FOCUS is unchanged."""
import torch
from src.reference.losa.generation import sample_with_confidence


def required_positions(active_mask, count):
    """The leftmost unresolved absolute block positions in this step's budget."""
    return torch.where(active_mask[0])[0][:max(0, int(count))]


def retain_required(selected, required):
    """Preserve FOCUS's choices and add only sequentially required queries."""
    return torch.unique(torch.cat((selected, required)), sorted=True)


def transfer_sequential(block_tokens, active_mask, logits, positions, *,
                        count, temperature, top_p, top_k, strategy, threshold):
    if strategy != "sequential":
        raise ValueError("sequential transfer called with another strategy")
    absolute = required_positions(active_mask, count)
    transfer = torch.zeros_like(active_mask)
    if absolute.numel() == 0:
        return transfer, 0
    # Do not silently decode a later retained mask when an earlier one is absent.
    matches = absolute[:, None] == positions[None, :]
    if not bool((matches.sum(dim=1) == 1).all().item()):
        raise RuntimeError("FOCUS omitted a required sequential query position")
    rows = matches.to(torch.int64).argmax(dim=1)
    predicted, _ = sample_with_confidence(
        logits, temperature=temperature, top_p=top_p, top_k=top_k)
    transfer[0, absolute] = True
    block_tokens[0, absolute] = predicted[0].index_select(0, rows)
    return transfer, int(absolute.numel())

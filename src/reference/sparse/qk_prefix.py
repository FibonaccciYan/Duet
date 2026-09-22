"""QK oracle: negative unscaled dot, FP32 products and adjacent-pair tree sums.

Ties are resolved by smaller candidate-local token index at EVERY ranking.
Do not reuse this arithmetic or tie policy in the legacy Raw L1 selector.
"""
import math
import torch

SCORE_DEFINITION = "negative_unscaled_dot_fp32_pairwise_tree"


def validate(query, key):
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("QK inputs must be [1,heads,tokens,dim]")
    if query.shape[0] != 1 or key.shape[0] != 1:
        raise ValueError("QK selector requires batch_size=1")
    if key.shape[1] <= 0 or query.shape[1] <= 0 or query.shape[1] % key.shape[1]:
        raise ValueError("QK selector requires valid GQA head mapping")
    if query.shape[-1] <= 0 or query.shape[-1] != key.shape[-1]:
        raise ValueError("QK head dimensions must match and be positive")
    if query.shape[2] <= 0:
        raise ValueError("QK requires at least one query per head")
    if query.device != key.device:
        raise ValueError("QK tensors must be on the same device")
    if not query.is_floating_point() or not key.is_floating_point():
        raise ValueError("QK requires floating point tensors")


def record(stats, length, budget, local_budget, union_size, selected_size, strict, bypassed=False):
    if stats is not None:
        stats.update(candidate_length=length, budget=budget, local_budget=local_budget,
                     union_size=union_size, selected_size=selected_size,
                     strict_budget=bool(strict), bypassed=bypassed, selector="qk",
                     score_definition=SCORE_DEFINITION)


def distances(query_rows, keys):
    """Rows [R,D], keys [N,D]; no GEMM/TF32, exact same tree as QK Triton."""
    products = query_rows.float()[:, None, :] * keys.float()[None, :, :]
    width = 1 << (products.shape[-1] - 1).bit_length()
    if width != products.shape[-1]:
        products = torch.nn.functional.pad(products, (0, width-products.shape[-1]))
    while width > 1:
        products = products.reshape(*products.shape[:-1], width // 2, 2)
        products = products[..., 0] + products[..., 1]
        width //= 2
    return -products.squeeze(-1)


def rank(scores, indices, count):
    """Lexicographic (distance, token index), works for rows or vectors."""
    by_index = torch.argsort(indices, dim=-1, stable=True)
    indices = indices.gather(-1, by_index)
    scores = scores.gather(-1, by_index)
    order = torch.argsort(scores, dim=-1, stable=True)[..., :count]
    return scores.gather(-1, order), indices.gather(-1, order)


def finish_union(candidates, token_scores, budget, strict_budget, stats, local_budget):
    selected = torch.unique(candidates.flatten(), sorted=True)
    union_size = selected.numel()
    if union_size < budget:
        # Rank only unselected tokens; avoid an inf-mask tie returning duplicates.
        available = torch.ones(token_scores.numel(), dtype=torch.bool, device=selected.device)
        available[selected] = False
        remaining = torch.arange(token_scores.numel(), device=selected.device)[available]
        _, fill = rank(token_scores[remaining], remaining, budget-union_size)
        selected = torch.cat((selected, fill))
    elif strict_budget and union_size > budget:
        _, selected = rank(token_scores[selected], selected, budget)
    selected = selected.sort().values
    record(stats, token_scores.numel(), budget, local_budget, union_size,
           selected.numel(), strict_budget)
    return selected


def prefix_indices(query, key, token_budget, chunk_size=256, bucket_thresholds=None,
                   strict_budget=False, selection_stats=None):
    validate(query, key)
    length = key.shape[2]
    budget = max(0, min(int(token_budget), length))
    if budget == 0 or budget == length:
        record(selection_stats, length, budget, 0, budget, budget, strict_budget, True)
        return torch.arange(budget, dtype=torch.long, device=key.device)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    heads, qlen, dim = query.shape[1:]
    group = heads // key.shape[1]
    local_budget = max(1, math.ceil(budget/(heads*qlen)))
    global_scores = torch.full((length,), float("inf"), dtype=torch.float32, device=key.device)
    all_candidates = []
    # Bound intermediate products even on full 32K candidates.
    for kv_head in range(key.shape[1]):
        queries = query[0, kv_head*group:(kv_head+1)*group].reshape(-1, dim)
        for row_start in range(0, queries.shape[0], 32):
            q = queries[row_start:row_start+32]
            best_scores = best_indices = None
            for start in range(0, length, chunk_size):
                end = min(start+chunk_size, length)
                scores = distances(q, key[0, kv_head, start:end])
                global_scores[start:end] = torch.minimum(global_scores[start:end], scores.amin(0))
                indices = torch.arange(start, end, device=key.device).expand(scores.shape[0], -1)
                if local_budget == 1:
                    values, positions = scores.min(dim=-1, keepdim=True)
                    positions = positions + start
                    if best_scores is not None:
                        better = values < best_scores  # earlier chunks win exact ties
                        values = torch.where(better, values, best_scores)
                        positions = torch.where(better, positions, best_indices)
                    best_scores, best_indices = values, positions
                else:
                    if best_scores is not None:
                        scores = torch.cat((best_scores, scores), -1)
                        indices = torch.cat((best_indices, indices), -1)
                    best_scores, best_indices = rank(scores, indices, min(local_budget, scores.shape[-1]))
            all_candidates.append(best_indices)
    return finish_union(torch.cat(all_candidates, 0), global_scores, budget,
                        strict_budget, selection_stats, local_budget)

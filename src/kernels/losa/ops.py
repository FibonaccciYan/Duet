"""Compatibility exports for the integrated paper-LoSA operators."""

from __future__ import annotations

from src.reference.losa.operators import (
    GQAMode,
    KVHeadUnion,
    LoSAPrefixState as LoSAState,
    LoSAResult as AttentionStepResult,
    QuestPageMetadata,
    QuestSelection,
    build_page_metadata,
    build_union_pages,
    dense_attention,
    locality_scores,
    merge_attention_states,
    quest_page_scores,
    select_active_rows,
    select_pages,
    sparse_prefix_attention_on_union,
    losa_attention_step as _losa_attention_step,
)


def losa_attention_step(*args, **kwargs):
    return _losa_attention_step(*args, **kwargs)


def adapted_quest_attention_step(*args, **kwargs):
    """QUEST baseline that runs the same union kernel on all query rows."""

    if "active_count" not in kwargs and len(args) < 8:
        # Mirror the paper-adapted baseline by treating all rows as active.
        query = args[0] if args else kwargs["query"]
        kwargs["active_count"] = query.shape[0]
    return _losa_attention_step(*args, **kwargs)

"""Public model-family resolution and sparse patch dispatcher."""

from .llada_patch import patch_llada_model, patch_moe_experts
from .sdar_patch import patch_sdar_model


MODEL_TYPES = {
    "llada": {"llada2_moe"},
    "sdar": {"sdar"},
}


def resolve_model_family(model, model_name="auto"):
    actual_type = getattr(model.config, "model_type", None)
    if model_name == "auto":
        for family, model_types in MODEL_TYPES.items():
            if actual_type in model_types:
                return family
        raise ValueError(f"Unsupported model_type: {actual_type!r}")
    if model_name not in MODEL_TYPES:
        raise ValueError(f"Unsupported model name: {model_name!r}")
    if actual_type not in MODEL_TYPES[model_name]:
        raise ValueError(
            f"Requested {model_name!r}, but checkpoint model_type is {actual_type!r}"
        )
    return model_name


def patch_model(
    model,
    model_name="auto",
    ratio=0.5,
    top_k=64,
    selection_interval=None,
    query_dense_threshold=None,
    refresh_step=-1,
    selection_layer=5,
    deep_only_transfer=False,
    query_sparse=False,
    prefix_sparse=False,
    prefix_token_budget=256,
    prefix_chunk_size=None,
    losa=False,
    losa_active_topk=5,
    losa_score_mode="query",
    losa_key_samples=32,
    query_losa_union=False,
    moe_expert_patch=True,
):
    """Enable requested sparse features through the matching model patch."""
    family = resolve_model_family(model, model_name)
    prefix_chunk_size = (
        1024 if family == "sdar" else 256
    ) if prefix_chunk_size is None else prefix_chunk_size
    if family == "llada":
        patch_llada_model(
            model,
            ratio=ratio,
            top_k=top_k,
            selection_interval=selection_interval or 4,
            query_dense_threshold=(
                4 if query_dense_threshold is None else query_dense_threshold
            ),
            selection_layer=selection_layer,
            query_sparse=query_sparse,
            prefix_sparse=prefix_sparse,
            prefix_token_budget=prefix_token_budget,
            prefix_chunk_size=prefix_chunk_size,
            losa=losa,
            losa_active_topk=losa_active_topk,
            losa_score_mode=losa_score_mode,
            losa_key_samples=losa_key_samples,
            query_losa_union=query_losa_union,
        )

        if moe_expert_patch:
            patch_moe_experts(model)
    else:
        patch_sdar_model(
            model,
            ratio=ratio,
            top_k=top_k,
            selection_interval=selection_interval or 1,
            query_dense_threshold=(
                0 if query_dense_threshold is None else query_dense_threshold
            ),
            refresh_step=refresh_step,
            selection_layer=selection_layer,
            deep_only_transfer=deep_only_transfer,
            query_sparse=query_sparse,
            prefix_sparse=prefix_sparse,
            prefix_token_budget=prefix_token_budget,
            prefix_chunk_size=prefix_chunk_size,
            losa=losa,
            losa_active_topk=losa_active_topk,
            losa_score_mode=losa_score_mode,
            losa_key_samples=losa_key_samples,
        )

    model._sparse_patch_family = family
    return model

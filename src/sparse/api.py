"""Public model-family resolution and sparse patch dispatcher."""

from .llada_patch import patch_llada_model, patch_moe_experts
from .sdar_patch import patch_sdar_model
from .config import LLaDASparseConfig, SDARSparseConfig, apply_overrides, save_to_model_config


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
    ratio=None,
    top_k=None,
    selection_interval=None,
    query_dense_threshold=None,
    query_min_prefix_length=None,
    refresh_step=None,
    selection_layer=None,
    deep_only_transfer=None,
    query_sparse=None,
    prefix_sparse=None,
    prefix_min_prefix_length=None,
    prefix_token_budget=None,
    prefix_chunk_size=None,
    prefix_share_layer_pairs=None,
    prefix_rescreen_full_kv=None,
    losa=None,
    losa_active_topk=None,
    losa_score_mode=None,
    losa_key_samples=None,
    query_losa_union=None,
    moe_expert_patch=None,
    sparse_config=None,
):
    """Enable requested sparse features through the matching model patch."""
    family = resolve_model_family(model, model_name)
    config_type = LLaDASparseConfig if family == "llada" else SDARSparseConfig
    if sparse_config is None:
        sparse_config = config_type()
    elif not isinstance(sparse_config, config_type):
        raise TypeError(f"sparse_config must be {config_type.__name__}")
    sparse_config = apply_overrides(
        sparse_config,
        {
            "ratio": ratio,
            "top_k": top_k,
            "selection_interval": selection_interval,
            "query_dense_threshold": query_dense_threshold,
            "query_min_prefix_length": query_min_prefix_length,
            "refresh_step": refresh_step,
            "selection_layer": selection_layer,
            "deep_only_transfer": deep_only_transfer,
            "query_sparse": query_sparse,
            "prefix_sparse": prefix_sparse,
            "prefix_min_prefix_length": prefix_min_prefix_length,
            "prefix_token_budget": prefix_token_budget,
            "prefix_chunk_size": prefix_chunk_size,
            "prefix_share_layer_pairs": prefix_share_layer_pairs,
            "prefix_rescreen_full_kv": prefix_rescreen_full_kv,
            "losa": losa,
            "losa_active_topk": losa_active_topk,
            "losa_score_mode": losa_score_mode,
            "losa_key_samples": losa_key_samples,
            "query_losa_union": query_losa_union,
            "moe_expert_patch": moe_expert_patch,
        },
    )
    save_to_model_config(model, sparse_config)
    values = sparse_config.__dict__
    if family == "llada":
        patch_llada_model(
            model,
            **{
                key: value
                for key, value in values.items()
                if key != "moe_expert_patch"
            },
        )

        if values["moe_expert_patch"]:
            patch_moe_experts(model)
    else:
        patch_sdar_model(
            model,
            **values,
        )

    model._sparse_patch_family = family
    return model

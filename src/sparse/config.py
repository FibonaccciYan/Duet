"""Model-family-specific sparse generation configuration."""

from dataclasses import asdict, dataclass, fields


@dataclass
class LLaDASparseConfig:
    ratio: float = 0.7
    top_k: int = 64
    selection_interval: int = 4
    query_dense_threshold: int = 4
    query_min_prefix_length: int = 0
    selection_layer: int = 1
    query_sparse: bool = True
    prefix_sparse: bool = True
    prefix_min_prefix_length: int = 0
    prefix_token_budget: int = 256
    prefix_chunk_size: int = 1024
    prefix_rescreen_full_kv: bool = False
    losa: bool = False
    losa_active_topk: int = 5
    losa_score_mode: str = "query"
    losa_key_samples: int = 32
    query_losa_union: bool = False
    moe_expert_patch: bool = True


@dataclass
class SDARSparseConfig:
    ratio: float = 0.5
    top_k: int = 64
    selection_interval: int = 1
    query_dense_threshold: int = 4
    refresh_step: int = -1
    selection_layer: int = 5
    deep_only_transfer: bool = False
    query_sparse: bool = True
    prefix_sparse: bool = False
    prefix_min_prefix_length: int = 0
    prefix_token_budget: int = 256
    prefix_chunk_size: int = 1024
    prefix_share_layer_pairs: bool = False
    prefix_rescreen_full_kv: bool = False
    losa: bool = False
    losa_active_topk: int = 5
    losa_score_mode: str = "query"
    losa_key_samples: int = 32


def apply_overrides(config, overrides):
    values = asdict(config)
    valid = {field.name for field in fields(config)}
    for key, value in overrides.items():
        if key in valid and value is not None:
            values[key] = value
    return type(config)(**values)


def save_to_model_config(model, config):
    """Persist a JSON-serializable, family-specific snapshot on the model."""
    name = (
        "llada_sparse_config"
        if isinstance(config, LLaDASparseConfig)
        else "sdar_sparse_config"
    )
    setattr(model.config, name, asdict(config))

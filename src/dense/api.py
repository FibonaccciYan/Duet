from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from src.losa.generation import (
    load_model_and_tokenizer,
    set_seed,
)
try:  # pragma: no cover - optional packed MoE backend
    from src.losa.moe_patch import patch_moe_experts
except Exception:  # pragma: no cover
    patch_moe_experts = None


DEFAULT_MODEL_PATHS = {
    "llada": "/data0/ysy/models/LLaDA2.1-mini",
    "sdar": "/data0/ysy/models/SDAR-8B-Chat-b32",
}


def _resolve_family(model: Any, model_name: str) -> str:
    actual = getattr(getattr(model, "config", None), "model_type", None)
    inferred = {"llada2_moe": "llada", "sdar": "sdar"}.get(actual)
    if model_name == "auto":
        if inferred is None:
            raise ValueError(f"Unsupported model_type: {actual!r}")
        return inferred
    if model_name not in {"llada", "sdar"}:
        raise ValueError(f"Unsupported model name: {model_name!r}")
    if inferred != model_name:
        raise ValueError(
            f"Requested {model_name!r}, but checkpoint model_type is {actual!r}"
        )
    return model_name


def patch_model(
    model: Any,
    model_name: str = "auto",
    *,
    moe_expert_patch: bool = True,
    **_: Any,
):
    """Install the dense baseline using the same ``patch_model`` entry point as sparse.

    The trusted sparse runtime supplies the cache-efficient block driver for
    both families with every approximation disabled.  This preserves dense
    attention while avoiding the quadratic attention mask and full-vocabulary
    prompt logits that make the checkpoint's monolithic 32K path run OOM.
    """
    family = _resolve_family(model, model_name)
    from src.sparse import patch_model as patch_sparse_model

    patch_sparse_model(
        model,
        model_name=family,
        query_sparse=False,
        prefix_sparse=False,
        losa=False,
        moe_expert_patch=bool(moe_expert_patch),
    )
    model._dense_patch_family = family
    return model


@dataclass
class DenseRuntime:
    family: str
    model_path: str | None = None
    dtype: str | None = None
    attn_implementation: str = "sdpa"
    moe_expert_patch: bool = True
    model: Any | None = field(init=False, default=None)
    tokenizer: Any | None = field(init=False, default=None)
    moe_patch_report: Any | None = field(init=False, default=None)

    def load(self):
        model_path = self.model_path or DEFAULT_MODEL_PATHS[self.family]
        dtype_name = self.dtype or ("bfloat16" if self.family == "llada" else "float16")
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.family,
            model_path=model_path,
            dtype=dtype_name,
            attn_implementation=self.attn_implementation,
        )
        patch_model(
            self.model,
            model_name=self.family,
            moe_expert_patch=self.moe_expert_patch,
        )
        if self.moe_expert_patch:
            # The sparse compatibility patch installs the packed routed-MoE
            # backend and sets _llada_moe_expert_patch_count.  Do not run the
            # LoSA patcher again: it would see already-packed weights and make
            # the benchmark report a false patched_blocks=0.
            count = int(getattr(self.model, "_llada_moe_expert_patch_count", 0))
            self.moe_patch_report = {
                "requested": True,
                "patched_blocks": count,
                "backend": "triton_packed_routed_moe",
            }
        return self.model, self.tokenizer

    def generate(self, inputs: torch.Tensor, **kwargs):
        if self.model is None or self.tokenizer is None:
            self.load()
        assert self.model is not None
        return type("DenseGeneration", (), {
            "tokens": self.model.generate(inputs=inputs, **kwargs),
            "trace": [],
        })()

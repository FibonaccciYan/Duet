"""Select checkpoint-compatible model implementations from src/model."""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelCode:
    family: str
    variant: str
    config_cls: Any
    model_cls: Any


def _load_code(module_name: str, config_name: str, model_name: str) -> tuple[Any, Any]:
    configuration = importlib.import_module(module_name + ".configuration")
    modeling = importlib.import_module(module_name + ".modeling")
    return getattr(configuration, config_name), getattr(modeling, model_name)


def _llada_variant(model_path: str | Path, override: str | None) -> str:
    if override:
        value = str(override).strip().lower()
        if value in {"2.0", "20", "llada2.0", "llada2_0"}:
            return "2.0"
        if value in {"2.1", "21", "llada2.1", "llada2_1"}:
            return "2.1"
        raise ValueError(f"unknown LLaDA variant: {override!r}")

    text = str(model_path).lower()
    if "llada2.0" in text or "llada2_0" in text or "llada-2.0" in text:
        return "2.0"
    if "llada2.1" in text or "llada2_1" in text or "llada-2.1" in text:
        return "2.1"
    # Existing project defaults use LLaDA 2.1.
    return "2.1"


def _model_type(model_path: str | Path) -> str:
    with (Path(model_path) / "config.json").open(encoding="utf-8") as handle:
        return str(json.load(handle).get("model_type", ""))


def resolve_model_code(
    family: str,
    model_path: str | Path,
    variant: str | None = None,
) -> ModelCode:
    """Return project-managed config/model classes for a supported checkpoint."""
    model_type = _model_type(model_path)
    if family == "llada" or model_type == "llada2_moe":
        resolved = _llada_variant(model_path, variant or os.getenv("SPARSEDLM_LLADA_VARIANT"))
        module = "src.model.llada2_0" if resolved == "2.0" else "src.model.llada2_1"
        config_cls, model_cls = _load_code(module, "LLaDA2MoeConfig", "LLaDA2MoeModelLM")
        return ModelCode("llada", resolved, config_cls, model_cls)

    if family == "sdar" or model_type == "sdar":
        config_cls, model_cls = _load_code("src.model.sdar", "SDARConfig", "SDARForCausalLM")
        return ModelCode("sdar", "8b-chat-b32", config_cls, model_cls)

    raise ValueError(f"unsupported model family={family!r}, model_type={model_type!r}")

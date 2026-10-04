"""Shared, deployment-independent checkpoint path defaults."""

from __future__ import annotations

import os


MODEL_PATH_ENV_VARS = {
    "llada": "LLADA_MODEL_PATH",
    "sdar": "SDAR_MODEL_PATH",
}


def _configured_model_path(env_var: str) -> str | None:
    value = os.environ.get(env_var)
    if value is None:
        return None
    value = value.strip()
    return value or None


def configured_model_paths() -> dict[str, str | None]:
    """Return current family paths, treating empty environment values as unset."""
    return {
        family: _configured_model_path(env_var)
        for family, env_var in MODEL_PATH_ENV_VARS.items()
    }


def resolve_model_path(model_path: str | None, family: str) -> str:
    """Resolve an explicit path or the family-specific environment default.

    There is intentionally no machine-specific fallback.  An explicitly empty
    ``model_path`` is an error; an empty or unset environment variable is not
    treated as a valid checkpoint path.
    """
    if model_path is not None:
        if str(model_path).strip():
            return model_path
        env_var = MODEL_PATH_ENV_VARS.get(family)
        if env_var is None:
            raise ValueError(f"Unsupported model family: {family!r}")
        raise ValueError(
            "model_path was empty; pass a non-empty checkpoint path or "
            f"set {env_var} and leave model_path unset."
        )

    env_var = MODEL_PATH_ENV_VARS.get(family)
    configured_path = _configured_model_path(env_var) if env_var else None
    if configured_path is not None:
        return configured_path

    if env_var is None:
        raise ValueError(f"Unsupported model family: {family!r}")
    supplied = "empty model_path" if model_path is not None else "model_path was not supplied"
    raise ValueError(
        f"{supplied}; pass a non-empty checkpoint path or set {env_var} to one."
    )


# Runtime modules re-export this mapping for API compatibility.  A missing
# path is represented by None rather than an empty string.
DEFAULT_MODEL_PATHS = configured_model_paths()

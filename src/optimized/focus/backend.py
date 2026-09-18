"""optimized FOCUS backend defaults; explicit overrides remain supported."""
def resolve_attention_backend(family, backend="auto"):
    if family not in ("llada", "sdar"):
        raise ValueError(f"unsupported family: {family}")
    if backend == "auto":
        return "flash" if family == "llada" else "sdpa"
    if backend not in ("flash", "sdpa"):
        raise ValueError(f"unsupported backend: {backend}")
    return backend

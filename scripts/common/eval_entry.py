"""Original lm-eval CLI with a narrowly scoped cached-feature compatibility shim."""


def normalize_primitive_lists(value):
    if isinstance(value, list):
        return [normalize_primitive_lists(v) for v in value]
    if not isinstance(value, dict):
        return value
    out = {k: normalize_primitive_lists(v) for k, v in value.items()}
    if out.get("_type") == "List":
        # Sequence(dict) transposes struct fields on older datasets, so it is
        # NOT a general replacement for List. Only scalar values are equivalent.
        if out.get("feature", {}).get("_type") != "Value":
            raise ValueError("This datasets version cannot safely read non-primitive List features")
        out["_type"] = "Sequence"
    return out


def install_cache_compat():
    import datasets.features.features as features
    if hasattr(features, "List"):
        return False
    original = features.generate_from_dict
    if getattr(original, "_unified_primitive_lists", False):
        return True
    def compatible(obj):
        return original(normalize_primitive_lists(obj))
    compatible._unified_primitive_lists = True
    features.generate_from_dict = compatible
    return True


def main():
    if install_cache_compat():
        print("UNIFIED_CACHE_COMPAT primitive List -> Sequence metadata; records unchanged", flush=True)
    from lm_eval.__main__ import cli_evaluate
    cli_evaluate()


if __name__ == "__main__":
    main()

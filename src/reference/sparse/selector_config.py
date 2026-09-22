"""Selector names and auditable arithmetic definitions; no runtime state."""
SCORE_DEFINITIONS = {
    "raw_l1": "raw_l1_legacy",
    "qk": "negative_unscaled_dot_fp32_pairwise_tree",
    "qk_tc": "negative_unscaled_dot_tensorcore_fp32_accum",
    # Existing adapter options retain their original transformed-L1 behavior.
    "adamas": "hadamard_bucketized_l1_legacy",
    "hadamard_qk": "hadamard_l1_legacy",
}


def validate_selector(value):
    if value not in SCORE_DEFINITIONS:
        raise ValueError(f"Unsupported prefix selector {value!r}; expected {tuple(SCORE_DEFINITIONS)}")
    return value

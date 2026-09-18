"""Load exact tokenizer-specific IDs; never silently retokenize the prompt."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_samples(family, length, root=ROOT):
    if family not in ("sdar", "llada21") or length not in (8192, 16384, 32768):
        raise ValueError("family must be sdar/llada21 and length 8192/16384/32768")
    with (Path(root) / family / f"{length}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            assert len(item["input_ids"]) == length
            yield item


def reserve_generation_window(item, gen_length, max_window=32768):
    if not isinstance(gen_length, int) or gen_length < 0 or gen_length >= max_window:
        raise ValueError("gen_length must be an integer in [0, max_window)")
    limit = max_window - gen_length
    ids = item["input_ids"]
    if len(ids) <= limit:
        return dict(item)
    head = limit // 2
    tail = limit - head
    return dict(sample_index=item["sample_index"], source_id=item["source_id"],
                tokenizer=item["tokenizer"], input_ids=ids[:head] + ids[-tail:],
                target_tokens=limit, head_tokens=head, tail_tokens=tail,
                derived_from_target_tokens=item["target_tokens"],
                generation_budget=gen_length, max_window=max_window)

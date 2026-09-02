"""Collect and analyze LLaDA Query-sparse/LoSA position scores."""

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sparse import patch_model, resolve_model_family


DEFAULT_PROMPTS = (
    "Write a Python function that sorts a list of integers.",
    "Write a Python function that checks whether a string is a palindrome.",
    "Write a Python function that merges overlapping intervals.",
    "Write a Python function that returns the prime factors of an integer.",
)


def _tolist(tensor):
    return tensor.detach().cpu().tolist()


class QueryLosaCollector:
    """Duck-typed runtime collector consumed by the optional patch hooks."""

    def __init__(self):
        self.records = []
        self.sample = None
        self.current = None

    def start_sample(self, sample):
        if self.current is not None:
            raise RuntimeError("Cannot change sample during an active denoising step")
        self.sample = int(sample)

    def begin_step(self, block, step, tokens, mask_id):
        if self.current is not None:
            raise RuntimeError("Previous denoising step was not finalized")
        mask = tokens == mask_id
        self.current = {
            "sample": self.sample,
            "block": int(block),
            "step": int(step),
            "mask_positions": _tolist(torch.where(mask)[0]),
            "decoded_positions": _tolist(torch.where(~mask)[0]),
            "query": None,
            "losa": [],
            "transfer_positions": [],
        }

    def record_query(self, score_data, selected_positions, mode):
        if self.current is None:
            return
        selected = [] if selected_positions is None else _tolist(selected_positions)
        mask_positions = set(self.current["mask_positions"])
        query = {
            "mode": mode,
            "selected_positions": selected,
            "selected_mask_positions": [p for p in selected if p in mask_positions],
            "confidence": {},
        }
        if score_data is not None:
            positions, scores = score_data
            query["confidence"] = {
                str(position): float(score)
                for position, score in zip(_tolist(positions), _tolist(scores))
            }
        self.current["query"] = query

    def record_losa(self, layer, positions, valid, delta, active_indices):
        if self.current is None:
            return
        positions = _tolist(positions)
        valid = _tolist(valid)
        delta = _tolist(delta)
        active_indices = _tolist(active_indices)
        self.current["losa"].append(
            {
                "layer": int(layer),
                "positions": positions,
                "valid_positions": [p for p, is_valid in zip(positions, valid) if is_valid],
                "missing_positions": [p for p, is_valid in zip(positions, valid) if not is_valid],
                "delta": {
                    str(p): (None if math.isnan(value) else float(value))
                    for p, value in zip(positions, delta)
                },
                "active_positions": [positions[index] for index in active_indices],
            }
        )

    def end_step(self, transfer):
        if self.current is None:
            return
        self.current["transfer_positions"] = _tolist(torch.where(transfer)[0])
        self.records.append(self.current)
        self.current = None


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _pearson(xs, ys):
    if len(xs) < 2:
        return None
    x_mean, y_mean = sum(xs) / len(xs), sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_scale = sum((x - x_mean) ** 2 for x in xs)
    y_scale = sum((y - y_mean) ** 2 for y in ys)
    denominator = math.sqrt(x_scale * y_scale)
    return numerator / denominator if denominator else None


def _ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _spearman(xs, ys):
    return _pearson(_ranks(xs), _ranks(ys)) if len(xs) >= 2 else None


def _normalized_ranks(values):
    if len(values) <= 1:
        return [1.0] * len(values)
    return [rank / (len(values) - 1) for rank in _ranks(values)]


def _jaccard(left, right):
    union = left | right
    return len(left & right) / len(union) if union else None


def _random_topk_jaccard(size, k):
    """Expected Jaccard of two independent k-subsets of a size-n universe."""
    denominator = math.comb(size, k)
    return sum(
        math.comb(k, overlap)
        * math.comb(size - k, k - overlap)
        / denominator
        * overlap
        / (2 * k - overlap)
        for overlap in range(max(0, 2 * k - size), k + 1)
    )


def _auroc(scores, labels):
    positive = [score for score, label in zip(scores, labels) if label]
    negative = [score for score, label in zip(scores, labels) if not label]
    if not positive or not negative:
        return None
    wins = sum(
        1.0 if pos > neg else 0.5 if pos == neg else 0.0
        for pos in positive
        for neg in negative
    )
    return wins / (len(positive) * len(negative))


def _step_observations(record, layer):
    query = record.get("query")
    if not query or query["mode"] != "fresh":
        return []
    losa = next((item for item in record["losa"] if item["layer"] == layer), None)
    if losa is None:
        return []
    confidence = {int(key): value for key, value in query["confidence"].items()}
    delta = {
        int(key): value
        for key, value in losa["delta"].items()
        if value is not None
    }
    selected = set(query["selected_mask_positions"])
    active = set(losa["active_positions"])
    transferred = set(record["transfer_positions"])
    return [
        {
            "sample": record["sample"],
            "block": record["block"],
            "step": record["step"],
            "layer": layer,
            "position": position,
            "confidence": confidence[position],
            "delta": delta[position],
            "query_selected": position in selected,
            "losa_active": position in active,
            "transferred": position in transferred,
        }
        for position in sorted(confidence.keys() & delta.keys())
    ]


def _summarize_layer(records, layer, active_topk):
    observations = []
    step_correlations = []
    topk_jaccards = []
    nontrivial_topk_jaccards = []
    random_topk_jaccards = []
    nontrivial_random_topk_jaccards = []
    candidate_counts = []
    query_losa_jaccards = []
    losa_active_in_query = []
    transfer_recalls = {"confidence": [], "delta": [], "combined": []}
    for record in records:
        current = _step_observations(record, layer)
        if not current:
            continue
        candidate_counts.append(len(current))
        observations.extend(current)
        confidence = [item["confidence"] for item in current]
        delta = [item["delta"] for item in current]
        step_correlations.append(_spearman(confidence, delta))
        k = min(int(active_topk), len(current))
        confidence_top = {
            current[index]["position"]
            for index in sorted(range(len(current)), key=confidence.__getitem__)[-k:]
        }
        delta_top = {
            current[index]["position"]
            for index in sorted(range(len(current)), key=delta.__getitem__)[-k:]
        }
        topk_jaccard = _jaccard(confidence_top, delta_top)
        random_topk_jaccard = _random_topk_jaccard(len(current), k)
        topk_jaccards.append(topk_jaccard)
        random_topk_jaccards.append(random_topk_jaccard)
        if len(current) > k:
            nontrivial_topk_jaccards.append(topk_jaccard)
            nontrivial_random_topk_jaccards.append(random_topk_jaccard)

        query_selected = {item["position"] for item in current if item["query_selected"]}
        losa_active = {item["position"] for item in current if item["losa_active"]}
        query_losa_jaccards.append(_jaccard(query_selected, losa_active))
        if losa_active:
            losa_active_in_query.append(
                len(query_selected & losa_active) / len(losa_active)
            )

        transferred = {item["position"] for item in current if item["transferred"]}
        if transferred:
            confidence_rank = _normalized_ranks(confidence)
            delta_rank = _normalized_ranks(delta)
            combined = [a + b for a, b in zip(confidence_rank, delta_rank)]
            for name, scores in (
                ("confidence", confidence),
                ("delta", delta),
                ("combined", combined),
            ):
                predicted = {
                    current[index]["position"]
                    for index in sorted(range(len(current)), key=scores.__getitem__)[-k:]
                }
                transfer_recalls[name].append(
                    len(predicted & transferred) / len(transferred)
                )

    confidence = [item["confidence"] for item in observations]
    delta = [item["delta"] for item in observations]
    labels = [item["transferred"] for item in observations]
    confidence_rank_scores = []
    delta_rank_scores = []
    combined = []
    for record in records:
        current = _step_observations(record, layer)
        if current:
            confidence_rank = _normalized_ranks([item["confidence"] for item in current])
            delta_rank = _normalized_ranks([item["delta"] for item in current])
            confidence_rank_scores.extend(confidence_rank)
            delta_rank_scores.extend(delta_rank)
            combined.extend(a + b for a, b in zip(confidence_rank, delta_rank))
    return {
        "layer": layer,
        "valid_steps": len(candidate_counts),
        "correlatable_steps": len(
            [value for value in step_correlations if value is not None]
        ),
        "pairs": len(observations),
        "mean_candidate_count": _mean(candidate_counts),
        "pearson_confidence_log1p_delta": _pearson(
            confidence, [math.log1p(value) for value in delta]
        ),
        "spearman_confidence_delta": _spearman(confidence, delta),
        "mean_step_spearman": _mean(step_correlations),
        "mean_topk_jaccard": _mean(topk_jaccards),
        "mean_random_topk_jaccard": _mean(random_topk_jaccards),
        "mean_topk_jaccard_lift": (
            _mean(topk_jaccards) - _mean(random_topk_jaccards)
            if topk_jaccards
            else None
        ),
        "nontrivial_topk_steps": len(nontrivial_topk_jaccards),
        "mean_nontrivial_topk_jaccard": _mean(nontrivial_topk_jaccards),
        "mean_nontrivial_random_topk_jaccard": _mean(
            nontrivial_random_topk_jaccards
        ),
        "mean_nontrivial_topk_jaccard_lift": (
            _mean(nontrivial_topk_jaccards)
            - _mean(nontrivial_random_topk_jaccards)
            if nontrivial_topk_jaccards
            else None
        ),
        "mean_query_losa_position_jaccard": _mean(query_losa_jaccards),
        "mean_losa_active_mask_in_query_fraction": _mean(losa_active_in_query),
        "transfer_auroc_confidence": _auroc(confidence, labels),
        "transfer_auroc_delta": _auroc(delta, labels),
        "transfer_auroc_confidence_step_rank": _auroc(
            confidence_rank_scores, labels
        ),
        "transfer_auroc_delta_step_rank": _auroc(delta_rank_scores, labels),
        "transfer_auroc_combined_rank_sum": _auroc(combined, labels),
        "mean_transfer_recall_at_k_confidence": _mean(transfer_recalls["confidence"]),
        "mean_transfer_recall_at_k_delta": _mean(transfer_recalls["delta"]),
        "mean_transfer_recall_at_k_combined": _mean(transfer_recalls["combined"]),
    }


def _slot_usage(records, layer=None):
    totals = {
        "active": 0,
        "active_mask": 0,
        "active_decoded": 0,
        "forced_missing": 0,
        "ranked_active": 0,
        "ranked_active_mask": 0,
        "ranked_active_decoded": 0,
    }
    for record in records:
        mask = set(record["mask_positions"])
        decoded = set(record["decoded_positions"])
        for losa in record["losa"]:
            if layer is not None and losa["layer"] != layer:
                continue
            active = set(losa["active_positions"])
            missing = set(losa["missing_positions"])
            ranked = active - missing
            totals["active"] += len(active)
            totals["active_mask"] += len(active & mask)
            totals["active_decoded"] += len(active & decoded)
            totals["forced_missing"] += len(missing)
            totals["ranked_active"] += len(ranked)
            totals["ranked_active_mask"] += len(ranked & mask)
            totals["ranked_active_decoded"] += len(ranked & decoded)
    active = max(totals["active"], 1)
    ranked = max(totals["ranked_active"], 1)
    totals["active_mask_fraction"] = totals["active_mask"] / active
    totals["active_decoded_fraction"] = totals["active_decoded"] / active
    totals["ranked_active_mask_fraction"] = totals["ranked_active_mask"] / ranked
    totals["ranked_active_decoded_fraction"] = (
        totals["ranked_active_decoded"] / ranked
    )
    return totals


def summarize_records(records, selection_layer, active_topk):
    layers = sorted({item["layer"] for record in records for item in record["losa"]})
    per_layer = [_summarize_layer(records, layer, active_topk) for layer in layers]
    primary = next(
        (item for item in per_layer if item["layer"] == selection_layer),
        _summarize_layer(records, selection_layer, active_topk),
    )

    mode_counts = {}
    for record in records:
        mode = (record.get("query") or {}).get("mode", "missing")
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    return {
        "primary_selection_layer": primary,
        "per_layer": per_layer,
        "losa_slot_usage": _slot_usage(records),
        "losa_slot_usage_by_layer": [
            {"layer": layer, **_slot_usage(records, layer)} for layer in layers
        ],
        "query_mode_counts": mode_counts,
        "record_count": len(records),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure correlation between LLaDA Query confidence and LoSA query delta."
    )
    parser.add_argument("--model_path", default="/data0/ysy/models/LLaDA2.1-mini")
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--prompts_file")
    parser.add_argument("--records", help="Analyze an existing position_records.jsonl")
    parser.add_argument("--gen_length", type=int, default=32)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--query_ratio", type=float, default=0.7)
    parser.add_argument("--selection_layer", type=int, default=1)
    parser.add_argument("--selection_interval", type=int, default=1)
    parser.add_argument("--losa_active_topk", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask_id", type=int, default=156895)
    parser.add_argument("--eos_id", type=int, default=156892)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--moe_expert_patch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output_dir",
        default="experiments/query_losa_correlation/llada_layer1_topk5",
    )
    return parser.parse_args()


def _prompts(args):
    prompts = list(args.prompts or ())
    if args.prompts_file:
        prompts.extend(
            line.strip()
            for line in Path(args.prompts_file).read_text().splitlines()
            if line.strip()
        )
    return prompts or list(DEFAULT_PROMPTS)


def _write_results(output_dir, args, prompts, texts, records, summary, config=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "position_records.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    payload = {
        "config": config or vars(args),
        "prompts": prompts,
        "generated_texts": texts,
        **summary,
    }
    (output_dir / "correlation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    rows = summary["per_layer"]
    if rows:
        with (output_dir / "correlation_by_layer.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    primary = summary["primary_selection_layer"]
    slots = next(
        item
        for item in summary["losa_slot_usage_by_layer"]
        if item["layer"] == primary["layer"]
    )
    correlation_label = (
        "weak"
        if abs(primary["spearman_confidence_delta"] or 0.0) < 0.3
        else "moderate-or-strong"
    )
    (output_dir / "analysis.md").write_text(
        "# Query-sparse / LoSA correlation\n\n"
        f"- Valid position pairs: {primary['pairs']}\n"
        f"- Pooled Spearman: {primary['spearman_confidence_delta']}\n"
        f"- Mean per-step Spearman: {primary['mean_step_spearman']}\n"
        f"- Mean top-k Jaccard: {primary['mean_topk_jaccard']}\n"
        f"- Nontrivial top-k steps: {primary['nontrivial_topk_steps']}\n"
        f"- Mean nontrivial top-k Jaccard: "
        f"{primary['mean_nontrivial_topk_jaccard']}\n"
        f"- Random nontrivial top-k Jaccard: "
        f"{primary['mean_nontrivial_random_topk_jaccard']}\n"
        f"- Nontrivial top-k Jaccard lift: "
        f"{primary['mean_nontrivial_topk_jaccard_lift']}\n"
        f"- Ranked LoSA slots used by decoded positions: "
        f"{slots['ranked_active_decoded_fraction']}\n"
        f"- Transfer AUROC (confidence): {primary['transfer_auroc_confidence']}\n"
        f"- Transfer AUROC (query delta): {primary['transfer_auroc_delta']}\n"
        f"- Transfer AUROC (step-ranked confidence): "
        f"{primary['transfer_auroc_confidence_step_rank']}\n"
        f"- Transfer AUROC (step-ranked query delta): "
        f"{primary['transfer_auroc_delta_step_rank']}\n"
        f"- Transfer AUROC (combined rank sum): "
        f"{primary['transfer_auroc_combined_rank_sum']}\n"
        "\n## Interpretation\n\n"
        f"- The selection-layer rank correlation is {correlation_label}; the two "
        "scores should not be treated as interchangeable.\n"
        "- The chance-adjusted top-k lift measures shared position signal without "
        "the small-candidate-set inflation.\n"
        "- The decoded-slot fraction measures how much ranked LoSA budget is not "
        "spent on currently masked output candidates.\n"
        "- The combined AUROC is exploratory: expand the prompt set before changing "
        "the runtime selector.\n"
    )


def main():
    args = parse_args()
    if args.selection_interval != 1:
        raise ValueError("Correlation collection requires --selection_interval 1")
    if not 0 < args.query_ratio < 1:
        raise ValueError("query_ratio must be in (0, 1)")
    if args.records:
        records = [json.loads(line) for line in Path(args.records).read_text().splitlines()]
        summary = summarize_records(records, args.selection_layer, args.losa_active_topk)
        output_dir = Path(args.output_dir)
        previous_path = output_dir / "correlation_summary.json"
        previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
        _write_results(
            output_dir,
            args,
            previous.get("prompts", []),
            previous.get("generated_texts", []),
            records,
            summary,
            config=previous.get("config"),
        )
        print(json.dumps(summary["primary_selection_layer"], indent=2))
        return
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(args.device).eval()
    resolve_model_family(model, "llada")
    patch_model(
        model,
        model_name="llada",
        ratio=args.query_ratio,
        top_k=64,
        selection_interval=args.selection_interval,
        dense_fallback_mask_count=0,
        selection_layer=args.selection_layer,
        query_sparse=True,
        prefix_sparse=False,
        losa=True,
        losa_active_topk=args.losa_active_topk,
        moe_expert_patch=args.moe_expert_patch,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    collector = QueryLosaCollector()
    model._llada_query_losa_collector = collector

    prompts = _prompts(args)
    texts = []
    with torch.no_grad():
        for sample, prompt in enumerate(prompts):
            collector.start_sample(sample)
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            ).to(model.device)
            generated = model.generate(
                inputs=input_ids,
                gen_length=args.gen_length,
                block_length=args.block_length,
                steps=args.steps,
                temperature=0.0,
                top_p=None,
                top_k=None,
                threshold=args.threshold,
                editing_threshold=0.0,
                num_to_transfer=1,
                mask_id=args.mask_id,
                eos_id=args.eos_id,
                eos_early_stop=False,
            )
            texts.append(tokenizer.decode(generated[0], skip_special_tokens=True))

    del model._llada_query_losa_collector
    summary = summarize_records(
        collector.records, args.selection_layer, args.losa_active_topk
    )
    output_dir = Path(args.output_dir)
    _write_results(output_dir, args, prompts, texts, collector.records, summary)
    print(f"Recorded {len(collector.records)} steps in {output_dir}")
    print(json.dumps(summary["primary_selection_layer"], indent=2))


if __name__ == "__main__":
    main()

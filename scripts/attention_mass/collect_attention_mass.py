#!/usr/bin/env python3
"""Collect query-resolved prefix attention mass for every LLaDA layer and step."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dense.api import DenseRuntime
from src.losa.generation import set_seed
import src.sparse.llada_patch as llada_patch


ALL_TASKS = [
    "2wikimqa", "dureader", "gov_report", "hotpotqa", "lcc", "lsht",
    "multi_news", "multifieldqa_en", "multifieldqa_zh", "musique",
    "narrativeqa", "passage_count", "passage_retrieval_en",
    "passage_retrieval_zh", "qasper", "qmsum", "repobench-p", "samsum",
    "trec", "triviaqa", "vcsum",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="/root/models/LLaDA2.0-mini")
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--prompt_tokens", type=int, required=True)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--gen_length", type=int, default=32)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--page_size", type=int, default=1,
                   help="KV tokens aggregated per column; 1 disables aggregation")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--threshold", type=float, default=1.1)
    p.add_argument("--editing_threshold", type=float, default=1.1)
    return p.parse_args()


def tokenize(tokenizer: Any, prompt: str) -> torch.Tensor:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
    except Exception:
        return tokenizer(prompt, return_tensors="pt").input_ids


def truncate_middle(input_ids: torch.Tensor, budget: int) -> tuple[torch.Tensor, bool, int]:
    length = int(input_ids.shape[-1])
    if length <= budget:
        return input_ids, False, length
    head = budget // 2
    tail = budget - head
    out = torch.cat((input_ids[:, :head], input_ids[:, -tail:]), dim=-1)
    return out, True, length


def choose_record(path: Path, tokenizer: Any, target: int) -> dict[str, Any]:
    """Pick a real record near the target length, then verify it with the model tokenizer."""
    indexed: list[tuple[int, int, dict[str, Any]]] = []
    with path.open(encoding="utf-8") as handle:
        for original_index, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            length_field = int(record.get("length", 0) or 0)
            if length_field >= target:
                indexed.append((abs(length_field - target), original_index, record))
    if not indexed:
        raise RuntimeError(f"{path}: no record has length field >= {target}")
    indexed.sort(key=lambda item: (item[0], item[1]))

    # Tokenize a small neighborhood of candidates and choose the closest actual
    # tokenized length that is still >= target. This avoids scanning all records.
    verified: list[tuple[int, int, int, dict[str, Any], int]] = []
    for _, original_index, record in indexed[:24]:
        ids = tokenize(tokenizer, str(record["prompt"]))
        actual = int(ids.shape[-1])
        if actual >= target:
            verified.append((abs(actual - target), actual, original_index, record, ids))
    if verified:
        _, original_length, original_index, record, original_ids = min(verified, key=lambda x: (x[0], x[1]))
    else:
        # The LongBench length field is not always the model tokenizer length
        # (especially for Chinese tasks). Scan all length-field-qualified records
        # and choose the closest actual tokenized length that is still >= target.
        fallback: list[tuple[int, int, int, dict[str, Any], torch.Tensor]] = []
        for _, original_index, record in indexed:
            ids = tokenize(tokenizer, str(record["prompt"]))
            actual = int(ids.shape[-1])
            if actual >= target:
                fallback.append((abs(actual - target), actual, original_index, record, ids))
        if not fallback:
            raise RuntimeError(f"{path}: no record tokenizes to at least {target} tokens")
        _, original_length, original_index, record, original_ids = min(
            fallback, key=lambda x: (x[0], x[1])
        )
    return {
        "original_index": original_index,
        "record": record,
        "original_ids": original_ids,
        "original_length": original_length,
        "length_field": int(record.get("length", 0) or 0),
    }


class AttentionMassCollector:
    def __init__(self, page_size: int):
        self.page_size = int(page_size)
        self.layer_indices: list[int] = []
        self.handles: list[Any] = []
        self.active = False
        self.prefix_length = 0
        self.current_step = 0
        self.rows: dict[int, list[np.ndarray]] = {}
        self.prefix_share: dict[int, list[float]] = {}
        self.prompt_tokens = 0
        self.block_length = 32
        self.seen_shapes: set[tuple[int, int, int]] = set()

    def install(self, model: torch.nn.Module) -> None:
        base = model.model
        self.layer_indices = list(range(len(base.layers)))
        self.rows = {i: [] for i in self.layer_indices}
        self.prefix_share = {i: [] for i in self.layer_indices}
        for layer_idx in self.layer_indices:
            layer = base.layers[layer_idx]
            attn = getattr(layer, "attention", None)
            if attn is None:
                raise RuntimeError(f"layer {layer_idx} has no .attention module")
            handle = attn.register_forward_hook(self._make_hook(layer_idx), with_kwargs=False)
            self.handles.append(handle)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def reset(self, prompt_tokens: int, block_length: int) -> None:
        self.prompt_tokens = int(prompt_tokens)
        self.block_length = int(block_length)
        self.rows = {i: [] for i in self.layer_indices}
        self.prefix_share = {i: [] for i in self.layer_indices}

    def begin_step(self, step: int, prefix_length: int) -> None:
        self.current_step = int(step)
        self.prefix_length = int(prefix_length)
        self.active = True

    def end_step(self) -> None:
        self.active = False

    def _make_hook(self, layer_idx: int):
        def hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: Any) -> None:
            if not self.active:
                return
            if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
                raise RuntimeError(
                    f"layer {layer_idx} did not return attention weights; eager attention is required"
                )
            weights = output[1]
            if weights.ndim != 4:
                raise RuntimeError(f"unexpected attention rank {weights.shape}")
            self.seen_shapes.add(tuple(map(int, weights.shape[-3:])))
            # weights: [query_heads, block_queries, prefix + current_block].
            # Preserve the 32 current-block query rows; average only query heads.
            prefix_weights = weights[0, :, :, : self.prefix_length].float()
            query_mass = prefix_weights.mean(dim=0).detach()  # [block_queries, prefix]
            query_total = query_mass.sum(dim=-1, keepdim=True)
            if not bool(torch.isfinite(query_total).all()) or bool((query_total <= 0).any()):
                raise RuntimeError("invalid prefix attention normalization")
            conditional_query_mass = query_mass / query_total
            n_pages = conditional_query_mass.shape[-1] // self.page_size
            pages = conditional_query_mass[
                :, : n_pages * self.page_size
            ].view(
                conditional_query_mass.shape[0], n_pages, self.page_size
            ).sum(dim=-1)
            self.rows[layer_idx].append(pages.detach().cpu().numpy())
            # Mean prefix share over query heads and current-block queries.
            total = float(prefix_weights.sum(dim=-1).mean().item())
            self.prefix_share[layer_idx].append(total)
        return hook


def install_wrappers(collector: AttentionMassCollector) -> None:
    original_cached_forward = llada_patch._cached_forward
    original_dense_block_prefill = llada_patch._dense_block_prefill

    def exact_current_block_prefill(model, input_ids, position_ids, cache):
        """Eager all-visible current-block prefill used only to capture step 0."""
        base = model.model
        hidden_states = base.word_embeddings(input_ids)
        position_embeddings = base.rotary_emb(hidden_states, position_ids)
        collector.begin_step(0, collector.prompt_tokens)
        try:
            for layer in base.layers:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_value=cache,
                    output_attentions=False,
                    output_router_logits=False,
                    use_cache=True,
                    position_embeddings=position_embeddings,
                )[0]
        finally:
            collector.end_step()
        hidden_states = base.norm(hidden_states)
        return SimpleNamespace(
            logits=model.lm_head(hidden_states), past_key_values=cache
        )

    def wrapped_dense_block_prefill(model, input_ids, position_ids, cache=None):
        is_current_block = (
            int(input_ids.shape[1]) == collector.block_length
            and int(position_ids.shape[1]) == collector.block_length
            and int(position_ids[0, 0].item()) == collector.prompt_tokens
        )
        if is_current_block and cache is not None:
            return exact_current_block_prefill(model, input_ids, position_ids, cache)
        return original_dense_block_prefill(model, input_ids, position_ids, cache)

    def wrapped_cached_forward(*args: Any, **kwargs: Any):
        if args:
            prefix_cache = args[4]
            selection_state = args[5]
            original_prefix_length = args[17] if len(args) > 17 else None
        else:
            prefix_cache = kwargs["prefix_cache"]
            selection_state = kwargs["selection_state"]
            original_prefix_length = kwargs.get("original_prefix_length")
        prefix_length = (
            int(original_prefix_length)
            if original_prefix_length is not None
            else int(prefix_cache[0][0].shape[-2])
        )
        # Step 0 is captured by exact block prefill before the first token reveal.
        # Cached forwards start from the state after one token has been revealed.
        step = int(selection_state.get("step", 0)) + 1
        collector.begin_step(step, prefix_length)
        try:
            return original_cached_forward(*args, **kwargs)
        finally:
            collector.end_step()

    llada_patch._dense_block_prefill = wrapped_dense_block_prefill
    llada_patch._cached_forward = wrapped_cached_forward


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.prompt_tokens <= 0 or args.page_size <= 0:
        raise ValueError("prompt_tokens and page_size must be positive")
    if args.prompt_tokens % args.page_size:
        raise ValueError("prompt_tokens must be divisible by page_size")

    print("[load] LLaDA 2.0 mini with eager attention", flush=True)
    runtime = DenseRuntime(
        family="llada",
        model_path=args.model_path,
        dtype="bfloat16",
        attn_implementation="eager",
        moe_expert_patch=True,
    )
    model, tokenizer = runtime.load()
    actual_impl = getattr(model.config, "_attn_implementation", None)
    if actual_impl != "eager":
        raise RuntimeError(f"expected eager attention, got {actual_impl!r}")
    print(f"[load] attention={actual_impl} layers={len(model.model.layers)}", flush=True)

    collector = AttentionMassCollector(args.page_size)
    collector.install(model)
    install_wrappers(collector)

    metadata: dict[str, Any] = {
        "purpose": "Query-resolved prefix attention-mass trajectory",
        "model_path": args.model_path,
        "data_dir": str(args.data_dir),
        "tasks": args.tasks,
        "prompt_tokens": args.prompt_tokens,
        "block_length": args.block_length,
        "gen_length": args.gen_length,
        "steps": args.steps,
        "page_size": args.page_size,
        "expected_matrix": [len(model.model.layers), args.block_length, args.block_length, args.prompt_tokens // args.page_size],
        "seed": args.seed,
        "threshold": args.threshold,
        "editing_threshold": args.editing_threshold,
        "step0": "exact eager all-mask current-block prefill",
        "steps_1_to_31": "cached eager forwards after each controlled token reveal",
        "attention_normalization": "mean over query heads only; each block-query row is prefix-conditional token/page mass",
        "samples": [],
    }

    for sample_idx, task in enumerate(args.tasks):
        path = args.data_dir / f"{task}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        chosen = choose_record(path, tokenizer, args.prompt_tokens)
        original_ids = chosen["original_ids"]
        ids, truncated, original_length = truncate_middle(original_ids, args.prompt_tokens)
        if int(ids.shape[-1]) != args.prompt_tokens:
            raise RuntimeError(
                f"{task}: expected exact {args.prompt_tokens}, got {ids.shape[-1]}"
            )
        ids = ids.to(model.device)
        collector.reset(args.prompt_tokens, args.block_length)
        print(
            f"[run] {sample_idx + 1}/{len(args.tasks)} task={task} "
            f"index={chosen['original_index']} length_field={chosen['length_field']} "
            f"original_tokens={original_length} truncated={truncated}",
            flush=True,
        )
        with torch.inference_mode():
            generated = model.generate(
                inputs=ids,
                gen_length=args.gen_length,
                block_length=args.block_length,
                steps=args.steps,
                temperature=0.0,
                threshold=args.threshold,
                editing_threshold=args.editing_threshold,
                max_post_steps=0,
                eos_early_stop=False,
                mask_id=156895,
                eos_id=156892,
                num_to_transfer=1,
            )
        row_counts = {layer: len(rows) for layer, rows in collector.rows.items()}
        if len(set(row_counts.values())) != 1:
            raise RuntimeError(f"{task}: inconsistent row counts {row_counts}")
        expected_steps = args.block_length
        if row_counts[collector.layer_indices[0]] != expected_steps:
            raise RuntimeError(
                f"{task}: expected {expected_steps} step rows, got {row_counts}"
            )
        expected_pages = args.prompt_tokens // args.page_size
        page_matrix = np.stack(
            [np.stack(collector.rows[layer], axis=0) for layer in collector.layer_indices],
            axis=0,
        ).astype(np.float16)
        share_matrix = np.stack(
            [np.asarray(collector.prefix_share[layer], dtype=np.float32) for layer in collector.layer_indices],
            axis=0,
        ).astype(np.float16)
        expected_shape = (len(collector.layer_indices), expected_steps, args.block_length, expected_pages)
        if page_matrix.shape != expected_shape:
            raise RuntimeError(f"{task}: unexpected shape {page_matrix.shape} != {expected_shape}")
        prediction = ""
        try:
            if isinstance(generated, torch.Tensor):
                prediction = tokenizer.decode(generated[0], skip_special_tokens=True)
        except Exception:
            prediction = ""
        sample_meta = {
            "sample_idx": sample_idx,
            "task": task,
            "original_index": chosen["original_index"],
            "length_field": chosen["length_field"],
            "original_input_tokens": original_length,
            "input_tokens": int(ids.shape[-1]),
            "truncated": truncated,
            "truncation": "drop_middle_keep_head_tail",
            "row_counts": row_counts,
            "page_matrix_shape": list(page_matrix.shape),
            "answer": chosen["record"].get("answer") or chosen["record"].get("answers"),
            "prediction_preview": prediction[:300],
        }
        metadata["samples"].append(sample_meta)
        np.savez_compressed(
            args.output_dir / f"sample_{sample_idx:02d}_{task}.npz",
            page_mass=page_matrix,
            prefix_share=share_matrix,
            layers=np.asarray(collector.layer_indices, dtype=np.int32),
        )
        print(f"[run] matrix={page_matrix.shape} rows={row_counts[0]}", flush=True)
        (args.output_dir / "manifest.partial.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        torch.cuda.empty_cache()

    collector.remove()
    metadata["attention_shapes_seen"] = sorted(list(collector.seen_shapes))
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Opt-in logical KV usage; never measures physical HBM traffic.

Compaction writes per-head counts into fixed layer slots. One copy after each
real sparse forward records all layers, outside graph capture/verification.
Dense rows are reconstructed from shapes. Export is the only CPU readback.
Collectors belong to one generation request and are not concurrency-safe.
"""
from collections import defaultdict
import csv
import json
from uuid import uuid4
from pathlib import Path

import torch


def count_compact_tokens(pages, page_size, prefix_length):
    """Torch oracle/fallback: ignore padding and count partial last pages."""
    if page_size <= 0 or prefix_length < 0:
        raise ValueError("invalid page size or prefix length")
    remaining = prefix_length - pages.clamp_min(0) * page_size
    counts = remaining.clamp(min=0, max=page_size)
    return torch.where(pages >= 0, counts, 0).sum(-1, dtype=torch.int32)


class KVStats:
    """Bounded-chunk device history, plus small CPU shape/phase metadata."""

    def __init__(self, model, family, block_length, chunk_size=1024):
        if chunk_size <= 0 or block_length <= 0:
            raise ValueError("statistics chunk size and block length must be positive")
        layers = model.model.layers
        heads, query_heads, layer_ids = [], [], []
        for index, layer in enumerate(layers):
            attn = getattr(layer, "attention" if family == "llada" else "self_attn")
            heads.append(int(attn.num_key_value_heads))
            query_heads.append(int(getattr(attn, "num_heads", 0) or attn.num_attention_heads))
            layer_ids.append(int(attn.layer_idx))
        if not heads or len(set(heads)) != 1 or len(set(query_heads)) != 1:
            raise ValueError("KV statistics currently require uniform GQA head counts")
        if len(set(layer_ids)) != len(layers):
            raise ValueError("duplicate layer IDs")
        if query_heads[0] % heads[0]:
            raise ValueError("invalid GQA grouping")
        self.layer_ids = layer_ids
        self.heads = heads[0]
        self.query_heads = query_heads[0]
        self.block_length = block_length
        self.chunk_size = chunk_size
        self.family = family
        self.request_id = uuid4().hex
        self.latest = torch.empty((len(layers), self.heads), device=model.device, dtype=torch.int32)
        self.layer_views = {layer_id: self.latest[i] for i, layer_id in enumerate(layer_ids)}
        self.chunks = []
        self.records = []
        self.sparse_records = 0
        self.block_steps = defaultdict(int)
        self.completed = False

    def _reserve(self):
        chunk, slot = divmod(self.sparse_records, self.chunk_size)
        if chunk == len(self.chunks):
            self.chunks.append(torch.empty(
                (self.chunk_size, *self.latest.shape), device=self.latest.device, dtype=torch.int32))
        return chunk, slot

    def forward(self, dispatch, eager, model, family, input_ids, attention_mask,
                position_ids, **kwargs):
        """Call dispatch once, then commit once; no Python hooks inside replay."""
        if input_ids.shape[0] != 1:
            raise ValueError("KV statistics support batch size one")
        queries = int(input_ids.shape[1])
        cache = kwargs.get("prefix_cache", ())
        prefix = int(cache[0][0].shape[-2]) if cache else 0
        if cache and (len(cache) != len(self.layer_ids)
                      or any(k.shape[-2] != prefix for k, _ in cache)):
            raise ValueError("KV statistics expect the complete uniform-prefix LoSA cache")
        context = kwargs.get("losa_context_kwargs")
        sparse = bool(context and context["states"] and prefix > 0)
        initialize = bool(context and not context["states"] and prefix > 0)
        active = min(int(context["active_count"]), queries) if sparse else queries
        block_id = prefix // self.block_length
        # LLaDA's initial block forward can also store KV. Only the driver's
        # explicit finalization marker distinguishes it from block_initial.
        finalize = kwargs.pop("kv_stats_phase", None) == "finalize"
        step = None if finalize else self.block_steps[block_id]
        phase = ("finalize" if finalize else "initialize" if initialize else
                 "block_initial" if step == 0 else "dense_eager")
        if sparse:
            # Reserve outside capture so allocator activity cannot enter the graph.
            chunk, slot = self._reserve()
        before = dict(getattr(model, "_losa_optimized_graph_stats", {})) if sparse else {}
        if context:
            # A request owns a stable collector pointer; each block gets its own
            # context/states. Graph kwargs retain this collector until reset.
            kwargs["losa_context_kwargs"] = {**context, "kv_stats": self}
        result = dispatch(eager, model, family, input_ids, attention_mask, position_ids, **kwargs)
        if sparse:
            after = getattr(model, "_losa_optimized_graph_stats", {})
            if after.get("captures", 0) > before.get("captures", 0):
                phase = "capture_replay"
            elif after.get("replays", 0) > before.get("replays", 0):
                phase = "replay"
            elif after.get("priming_steps", 0) > before.get("priming_steps", 0):
                phase = "prime"
            else:
                phase = "sparse_eager"
            # Same current stream as replay return; one small D2D copy for ALL
            # layers. Nothing is synchronized or transferred to CPU here.
            self.chunks[chunk][slot].copy_(self.latest)
            self.sparse_records += 1
        self.records.append(dict(
            forward_id=len(self.records), block_id=block_id, step_id=step, phase=phase,
            query_count=queries, active_count=active, prefix_length=prefix,
            block_kv_tokens=queries, sparse_slot=self.sparse_records-1 if sparse else None))
        if not finalize:
            self.block_steps[block_id] += 1
        return result

    def export(self, output_dir=None, include_heads=False, compact=False):
        """Explicit synchronization/readback, kept outside generation timing."""
        if self.latest.is_cuda:
            torch.cuda.synchronize(self.latest.device)
        buffers = []
        left = self.sparse_records
        for chunk in self.chunks:
            used = min(left, self.chunk_size)
            buffers.append(chunk[:used].cpu().tolist())
            left -= used
        rows, heads = [], []
        for record in self.records:
            slot = record["sparse_slot"]
            data = None if slot is None else buffers[slot // self.chunk_size][slot % self.chunk_size]
            for layer, layer_id in enumerate(self.layer_ids):
                counts = [record["prefix_length"]] * self.heads if data is None else data[layer]
                prefix_mean = sum(counts) / self.heads
                row = {k: v for k, v in record.items() if k != "sparse_slot"}
                row.update(request_id=self.request_id, layer_id=layer_id,
                           kv_heads=self.heads, query_heads=self.query_heads,
                           prefix_union_mean_tokens=prefix_mean,
                           prefix_union_min_tokens=min(counts), prefix_union_max_tokens=max(counts),
                           active_query_mean_kv=record["block_kv_tokens"] + prefix_mean,
                           all_query_mean_fresh_kv=record["block_kv_tokens"] +
                           record["active_count"] / record["query_count"] * prefix_mean)
                rows.append(row)
                if include_heads:
                    for head, count in enumerate(counts):
                        heads.append(dict(forward_id=record["forward_id"], layer_id=layer_id,
                                          kv_head=head, prefix_tokens=count,
                                          block_tokens=record["block_kv_tokens"]))
        groups = defaultdict(list)
        for row in rows:
            groups[(row["layer_id"], row["phase"])].append(row)
        summary = []
        for (layer_id, phase), group in sorted(groups.items()):
            query_total = sum(r["query_count"] for r in group)
            active_total = sum(r["active_count"] for r in group)
            summary.append(dict(
                layer_id=layer_id, phase=phase, calls=len(group),
                query_count=query_total, active_count=active_total,
                active_query_mean_kv=sum(r["active_query_mean_kv"] for r in group)/len(group),
                active_query_weighted_mean_kv=sum(r["active_query_mean_kv"]*r["active_count"]
                                                 for r in group)/active_total,
                all_query_mean_fresh_kv=sum(r["all_query_mean_fresh_kv"] for r in group)/len(group),
                all_query_weighted_mean_fresh_kv=sum(r["all_query_mean_fresh_kv"]*r["query_count"]
                                                    for r in group)/query_total,
                min_active_query_mean_kv=min(r["active_query_mean_kv"] for r in group),
                max_active_query_mean_kv=max(r["active_query_mean_kv"] for r in group)))
        result = dict(
            schema_version=1, request_id=self.request_id, family=self.family,
            completed=self.completed,
            scope="generation_forwards_excluding_prefill",
            units="logical KV token pairs per query; not HBM loads or bytes",
            forward_count=len(self.records), sparse_forward_count=self.sparse_records,
            chunk_size=self.chunk_size, allocated_chunks=len(self.chunks),
            allocated_bytes=(self.latest.numel() + sum(c.numel() for c in self.chunks))*4,
            kv_steps=rows, kv_layers=summary)
        if compact:
            if not rows:
                raise RuntimeError("cannot compact empty KV statistics")
            result["avg_kv"] = sum(float(r["active_query_mean_kv"]) for r in rows) / len(rows)
            result["avg_all_query_fresh_kv"] = sum(float(r["all_query_mean_fresh_kv"]) for r in rows) / len(rows)
            result["aggregation"] = "mean over all layers and generation forwards for this sample"
            result.pop("kv_steps", None)
            result.pop("kv_layers", None)
            result.pop("kv_heads", None)
            if output_dir is not None:
                path = Path(output_dir)
                path.mkdir(parents=True, exist_ok=True)
                (path / "kv_stats.json").write_text(json.dumps(result, indent=2))
            return result
        if include_heads:
            result["kv_heads"] = heads
        if output_dir is not None:
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            for name, records in (("kv_steps", rows), ("kv_layers", summary),
                                  *((("kv_heads", heads),) if include_heads else ())):
                with (path / (name + ".csv")).open("w", newline="") as f:
                    if records:
                        writer = csv.DictWriter(f, fieldnames=list(records[0]))
                        writer.writeheader()
                        writer.writerows(records)
            (path / "kv_stats.json").write_text(json.dumps(result, indent=2))
        return result


def export_kv_stats(model, output_dir=None, include_heads=False, compact=False):
    collector = getattr(model, "_losa_kv_stats", None)
    return None if collector is None else collector.export(output_dir, include_heads, compact)

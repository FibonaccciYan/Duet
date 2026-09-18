"""CPU-only NarrativeQA speed corpus, paired model tokenizations."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import shutil
import tokenizers
from tokenizers import Tokenizer

p = argparse.ArgumentParser()
p.add_argument("--source", required=True, type=Path)
p.add_argument("--sdar", required=True, type=Path)
p.add_argument("--llada21", required=True, type=Path)
p.add_argument("--output", required=True, type=Path)
a = p.parse_args()
root = a.output
root.mkdir(parents=True, exist_ok=True)
if (root / "manifest.json").exists():
    raise RuntimeError("Dataset already exists; choose a new output directory")
sha = lambda b: hashlib.sha256(b).hexdigest()
write = lambda path, obj: path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+"\n")
source_bytes = a.source.read_bytes()
rows = [json.loads(line) for line in source_bytes.decode().splitlines()]
paths = {"sdar": a.sdar, "llada21": a.llada21}
toks = {k: Tokenizer.from_file(str(v / "tokenizer.json")) for k, v in paths.items()}
encoded, audit, eligible = {}, [], []
for idx, row in enumerate(rows):
    ids = {k: t.encode(row["prompt"], add_special_tokens=False).ids for k, t in toks.items()}
    lengths = {k: len(v) for k, v in ids.items()}
    ok = min(lengths.values()) > 32768
    audit.append(dict(source_line_zero_based=idx, source_id=row["_id"],
                      original_prompt_token_lengths=lengths, eligible=ok))
    if ok:
        eligible.append(idx)
        encoded[idx] = ids
assert len(eligible) >= 80
selected = sorted(random.Random(42).sample(eligible, 80))
assert len(set(rows[i]["_id"] for i in selected)) == 80
write(root / "selection_audit.json", dict(seed=42, eligible_count=len(eligible),
                                        selected_source_lines=selected, candidates=audit))
with (root / "selected_originals.jsonl").open("w") as f:
    for rank, idx in enumerate(selected):
        f.write(json.dumps(dict(sample_index=rank, source_line_zero_based=idx,
                               original_prompt_token_lengths=audit[idx]["original_prompt_token_lengths"],
                               original=rows[idx]), ensure_ascii=False)+"\n")
roundtrip_mismatches = {}
for family, tokenizer in toks.items():
    dest = root / family
    dest.mkdir(exist_ok=True)
    for size in (8192, 16384, 32768):
        mismatch = 0
        with (dest / f"{size}.jsonl").open("w") as f:
            for rank, idx in enumerate(selected):
                original = rows[idx]
                all_ids = encoded[idx][family]
                cropped = all_ids[:size//2] + all_ids[-size//2:]
                text = tokenizer.decode(cropped, skip_special_tokens=False)
                retokenized = tokenizer.encode(text, add_special_tokens=False).ids
                exact = retokenized == cropped
                mismatch += not exact
                rec = dict(sample_index=rank, source_id=original["_id"],
                           source_line_zero_based=idx, dataset="LongBench/narrativeqa",
                           tokenizer=family, target_tokens=size,
                           original_prompt_tokens=len(all_ids),
                           head_tokens=size//2, tail_tokens=size//2,
                           token_ranges=[[0,size//2],[len(all_ids)-size//2,len(all_ids)]],
                           input_ids=cropped, prompt=text, input=original.get("input",""),
                           answers=original.get("answers",[]),
                           token_ids_sha256=sha(json.dumps(cropped,separators=(",",":")).encode()),
                           text_roundtrip_exact=exact, text_retokenized_tokens=len(retokenized))
                assert len(cropped) == size
                f.write(json.dumps(rec,ensure_ascii=False)+"\n")
        roundtrip_mismatches[f"{family}/{size}"] = mismatch
    target = root / "tokenizers" / family
    target.mkdir(parents=True,exist_ok=True)
    for name in ("tokenizer.json","tokenizer_config.json","special_tokens_map.json",
                 "added_tokens.json"):
        if (paths[family]/name).exists():
            shutil.copyfile(paths[family]/name,target/name)
shutil.copyfile(__file__,root/"build_dataset.py") if Path(__file__).resolve() != (root/"build_dataset.py").resolve() else None
files = {str(f.relative_to(root)):dict(bytes=f.stat().st_size,sha256=sha(f.read_bytes()))
         for f in sorted(root.rglob("*")) if f.is_file()}
write(root / "manifest.json", dict(created_date="2026-09-17", source_path=str(a.source),
      source_sha256=sha(source_bytes), source_records=len(rows), eligible_records=len(eligible),
      selected_records=80, seed=42, threshold_strictly_greater_than=32768,
      lengths=[8192,16384,32768], families=list(toks), tokenizers_version=tokenizers.__version__,
      input_convention="stored source prompt; no chat template; add_special_tokens=False",
      truncation="first N/2 token IDs + last N/2 token IDs; no separator",
      authoritative_field="input_ids", roundtrip_mismatches=roundtrip_mismatches,files=files))
print(json.dumps(dict(selected=len(selected),eligible=len(eligible),
                      roundtrip_mismatches=roundtrip_mismatches),indent=2))

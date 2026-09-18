import collections
import json
import sys
from pathlib import Path

root, stage, budget = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
report = json.loads((root / "report.json").read_text())
expected = dict(hotpotqa=200, triviaqa=200, narrativeqa=200, qasper=200, multifieldqa_en=150)
lengths = dict(hotpotqa=32, triviaqa=32, narrativeqa=128, qasper=128, multifieldqa_en=64)
if stage == "smoke":
    expected = dict.fromkeys(expected, 1)
assert collections.Counter(r["task"] for r in report["rows"]) == expected
cfg = report["sparse_config"]
assert cfg["prefix_strict_budget"] and cfg["prefix_sparse"] and cfg["query_sparse"]
assert cfg["prefix_token_budget"] == budget and cfg["selection_interval"] == 4
assert report["generation_config"]["eos_early_stop"]
for r in report["rows"]:
    assert 0 <= r["generated_tokens"] <= lengths[r["task"]]
    assert r["input_tokens"] + lengths[r["task"]] <= 32768
if stage == "smoke":
    stats = report["prefix_selection_stats"]
    assert stats["records"] > 0 and stats["strict_final_violations"] == 0
    for line in (root / "prefix_selection.jsonl").read_text().splitlines():
        item = json.loads(line)
        assert item["selected_size"] <= item["budget"]
(root / "VALIDATED.json").write_text(json.dumps(dict(stage=stage, budget=budget,
    rows=len(report["rows"]), strict=True, eos_early_stop=True), indent=2))
print("VALIDATED", root, len(report["rows"]), flush=True)

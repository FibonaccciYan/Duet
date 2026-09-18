"""Verify checksum, pair identity, exact token count, and reproducible head/tail cuts."""
import hashlib
import json
import random
from pathlib import Path
from tokenizers import Tokenizer

root = Path(__file__).resolve().parent
manifest = json.loads((root / "manifest.json").read_text())
for filename, expected in manifest["files"].items():
    data = (root / filename).read_bytes()
    assert len(data) == expected["bytes"], filename
    assert hashlib.sha256(data).hexdigest() == expected["sha256"], filename
audit = json.loads((root / "selection_audit.json").read_text())
eligible = [r["source_line_zero_based"] for r in audit["candidates"] if r["eligible"]]
assert len(eligible) == 85
assert sorted(random.Random(42).sample(eligible, 80)) == audit["selected_source_lines"]
originals = [json.loads(s) for s in (root / "selected_originals.jsonl").read_text().splitlines()]
assert len(originals) == 80
expected_ids = [r["original"]["_id"] for r in originals]
assert len(set(expected_ids)) == 80
checked = 0
for family in ("sdar", "llada21"):
    tok = Tokenizer.from_file(str(root / "tokenizers" / family / "tokenizer.json"))
    encoded = [tok.encode(r["original"]["prompt"], add_special_tokens=False).ids for r in originals]
    assert all(len(x) > 32768 for x in encoded)
    for n in (8192, 16384, 32768):
        records = [json.loads(s) for s in (root / family / f"{n}.jsonl").read_text().splitlines()]
        assert [r["source_id"] for r in records] == expected_ids
        for r, ids in zip(records, encoded):
            assert len(r["input_ids"]) == n
            assert r["input_ids"] == ids[:n//2] + ids[-n//2:]
            assert tok.decode(r["input_ids"], skip_special_tokens=False) == r["prompt"]
            checked += 1
print(f"PASS: {len(expected_ids)} paired samples, {checked} exact head-tail crops, all manifest hashes")

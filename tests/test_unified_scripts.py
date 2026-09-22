import hashlib
import json
from pathlib import Path
import subprocess
import os

import pytest
from scripts.common.run import arguments, plan

ROOT = Path(__file__).resolve().parents[1]


def test_qk_longbench_cli_forwarding():
    qk = jobs("sparse", "longbench", "llada21", "--set", "prefix_selector=qk_tc")
    for job in qk:
        cmd = job["command"]
        assert cmd[cmd.index("--prefix_selector")+1] == "qk_tc"
    for job in jobs("sparse", "longbench", "llada21"):
        assert "--prefix_selector" not in job["command"]


def test_prefix_dense_before_query_selection_cli_forwarding():
    jobs_on = jobs("sparse", "longbench", "llada21",
                   "--set", "prefix_dense_before_query_selection=true")
    for job in jobs_on:
        assert "--prefix_dense_before_query_selection" in job["command"]
    jobs_off = jobs("sparse", "longbench", "llada21",
                    "--set", "prefix_dense_before_query_selection=false")
    for job in jobs_off:
        assert "--no-prefix_dense_before_query_selection" in job["command"]


def test_short_sparse_forwards_qk_tc_and_shallow_full_prefix():
    job = jobs("sparse", "short", "sdar", "--tasks", "math500",
               "--set", "PREFIX_SELECTOR=qk_tc",
               "--set", "PREFIX_DENSE_BEFORE_QUERY_SELECTION=true",
               "--set", "PREFIX_TOKEN_BUDGET=256")[0]
    env = job["env"]
    assert env["PREFIX_SELECTOR"] == "qk_tc"
    assert env["PREFIX_DENSE_BEFORE_QUERY_SELECTION"] == "true"
    assert env["PREFIX_TOKEN_BUDGET"] == "256"


def test_short_scoring_routes_to_correct_tasks(tmp_path):
    capture = tmp_path / "capture.sh"
    record = tmp_path / "argv.txt"
    capture.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE_ARGV"\n')
    capture.chmod(0o755)
    expected = {
        ("math500", "llada21"): "math_500",
        ("math500", "sdar"): "math_500",
        ("gsm8k", "llada21"): "gsm8k",
        ("gsm8k", "sdar"): "gsm8k_sdar",
    }
    for (benchmark, model), task_name in expected.items():
        job = jobs("sparse", "short", model, "--tasks", benchmark, "--stage", "full")[0]
        env = {**os.environ, **job["env"], "PYTHON": str(capture),
               "OUTPUT_ROOT": str(tmp_path / f"{model}-{benchmark}"),
               "CAPTURE_ARGV": str(record)}
        subprocess.run(job["command"], env=env, cwd=ROOT, check=True)
        argv = record.read_text().splitlines()
        assert argv[argv.index("--tasks") + 1] == task_name


def jobs(method, task, model, *extra):
    return plan(arguments(["--method", method, "--task", task, "--model", model, *extra]))


@pytest.mark.parametrize("method", ["dense", "sparse", "losa", "focus"])
@pytest.mark.parametrize("model", ["sdar", "llada21"])
def test_short_optimized_and_explicit_sdar_threshold(method, model):
    for job in jobs(method, "short", model):
        env = job["env"]
        assert env["IMPLEMENTATION"] == "optimized"
        assert env["METHOD"] == method
        assert float(env["THRESHOLD"]) == (.95 if model == "sdar" else .7)
        assert job["command"] == ["bash", str(ROOT / "eval_instruct/eval.sh")]


def test_short_protocols_remain_distinct():
    dense = jobs("dense", "short", "sdar", "--stage", "full")
    assert [j["env"]["BENCHMARK"] for j in dense] == ["gsm8k", "humaneval", "mmlu", "math"]
    assert all(j["env"]["GEN_LENGTH"] == "4096" for j in dense)
    assert dense[2]["env"]["NUM_FEWSHOT"] == "5"
    assert all(j["env"]["REMASKING_STRATEGY"] == "low_confidence_static" for j in dense)
    native = jobs("sparse", "short", "sdar", "--tasks", "mmlu", "humaneval")
    assert [j["env"]["GEN_LENGTH"] for j in native] == ["128", "768"]
    assert native[0]["env"]["NUM_FEWSHOT"] == "5"
    assert native[0]["env"]["REMASKING_STRATEGY"] == "sequential"
    assert jobs("focus", "short", "sdar")[0]["env"]["REMASKING_STRATEGY"] == "low_confidence_dynamic"
    assert all(j["env"]["GEN_LENGTH"] == "16384" for j in jobs("dense", "short", "llada21"))


def test_cli_override_does_not_change_defaults():
    env = jobs("dense", "short", "sdar", "--threshold", ".91", "--gen-length", "64",
               "--fewshot", "0", "--tasks", "gsm8k", "--stage", "smoke", "--limit", "1")[0]["env"]
    assert env["THRESHOLD"] == "0.91" and env["GEN_LENGTH"] == "64"
    assert env["LIMIT"] == "1"
    assert jobs("dense", "short", "sdar")[0]["env"]["THRESHOLD"] == "0.95"


@pytest.mark.parametrize("method", ["dense", "sparse", "losa", "focus"])
def test_speed_retains_full_protocol(method):
    j = jobs(method, "speed", "llada21")
    assert len(j) == 3
    for item in j:
        c = item["config"]
        assert (c["samples"], c["warmups"], c["repeats"], c["gen_length"]) == (80, 2, 3, 256)
        assert (c["threshold"], c["editing_threshold"]) == (.7, .5)
        assert Path(c["data_dir"]) == ROOT / "data/narrativeqa_speed80"
    smoke = jobs(method, "speed", "sdar", "--stage", "smoke", "--contexts", "8192")[0]["config"]
    assert (smoke["samples"], smoke["warmups"], smoke["repeats"]) == (1, 2, 1)


def test_no_smode_entry_and_exactly_twelve_entries():
    assert len(list((ROOT / "scripts/unified").glob("*.sh"))) == 12
    assert not list((ROOT / "scripts").rglob("*smode*"))
    assert not (ROOT / "scripts/performance").exists()


def test_bundled_dataset_not_ignored_and_checksums_match():
    root = ROOT / "data/narrativeqa_speed80"
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        content = (root / name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == expected["sha256"]
        assert len(content) < 100_000_000
    # Isolated validation snapshots may not have .git.
    if (ROOT / ".git").exists():
        proc = subprocess.run(["git", "check-ignore", str(root / "sdar/32768.jsonl")],
                              cwd=ROOT, capture_output=True)
        assert proc.returncode == 1


def test_longbench_preserves_task_lengths_and_threshold_override():
    from scripts.original.quality.longbench_quality import GEN_LENGTHS, TASKS
    assert GEN_LENGTHS["hotpotqa"] == 32 and GEN_LENGTHS["qasper"] == 128
    assert len(TASKS) == 5
    cmd = jobs("sparse", "longbench", "sdar")[0]["command"]
    assert cmd[cmd.index("--threshold")+1] == "0.95"
    assert "--tasks" not in cmd and "--gen_length" not in cmd


def test_cache_compat_preserves_primitive_list_not_struct():
    from scripts.common.eval_entry import normalize_primitive_lists
    original = {"choices": {"_type": "List", "feature": {"_type": "Value", "dtype": "string"}}}
    result = normalize_primitive_lists(original)
    assert original["choices"]["_type"] == "List"
    assert result["choices"] == {"_type": "Sequence", "feature": {"_type": "Value", "dtype": "string"}}
    with pytest.raises(ValueError):
        normalize_primitive_lists({"_type": "List", "feature": {"field": {"_type": "Value"}}})


@pytest.mark.parametrize("method", ["dense", "sparse", "losa", "focus"])
@pytest.mark.parametrize("model", ["sdar", "llada21"])
def test_all_mmlu_defaults_are_five_shot(method, model):
    for job in jobs(method, "short", model, "--tasks", "mmlu"):
        assert job["env"]["NUM_FEWSHOT"] == "5"


@pytest.mark.parametrize("method", ["dense", "sparse", "losa", "focus"])
@pytest.mark.parametrize("model", ["sdar", "llada21"])
def test_real_short_shell_forwards_optimized_five_shot(tmp_path, method, model):
    # Execute the real shell, intercept only the final Python invocation.
    # This verifies that eval.sh does not override the launcher's settings.
    capture = tmp_path / "capture.sh"
    record = tmp_path / "argv.txt"
    capture.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE_ARGV"\n')
    capture.chmod(0o755)
    job = jobs(method, "short", model, "--tasks", "mmlu", "--stage", "full")[0]
    env = {**os.environ, **job["env"], "PYTHON": str(capture),
           "OUTPUT_ROOT": str(tmp_path / "output"), "CAPTURE_ARGV": str(record)}
    subprocess.run(job["command"], env=env, cwd=ROOT, check=True)
    argv = record.read_text().splitlines()
    assert argv[argv.index("--num_fewshot")+1] == "5"
    assert argv[argv.index("--tasks")+1] == "mmlu_generative"
    model_args = argv[argv.index("--model_args")+1]
    assert "implementation=optimized" in model_args
    assert f"method={method}" in model_args
    if model == "sdar":
        assert "threshold=0.95" in model_args
    assert argv[argv.index("--batch_size")+1] == "1"

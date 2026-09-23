"""Validated unified front end. Planning imports no CUDA/model libraries."""
import argparse
import csv
from datetime import datetime
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = json.loads((ROOT / "scripts/configs/defaults.json").read_text())
METHODS = ("dense", "sparse", "losa", "focus")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--task", choices=("longbench", "short", "speed"), required=True)
    p.add_argument("--model", choices=("llada21", "sdar", "both"), default=None)
    p.add_argument("--model-path")
    p.add_argument("--gpu", type=int)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--output", type=Path)
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--config", type=Path, help="JSON object containing CLI option defaults")
    p.add_argument("--implementation", choices=("reference", "optimized"))
    p.add_argument("--profile", choices=("default", "strict-budget", "all-tasks"), default="default")
    p.add_argument("--stage", choices=("smoke", "full", "all"))
    p.add_argument("--tasks", nargs="+")
    p.add_argument("--contexts", nargs="+", type=int)
    for key in ("samples", "limit", "warmups", "repeats", "gen-length",
                "block-length", "steps", "seed", "fewshot", "port", "max-context-tokens"):
        p.add_argument("--" + key, type=int)
    for key in ("threshold", "editing-threshold", "temperature"):
        p.add_argument("--" + key, type=float)
    p.add_argument("--remasking-strategy")
    p.add_argument("--eos-early-stop", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--resume-from", type=Path)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="Method generation options (speed), CLI option (LongBench), env key (short)")
    p.add_argument("--dry-run", "--print-config", dest="dry_run", action="store_true")
    p.add_argument("--offline", action="store_true",
                   help="Use existing Hugging Face caches without network requests")
    p.add_argument("--restore-holder", action="store_true",
                   help="GPU5 only; opt in to stopping/restoring its existing holder")
    return p


def arguments(argv=None):
    p = parser()
    pre, _ = p.parse_known_args(argv)
    if pre.config:
        cfg = json.loads(pre.config.read_text())
        if not isinstance(cfg, dict):
            p.error("config must be an object")
        valid = {a.dest for a in p._actions} - {"method", "task", "config", "help"}
        if set(cfg) - valid:
            p.error("unknown config keys: " + str(sorted(set(cfg) - valid)))
        p.set_defaults(**cfg)
    a = p.parse_args(argv)
    if a.model is None:
        a.model = "both" if a.task == "speed" else "llada21"
    if a.model == "both" and a.model_path:
        p.error("--model-path requires one model")
    if a.implementation is None:
        a.implementation = ("optimized" if a.profile == "strict-budget"
                            else DEFAULTS[a.task]["implementation"])
    if a.task in ("speed", "short") and a.implementation != "optimized":
        p.error("unified speed/short entries use optimized implementations")
    if a.stage is None:
        a.stage = "all" if a.task == "short" and a.method == "dense" else "full"
    if a.task != "speed" and any(getattr(a, k) is not None for k in ("contexts", "samples", "warmups", "repeats")):
        p.error("contexts/samples/warmups/repeats are speed-only")
    if a.task == "speed" and (a.tasks or a.limit is not None or a.fewshot is not None):
        p.error("speed uses --samples, not tasks/limit/fewshot")
    if a.task != "short" and (a.port is not None or a.fewshot is not None):
        p.error("port/fewshot apply only to short benchmarks")
    if a.task != "longbench" and a.resume_from:
        p.error("resume-from applies only to LongBench predictions")
    if a.task == "short" and a.eos_early_stop is not None:
        p.error("short benchmarks retain the evaluator EOS behavior")
    if a.task == "speed" and a.max_context_tokens is not None:
        p.error("speed context is set by --contexts and the stored token IDs")
    if a.task != "longbench" and a.profile != "default":
        p.error("profiles apply only to LongBench")
    if a.profile == "strict-budget" and a.method != "sparse":
        p.error("strict-budget profile requires sparse")
    if a.profile == "all-tasks" and (a.method != "dense" or a.model != "llada21"):
        p.error("original all-tasks runner supports dense LLaDA only")
    if a.restore_holder and a.gpu != 5:
        p.error("--restore-holder is restricted to the existing GPU5 holder")
    if a.gpu is not None and a.gpu < 0:
        p.error("GPU index must be nonnegative")
    for k in ("samples", "limit", "repeats", "gen_length", "block_length", "steps"):
        if getattr(a, k) is not None and getattr(a, k) <= 0:
            p.error(k + " must be positive")
    if a.warmups is not None and a.warmups < 0:
        p.error("warmups must be nonnegative")
    if a.output is None:
        a.output = ROOT / "results/unified" / f"{a.method}_{a.task}_{datetime.now():%Y%m%d_%H%M%S}"
    a.output = Path(a.output).resolve()
    a.overrides = {}
    for item in a.set:
        key, sep, value = item.partition("=")
        if not sep or not key:
            p.error("--set requires KEY=VALUE")
        if key in a.overrides:
            p.error("duplicate --set: " + key)
        a.overrides[key] = value
    return a


def value(a, name, default):
    v = getattr(a, name)
    return default if v is None else v


def plan(a):
    jobs = []
    models = ("sdar", "llada21") if a.model == "both" else (a.model,)
    stages = ("smoke", "full") if a.stage == "all" else (a.stage,)
    for stage in stages:
        for model in models:
            family = "llada" if model == "llada21" else "sdar"
            model_path = a.model_path or DEFAULTS["models"][model]
            threshold = value(a, "threshold", .7 if family == "llada" else .95)
            editing = value(a, "editing_threshold", .5)
            out = a.output / stage / model
            if a.task == "speed":
                d = DEFAULTS["speed"]
                for length in value(a, "contexts", d["contexts"]):
                    if length not in (8192, 16384, 32768):
                        raise ValueError("stored speed contexts are 8192/16384/32768")
                    config = dict(method=a.method, model=model, model_path=model_path,
                                  length=length, output=str(out / str(length)), stage=stage,
                                  data_dir=str(a.data_dir or ROOT / "data/narrativeqa_speed80"),
                                  samples=value(a, "samples", 1 if stage == "smoke" else 80),
                                  repeats=value(a, "repeats", 1 if stage == "smoke" else 3),
                                  warmups=value(a, "warmups", 2), threshold=threshold,
                                  editing_threshold=editing,
                                  remasking_strategy=value(a, "remasking_strategy", "low_confidence_dynamic"),
                                  overrides=a.overrides, gpu=a.gpu)
                    for k in ("gen_length", "block_length", "steps", "temperature", "seed", "eos_early_stop"):
                        config[k] = value(a, k, d[k])
                    if config["samples"] > 80:
                        raise ValueError("dataset has exactly 80 samples")
                    if config["eos_early_stop"]:
                        raise ValueError("fixed-budget speed protocol requires EOS early-stop disabled")
                    jobs.append(dict(kind="speed", config=config))
            elif a.task == "short":
                d = DEFAULTS["short"]
                tasks = a.tasks or d["dense_tasks" if a.method == "dense" else "other_tasks"]
                for task in tasks:
                    if task not in d["sdar_task_gen_lengths"]:
                        raise ValueError("unsupported short task: " + task)
                    gen = d["llada_gen_length"] if family == "llada" else (
                        d["sdar_dense_gen_length"] if a.method == "dense" else d["sdar_task_gen_lengths"][task])
                    few = d["mmlu_fewshot"] if task == "mmlu" else 0
                    remask = d["sdar_dense_remasking"] if a.method == "dense" else d[
                        "sdar_focus_remasking" if a.method == "focus" else "sdar_other_remasking"]
                    env = dict(MODEL_TYPE=family, METHOD=a.method, IMPLEMENTATION="optimized",
                               EVAL_ENTRY="scripts.common.eval_entry",
                               MODEL=model_path, PYTHON=a.python, BENCHMARK=task,
                               OUTPUT_ROOT=str(out / task), THRESHOLD=str(threshold),
                               EDITING_THRESHOLD=str(editing), GEN_LENGTH=str(value(a, "gen_length", gen)),
                               NUM_FEWSHOT=str(value(a, "fewshot", few)),
                               MAIN_PROCESS_PORT=str(value(a, "port", d["port"])),
                               BLOCK_LENGTH=str(value(a, "block_length", 32)),
                               STEPS=str(value(a, "steps", 32)),
                               TEMPERATURE=str(value(a, "temperature", 0.)),
                               REMASKING_STRATEGY=value(a, "remasking_strategy", remask))
                    # Optimized LoSA's GQA default is group_mean, not the old adapter's per-head default.
                    if a.method == "losa":
                        env["PAPER_LOSA_GQA_MODE"] = "group_mean"
                    if a.seed is not None:
                        env["EVAL_SEED"] = str(a.seed)
                    if a.max_context_tokens is not None:
                        env["MAX_PROMPT_LEN"] = str(a.max_context_tokens)
                    if stage == "smoke" or a.limit is not None:
                        env["LIMIT"] = str(value(a, "limit", d["smoke_limit"]))
                    allowed = {"QUERY_SPARSE", "PREFIX_SPARSE", "PREFIX_TOKEN_BUDGET", "PREFIX_STRICT_BUDGET",
                               "SPARSE_DLM_RATIO", "SPARSE_DLM_TOP_K", "SPARSE_DLM_SELECTION_INTERVAL",
                               "QUERY_DENSE_THRESHOLD", "SPARSE_DLM_SELECTION_LAYER", "PREFIX_SELECTOR",
                               "PREFIX_DENSE_BEFORE_QUERY_SELECTION", "PREFIX_RESCREEN_FULL_KV",
                               "FOCUS_ALPHA", "PAPER_LOSA_PAGE_SIZE", "PAPER_LOSA_TOKEN_BUDGET",
                               "PAPER_LOSA_ACTIVE_TOPK", "PAPER_LOSA_GQA_MODE", "PAPER_LOSA_BACKEND",
                               "PAPER_LOSA_KV_STATS", "PAPER_LOSA_KV_STATS_CHUNK_SIZE",
                               "PAPER_LOSA_KV_STATS_OUTPUT_DIR", "PAPER_LOSA_KV_STATS_INCLUDE_HEADS",
                               "PAPER_LOSA_KV_STATS_COMPACT",
                               "MOE_EXPERT_PATCH", "DTYPE", "ATTN_IMPLEMENTATION", "MAX_PROMPT_LEN"}
                    specific = {
                        "dense": set(),
                        "sparse": {k for k in allowed if k.startswith(("QUERY_", "PREFIX_", "SPARSE_"))},
                        "focus": {"FOCUS_ALPHA"},
                        "losa": {k for k in allowed if k.startswith("PAPER_LOSA_")},
                    }
                    allowed = specific[a.method] | {"MOE_EXPERT_PATCH", "DTYPE", "ATTN_IMPLEMENTATION"}
                    if set(a.overrides) - allowed:
                        raise ValueError("unsupported short --set: " + str(set(a.overrides) - allowed))
                    env.update(a.overrides)
                    jobs.append(dict(kind="command", command=["bash", str(ROOT / "eval_instruct/eval.sh")],
                                     env=env, output=str(out / task),
                                     metadata=dict(model=model, method=a.method, stage=stage, task=task)))
            else:
                if a.profile == "all-tasks" and a.overrides:
                    raise ValueError("all-tasks does not accept method --set overrides")
                module = ("scripts.original.quality.longbench_dense_all_tasks" if a.profile == "all-tasks"
                          else "scripts.common.longbench")
                cmd = [a.python, "-B", "-m", module, "--model_path", model_path,
                       "--data_dir", str(a.data_dir or os.environ.get("LONGBENCH_DATA", ROOT / "data/longbench")),
                       "--output_dir", str(out), "--threshold", str(threshold),
                       "--editing_threshold", str(editing),
                       "--max_context_tokens", str(value(a, "max_context_tokens", 32768)),
                       "--block_length", str(value(a, "block_length", 32)),
                       "--steps", str(value(a, "steps", 32))]
                if a.profile != "all-tasks":
                    cmd += ["--family", family, "--method", a.method,
                            "--implementation", a.implementation, "--seed", str(value(a, "seed", 42))]
                    if a.tasks:
                        cmd += ["--tasks", *a.tasks]
                    if a.remasking_strategy:
                        cmd += ["--remasking_strategy", a.remasking_strategy]
                    if a.eos_early_stop is not None:
                        cmd += ["--eos_early_stop" if a.eos_early_stop else "--no-eos_early_stop"]
                elif a.tasks:
                    raise ValueError("all-tasks uses the original dataset manifest task list")
                if stage == "smoke" or a.limit is not None:
                    cmd += ["--limit", str(value(a, "limit", 1))]
                if a.resume_from:
                    cmd += ["--resume_from", str(a.resume_from)]
                if a.gen_length is not None or a.temperature is not None:
                    raise ValueError("LongBench uses original task lengths and temperature0; not short/speed generation overrides")
                for key, v in a.overrides.items():
                    if key not in {"ratio", "selection_layer", "query_dense_threshold",
                                   "query_sparse", "prefix_sparse", "prefix_token_budget",
                                   "prefix_selector", "prefix_dense_before_query_selection", "prefix_strict_budget", "focus_alpha", "losa_token_budget",
                                   "losa_page_size", "losa_active_topk", "losa_gqa_mode",
                                   "losa_backend", "moe_expert_patch", "dtype"}:
                        raise ValueError("unsupported LongBench option: " + key)
                    relevant = (
                        key in {"moe_expert_patch", "dtype"}
                        or a.method == "sparse" and key in {"ratio", "selection_layer", "query_dense_threshold",
                            "query_sparse", "prefix_sparse", "prefix_token_budget", "prefix_dense_before_query_selection", "prefix_strict_budget", "prefix_selector"}
                        or a.method == "focus" and key == "focus_alpha"
                        or a.method == "losa" and key.startswith("losa_"))
                    if not relevant:
                        raise ValueError(f"{key} does not apply to {a.method}")
                    if v.lower() in ("true", "false"):
                        cmd += ["--" + ("" if v.lower() == "true" else "no-") + key]
                    else:
                        cmd += ["--" + key, v]
                if a.profile == "strict-budget":
                    if "prefix_token_budget" in a.overrides:
                        raise ValueError("strict-budget profile fixes the 256/512/1024 matrix")
                    for budget in (256, 512, 1024):
                        target = out / f"budget{budget}"
                        copy = list(cmd)
                        copy[copy.index("--output_dir")+1] = str(target)
                        copy += ["--query_sparse", "--prefix_sparse", "--prefix_strict_budget",
                                 "--prefix_token_budget", str(budget)]
                        if family == "sdar" and not a.remasking_strategy:
                            copy += ["--remasking_strategy", "low_confidence_dynamic"]
                        jobs.append(dict(kind="command", command=copy, env={}, output=str(target)))
                else:
                    jobs.append(dict(kind="command", command=cmd, env={}, output=str(out)))
    return jobs


def run(a, jobs):
    if a.gpu is None:
        raise ValueError("execution requires explicit --gpu; planning does not")
    if a.output.exists() and any(a.output.iterdir()):
        raise ValueError("refusing nonempty output directory: " + str(a.output))
    used = int(subprocess.check_output(
        ["nvidia-smi", "-i", str(a.gpu), "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"], text=True).strip())
    if used > 256:
        raise RuntimeError(f"GPU{a.gpu} busy ({used} MiB); no process will be stopped")
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "plan.json").write_text(json.dumps(jobs, indent=2))
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(a.gpu), PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / "eval_instruct"),
               PYTHONDONTWRITEBYTECODE="1", HF_HUB_DOWNLOAD_TIMEOUT="60")
    if a.offline:
        env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", HF_EVALUATE_OFFLINE="1")
    results = []
    try:
        for i, job in enumerate(jobs):
            if job["kind"] == "speed":
                cfg = a.output / f"cell_{i:03d}.json"
                cfg.write_text(json.dumps(job["config"], indent=2))
                command = [a.python, "-B", "-m", "scripts.common.speed", "--config", str(cfg)]
            else:
                command = job["command"]
            child_env = {**env, **job.get("env", {})}
            if a.task == "short":
                # Do not let unrelated shell generation variables silently
                # override the resolved plan. HF cache/offline vars are retained.
                controlled = set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)",
                                             (ROOT / "eval_instruct/eval.sh").read_text()))
                for key in controlled - {"PYTHONPATH", "HF_ALLOW_CODE_EVAL"}:
                    child_env.pop(key, None)
                child_env.update(job["env"])
            log = a.output / f"job_{i:03d}.log"
            print(f"START {i+1}/{len(jobs)} {command}", flush=True)
            with log.open("w") as f:
                child = subprocess.Popen(command, env=child_env, cwd=ROOT, stdout=f,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                def terminate(signum, frame):
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    raise SystemExit(128+signum)
                previous = {s: signal.signal(s, terminate) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
                try:
                    rc = child.wait()
                finally:
                    for s, handler in previous.items():
                        signal.signal(s, handler)
            results.append(dict(index=i, exit_code=rc, log=str(log)))
            (a.output / "status.json").write_text(json.dumps(results, indent=2))
            if rc:
                print(log.read_text()[-6000:], file=sys.stderr)
                raise RuntimeError(f"job {i} failed, exit {rc}")
            print(f"DONE {i+1}/{len(jobs)}", flush=True)
        report_rows = []
        for i, job in enumerate(jobs):
            path = Path(job["config"]["output"] if job["kind"] == "speed" else job["output"])
            row = dict(job=i, output=str(path), exit_code=0)
            if job["kind"] == "speed":
                row.update(json.loads((path / "summary.json").read_text()))
                row["method"] = job["config"]["method"]
                row["model"] = job["config"]["model"]
                row["length"] = job["config"]["length"]
            else:
                row["result_files"] = [str(p) for p in path.rglob("*.json")]
            report_rows.append(row)
        (a.output / "summary.json").write_text(json.dumps(report_rows, indent=2))
        with (a.output / "summary.csv").open("w") as f:
            keys = sorted({k for r in report_rows for k in r})
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(report_rows)
        (a.output / "REPORT.md").write_text(
            "# Unified run\n\nSee `plan.json` for resolved parameters and `summary.json` "
            "for results. Quality scores remain in the original evaluator outputs; "
            "smoke scores are not formal quality measurements.\n")
        (a.output / "COMPLETE.json").write_text(json.dumps(dict(completed=True, jobs=len(jobs))))
    finally:
        (a.output / "execution.json").write_text(json.dumps(
            dict(gpu=a.gpu, method=a.method, task=a.task,
                 optimized_short=a.task == "short", finished_jobs=results), indent=2))


def main():
    a = arguments()
    jobs = plan(a)
    if a.dry_run:
        print(json.dumps(jobs, indent=2))
        return
    if a.restore_holder and os.environ.get("UNIFIED_RESERVED") != "1":
        argv = [x for x in sys.argv[1:] if x != "--restore-holder"]
        env = {**os.environ, "UNIFIED_RESERVED": "1"}
        code = subprocess.call(
            ["bash", str(ROOT / "scripts/original/performance/run_reserved_gpu5.sh"),
             a.python, "-B", "-m", "scripts.common.run", *argv], cwd=ROOT, env=env)
        raise SystemExit(code)
    run(a, jobs)


if __name__ == "__main__":
    main()

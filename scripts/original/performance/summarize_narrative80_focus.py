"""Reaggregate saved dense and new FOCUS with exactly the same step exclusion."""
import argparse
import json
from pathlib import Path
import statistics
from scripts.original.performance.narrative80_focus_benchmark import (
    MODELS, LENGTHS, DENSE, noninitial_rows, atomic_json)


def aggregate(samples):
    records=[r for sample in samples for r in noninitial_rows(sample["step_records"])]
    total=sum(r["cuda_ms"] for r in records)
    seconds=sum(sample["e2e_median_seconds"] for sample in samples)
    return dict(samples=len(samples),sum_sample_median_seconds=seconds,
                throughput_tokens_per_second=256*len(samples)/seconds,
                noninitial_step_count=len(records),noninitial_step_total_ms=total,
                noninitial_step_mean_ms=total/len(records) if records else None,
                sample_unweighted_step_mean_ms=statistics.mean(
                    sum(r["cuda_ms"] for r in noninitial_rows(s["step_records"]))/
                    len(noninitial_rows(s["step_records"]))
                    for s in samples if noninitial_rows(s["step_records"]))
                    if records else None)


def load_samples(folder):
    files=sorted(folder.glob("sample_[0-9][0-9][0-9].json"))
    if len(files)!=80:
        raise ValueError(f"{folder}: expected 80 complete samples, found {len(files)}")
    samples=[json.loads(p.read_text()) for p in files]
    if [s["sample_index"] for s in samples]!=list(range(80)):
        raise ValueError("sample index mismatch")
    for sample in samples:
        assert sample["instrumentation_matches"]
        expected=sample["runs"][0]["step_count"]-len(sample["runs"][0]["steps_per_block"])
        assert len(noninitial_rows(sample["step_records"]))==expected
    return samples


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--focus-root",type=Path,
                   default=Path("results/narrative80_focus_gpu4_20260917/formal"))
    p.add_argument("--dense-root",type=Path,default=DENSE)
    p.add_argument("--dense-only",action="store_true",help="Read-only reaggregation; never runs dense")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    cells=[]
    for model in MODELS:
        for length in LENGTHS:
            dense=load_samples(args.dense_root/f"{model}_{length}_dense_optimized")
            row=dict(model=model,requested_context=length,prompt_tokens=dense[0]["prompt_tokens"],
                     dense=aggregate(dense))
            if not args.dense_only:
                folder=args.focus_root/f"{model}_{length}_focus_optimized"
                assert json.loads((folder/"COMPLETE.json").read_text())["samples"]==80
                focus=load_samples(folder)
                for d,f in zip(dense,focus):
                    for k in ("sample_index","source_id","prompt_sha256","prompt_tokens","generation_length"):
                        assert d[k]==f[k],(model,length,k)
                row["focus"]=aggregate(focus)
                row["e2e_speedup"]=row["dense"]["sum_sample_median_seconds"]/row["focus"]["sum_sample_median_seconds"]
                a,b=row["dense"]["noninitial_step_mean_ms"],row["focus"]["noninitial_step_mean_ms"]
                row["noninitial_step_speedup"]=a/b if a is not None and b is not None else None
            cells.append(row)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(args.output,dict(definition="exclude per-block zero-based index0 only; step-weighted mean",
                                dense_rerun=False,cells=cells))
    print(json.dumps(cells,indent=2))


if __name__=="__main__":
    main()

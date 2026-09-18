"""Attribute optional strict-budget trimming to block setup, not every decode."""
import ast
import contextlib
import inspect
import json
import os
from pathlib import Path
import statistics
import textwrap
import time
import types
import torch
from src.reference.losa.generation import load_model_and_tokenizer
from src.runtime import patch_method
from src.kernels.optimized.function_binding import _bind_globals
import src.reference.sparse.sparse_ops as ops
import src.optimized.sparse.prefix as prefix

OUT = Path("results/prefix_strict_budget_20260918/overhead")
OUT.mkdir(parents=True, exist_ok=True)


class Probe:
    def __init__(self):
        self.events = []
        self.selections = []
        self.block = -1

    @contextlib.contextmanager
    def region(self, name, candidate_length=0, union_size=0):
        a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        a.record()
        yield
        b.record()
        self.events.append((name, self.block, candidate_length, union_size, a, b))

    def rows(self):
        return [dict(name=n, block=block, candidate_length=l, union_size=u,
                     ms=a.elapsed_time(b)) for n, block, l, u, a, b in self.events]


def tree_of(f):
    f = inspect.unwrap(f)
    if hasattr(f, "__func__"):
        f = f.__func__
    tree = ast.parse(textwrap.dedent(inspect.getsource(f)))
    tree.body[0].decorator_list = []
    return f, tree


def compile_fn(f, tree, overrides):
    ns = {**f.__globals__, **overrides}
    exec(compile(ast.fix_missing_locations(tree), "<strict-budget-profile>", "exec"), ns)
    return ns[f.__name__]


def clip_probe(f, probe):
    original, tree = tree_of(f)
    branches = [n for n in ast.walk(tree) if isinstance(n, ast.If)
                and ast.unparse(n.test).startswith("strict_budget and")]
    assert len(branches) == 1
    length = "length" if f is prefix.prefix_indices else "prefix_length"
    wrap = ast.parse(f'with _probe.region("strict_clip", {length}, union_size):\n pass').body[0]
    wrap.body = branches[0].body
    branches[0].body = [wrap]
    return compile_fn(original, tree, {"_probe": probe})


def instrument(generate, family, probe):
    original, tree = tree_of(generate)
    distance = clip_probe(ops._distance_prefix_indices, probe)
    raw, _ = _bind_globals(ops._raw_l1_prefix_indices, {"_distance_prefix_indices": distance})
    selector = clip_probe(prefix.prefix_indices, probe)
    selector, _ = _bind_globals(selector, {"reference": raw})

    def select(*a, **kw):
        with probe.region("selector"):
            result = selector(*a, **kw)
        probe.selections.append(dict(block=probe.block, candidate_length=a[1].shape[-2],
                                     selected_size=result.numel(), budget=min(a[2],a[1].shape[-2])))
        return result

    compactor = original.__globals__["_compact_prefix_cache"]
    compactor, _ = _bind_globals(compactor, {"_prefix_indices": select})
    def compact(*a, **kw):
        with probe.region("selection_and_kv_compaction"):
            return compactor(*a, **kw)

    cached = original.__globals__["_cached_forward" if family == "llada" else "_sparse_cached_forward"]
    def forward(*a, **kw):
        with probe.region("cached_forward"):
            return cached(*a, **kw)

    overrides = {"_compact_prefix_cache": compact,
                 "_cached_forward" if family == "llada" else "_sparse_cached_forward": forward}
    if family == "llada":
        loops = [n for n in ast.walk(tree) if isinstance(n,ast.For)
                 and isinstance(n.target,ast.Name) and n.target.id=="block_idx"]
        assert len(loops)==1
        block=loops[0]
        idx=next(i for i,n in enumerate(block.body) if isinstance(n,ast.For)
                 and isinstance(n.target,ast.Name) and n.target.id=="_")
        wrap=ast.parse('with _probe.region("block_initialization"):\n pass').body[0]
        wrap.body=block.body[:idx]
        block.body=[ast.parse("_probe.block = block_idx").body[0],wrap]+block.body[idx:]
        overrides.update(_probe=probe, _trim_to_first_eos=lambda x,e:x)
    else:
        blockfn=original.__globals__["block_diffusion_generate"]
        bf, bt=tree_of(blockfn)
        blocks=[n for n in ast.walk(bt) if isinstance(n,ast.For)
                and isinstance(n.target,ast.Name) and n.target.id=="num_block"]
        loops=[n for n in ast.walk(bt) if isinstance(n,ast.For)
               and isinstance(n.target,ast.Name) and n.target.id=="step"]
        assert len(blocks)==len(loops)==1
        blocks[0].body.insert(0,ast.parse("_probe.block = num_block").body[0])
        wrap=ast.parse('with _probe.region("block_initialization") if step == 0 else _null():\n pass').body[0]
        wrap.body=loops[0].body
        loops[0].body=[wrap]
        overrides["block_diffusion_generate"]=compile_fn(bf,bt,{"_probe":probe,"_null":contextlib.nullcontext})
    return torch.inference_mode()(compile_fn(original,tree,overrides))


def main():
    family=os.environ["BENCH_FAMILY"]
    model_path="/data0/ysy/models/"+("LLaDA2.1-mini" if family=="llada" else "SDAR-8B-Chat-b32")
    model,tok=load_model_and_tokenizer(family,model_path=model_path)
    patch_method(model,"sparse_optimized",model_name=family,query_sparse=True,
                 prefix_sparse=True,prefix_token_budget=256,moe_expert_patch=True)
    original=model.generate
    if family=="llada":
        f=inspect.unwrap(original)
        if hasattr(f, "__func__"):
            f=f.__func__
        f,_=_bind_globals(f,{"_trim_to_first_eos":lambda x,e:x})
        original=types.MethodType(torch.inference_mode()(f),model)
    options=dict(gen_length=64,block_length=32,steps=32,temperature=0.,eos_early_stop=False)
    options.update(dict(threshold=.7,editing_threshold=.5,num_to_transfer=1)
                   if family=="llada" else dict(threshold=.95,remasking_strategy="low_confidence_dynamic",mask_id=151669))
    data=Path("/data0/gs/dataset_preparation/narrativeqa_speed80_20260917/dataset")
    results=[]
    for n in (8192,16384,32768):
        item=json.loads(next((data/("llada21" if family=="llada" else "sdar")/f"{n}.jsonl").open()))
        ids=item["input_ids"]; actual=min(n,32768-64)
        ids=ids[:actual//2]+ids[-(actual-actual//2):]
        inputs=torch.tensor([ids],dtype=torch.long,device="cuda")
        for strict in (False,True):
            setattr(model.config,family+"_prefix_strict_budget",strict)
            for _ in range(2):
                torch.manual_seed(42);original(inputs=inputs,**options);torch.cuda.synchronize()
        references={};times={False:[],True:[]}
        for repeat in range(3):
            for strict in ((False,True) if repeat%2==0 else (True,False)):
                setattr(model.config,family+"_prefix_strict_budget",strict)
                torch.manual_seed(42);torch.cuda.synchronize();start=time.perf_counter()
                out=original(inputs=inputs,**options);torch.cuda.synchronize()
                times[strict].append(time.perf_counter()-start);references[strict]=out.cpu()
        for strict in (False,True):
            setattr(model.config,family+"_prefix_strict_budget",strict)
            runs=[]
            for repeat in range(3):
                probe=Probe();fn=instrument(original,family,probe)
                torch.manual_seed(42)
                output=fn(model,inputs=inputs,**options);torch.cuda.synchronize()
                assert torch.equal(output.cpu(),references[strict])
                runs.append(dict(events=probe.rows(),selections=probe.selections))
            row=dict(family=family,context=n,prompt=actual,gen_length=64,budget=256,
                     sample_index=item["sample_index"],source_id=item["source_id"],
                     strict=strict,e2e_seconds=times[strict],runs=runs,
                     options=options,stats_collector_enabled=False)
            results.append(row)
            (OUT/f"{family}.json").write_text(json.dumps(results,indent=2))
            print("DONE",family,n,strict,"e2e",statistics.median(times[strict]),
                  "clip_calls",sum(e["name"]=="strict_clip" for e in runs[0]["events"]),flush=True)

if __name__=="__main__":
    main()

from types import SimpleNamespace
import pytest
import torch
from scripts.original.performance.narrative80_focus_benchmark import (
    Clock, clone_driver, noninitial_rows, step_summary, validate_mode)
from scripts.original.performance.summarize_narrative80_focus import aggregate


class Event:
    def record(self):
        pass
    def elapsed_time(self, other):
        return 2.0


def test_excludes_only_first_not_second_and_global_weighting():
    a=[dict(block=0,loop_iteration=0,cuda_ms=100),
       dict(block=0,loop_iteration=1,steady=False,cuda_ms=2)]
    b=[dict(block=1,loop_iteration=0,cuda_ms=100),
       dict(block=1,loop_iteration=1,steady=False,cuda_ms=8),
       dict(block=1,loop_iteration=2,cuda_ms=8),
       dict(block=1,loop_iteration=3,cuda_ms=8)]
    result=aggregate([dict(step_records=a,e2e_median_seconds=1),
                      dict(step_records=b,e2e_median_seconds=1)])
    assert result["noninitial_step_mean_ms"]==6.5
    assert result["sample_unweighted_step_mean_ms"]==5.0
    assert step_summary(a)["noninitial_step_mean_ms"]==2
    assert noninitial_rows(a)[0]["loop_iteration"]==1


def test_empty_metric_is_not_zero():
    assert step_summary([dict(loop_iteration=0,cuda_ms=1)])["noninitial_step_mean_ms"] is None


@pytest.mark.parametrize("family",["sdar","llada"])
def test_counts_final_edit_transfer_and_excludes_finalize(monkeypatch,family):
    import src.optimized.focus.generation as generation
    import src.optimized.focus.prefill as prefill
    monkeypatch.setattr(prefill,"build_prefix",lambda *a,**kw:())
    forwards=[]
    def forward(model,**kw):
        final=kw.get("cache_only",False)
        forwards.append(final)
        return SimpleNamespace(positions=torch.arange(4),logits=torch.zeros(1,4,4),
                               cache=SimpleNamespace(to_legacy_cache=lambda:()))
    monkeypatch.setattr(generation,"focus_optimized_forward",forward)
    # Each transfer resolves every mask. LLaDA must additionally execute its
    # no-edit confirmation step, whose transfer happens before a break.
    def sdar(tokens,active,*a,**kw):
        transfer=active.clone();tokens[active]=1
        return transfer,int(transfer.sum())
    def llada(tokens,old,active,*a,**kw):
        transfer=active.clone();tokens[active]=1
        return transfer,int(transfer.sum()),0
    monkeypatch.setattr(generation,"_selected_sdar_transfer",sdar)
    monkeypatch.setattr(generation,"_selected_llada_transfer",llada)
    clock=Clock(timing=True,event_factory=Event)
    fn=clone_driver(generation.focus_optimized_generate,clock)
    model=SimpleNamespace(device=torch.device("cpu"),generation_config=SimpleNamespace(eos_token_id=None))
    out=fn(model,family=family,inputs=torch.tensor([[0,0,0,0]]),
           gen_length=8,block_length=4,steps=4,mask_id=3,eos_early_stop=False)
    assert out.tokens.tolist()==[[1]*8]
    assert sum(forwards)==2
    expected=2 if family=="llada" else 1
    assert list(clock.counts.values())==[expected,expected]
    assert len(clock.rows())==expected*2
    assert len(noninitial_rows(clock.rows()))==(2 if family=="llada" else 0)


def test_gpu_and_formal_guards(monkeypatch):
    formal=SimpleNamespace(smoke=False,limit=80,repeats=3)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","5")
    with pytest.raises(ValueError):validate_mode(formal)
    validate_mode(SimpleNamespace(smoke=True,limit=1,repeats=1))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","4")
    validate_mode(formal)
    with pytest.raises(ValueError):validate_mode(SimpleNamespace(smoke=True,limit=1,repeats=1))

from types import SimpleNamespace
import pytest
import torch
import src.optimized.focus.generation as generation
import src.optimized.focus.prefill as prefill

@pytest.mark.parametrize("strategy,expected",[("low_confidence_dynamic",1),("low_confidence_static",4)])
def test_sdar_no_empty_forward_and_exact_finalize(monkeypatch,strategy,expected):
    calls=[]
    monkeypatch.setattr(prefill,"build_prefix",lambda *a,**k:())
    def forward(model,**kw):
        tokens=kw["input_ids"]
        final=kw.get("cache_only",False)
        calls.append((final,tokens.clone()))
        if final:
            assert not (tokens==3).any()
        else:
            assert (tokens==3).any(), "redundant forward after all masks resolved"
        logits=torch.zeros(1,4,4);logits[...,1]=20
        return SimpleNamespace(positions=torch.arange(4),logits=logits,
                               cache=SimpleNamespace(to_legacy_cache=lambda:()))
    monkeypatch.setattr(generation,"focus_optimized_forward",forward)
    model=SimpleNamespace(device=torch.device("cpu"),generation_config=SimpleNamespace(eos_token_id=None))
    out=generation.focus_optimized_generate(model,family="sdar",inputs=torch.tensor([[0,0,0,0]]),gen_length=8,block_length=4,steps=4,mask_id=3,eos_early_stop=False,remasking_strategy=strategy,threshold=.95)
    assert out.tokens.tolist()==[[1]*8]
    assert len(out.trace)==2*expected
    assert all(x["active_before"]>0 for x in out.trace)
    assert sum(final for final,_ in calls)==2
    assert len(calls)==2*expected+2

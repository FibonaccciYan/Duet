"""Test numerical differences separately from selection correctness."""
import math
from unittest.mock import patch
import pytest
import torch
from src.reference.sparse.qk_prefix import distances, rank, finish_union, prefix_indices as oracle
from src.optimized.sparse.prefix import prefix_indices as dispatch
from src.optimized.sparse.qk_tc_prefix import summaries, prefix_indices as tc

CUDA=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


def matrix_reference(q,k):
    return torch.cat([distances(q[0,h],k[0,h//(q.shape[1]//k.shape[1])])
                      for h in range(q.shape[1])],0)


def select_from_scores(scores,budget,strict):
    local=max(1,math.ceil(budget/scores.shape[0]))
    indices=torch.arange(scores.shape[1],device=scores.device).expand_as(scores).long()
    _,candidates=rank(scores,indices,local)
    return finish_union(candidates,scores.amin(0),budget,strict,None,local)


@CUDA
@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16])
@pytest.mark.parametrize("length",[17,63,65,257,2049,8192,32768])
@pytest.mark.parametrize("local",[1,4])
def test_gqa_tiles_and_score_arithmetic(dtype,length,local):
    torch.manual_seed(13)
    # Each GQA group has9 rows: every tile has tail rows. Non-contiguous inputs.
    q=torch.randn(1,3,6,256,device="cuda",dtype=dtype).transpose(1,2)[...,::2]
    k=torch.randn(1,2,length,256,device="cuda",dtype=dtype)[...,::2]
    m,i,s,full,kernel=summaries(q,k,local,return_scores=True)
    assert torch.isfinite(full).all()
    ref=matrix_reference(q,k)
    torch.testing.assert_close(full,ref,atol=3e-4,rtol=3e-5)
    for chunk in range(math.ceil(length/64)):
        lo,hi=chunk*64,min(length,(chunk+1)*64)
        ids=torch.arange(lo,hi,device=q.device).expand(full.shape[0],-1)
        scores,indices=rank(full[:,lo:hi],ids,min(local,hi-lo))
        assert torch.equal(scores,m[:,chunk*local:chunk*local+scores.shape[1]])
        assert torch.equal(indices,i[:,chunk*local:chunk*local+scores.shape[1]])
    assert torch.equal(s.amin(0),full.amin(0))
    # MMA instruction is direct evidence that this is actually Tensor Core code.
    ptx=kernel.asm["ptx"]
    assert "mma.sync" in ptx or "wgmma.mma_async" in ptx
    budget=min(length-1,full.shape[0]*local)
    for strict in (False,True):
        out=tc(q,k,budget,strict_budget=strict)
        assert torch.equal(out,select_from_scores(full,budget,strict))
        assert out.dtype==torch.int64 and out.unique().numel()==out.numel()
        assert torch.equal(out,out.sort().values)
        assert bool((out>=0).all() and (out<length).all())
        if strict:assert out.numel()==budget


@CUDA
@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16])
def test_exact_ties_near_ties_overflow_fill(dtype):
    q=torch.zeros(1,4,3,128,device="cuda",dtype=dtype)
    k=torch.zeros(1,2,259,128,device="cuda",dtype=dtype)
    for budget in (1,7,24):
        assert tc(q,k,budget).tolist()==list(range(budget))
    q[...,0]=1
    k[:,:,65,0]=1
    k[:,:,66,0]=1
    k[:,:,67,0]=1+torch.finfo(dtype).eps
    for strict in (False,True):
        for budget in (1,7,24):
            assert torch.equal(tc(q,k,budget,strict_budget=strict),
                               oracle(q,k,budget,strict_budget=strict))
    # Opposite-sign queries must retain the union, not just global token top-B.
    q[0,0,:,0]=-1;k[:,:,130,0]=-2
    soft=tc(q,k,1);hard=tc(q,k,1,strict_budget=True)
    assert soft.tolist()==[67,130] and hard.tolist()==[130]


@pytest.mark.parametrize("budget",[0,3,17,30])
def test_cpu_and_bypass_fallback(budget):
    q=torch.ones(1,2,3,7);k=torch.arange(119.).reshape(1,1,17,7)
    stats={}
    out=dispatch(q,k,budget,selection_stats=stats,selector="qk_tc")
    assert torch.equal(out,oracle(q,k,budget))
    assert stats["selector"]=="qk_tc"
    assert stats["execution_backend"]=="qk_reference"
    assert stats["score_definition"]=="negative_unscaled_dot_fp32_pairwise_tree"


@CUDA
def test_fallback_and_backend_stats():
    for dtype,dim,qlen,budget in [(torch.float32,128,3,2),
                                  (torch.float16,64,3,2),
                                  (torch.float16,128,1,16)]:
        q=torch.randn(1,1,qlen,dim,device="cuda",dtype=dtype)
        k=torch.randn(1,1,64,dim,device="cuda",dtype=dtype)
        with patch("src.optimized.sparse.qk_tc_prefix.summaries",side_effect=AssertionError):
            out=tc(q,k,budget)
        assert torch.equal(out,oracle(q,k,budget))
    q=torch.ones(1,4,3,128,device="cuda",dtype=torch.float16)
    k=torch.ones(1,2,257,128,device="cuda",dtype=torch.float16)
    st={}
    assert torch.equal(tc(q,k,6,selection_stats=st),tc(q,k,6))
    assert st["execution_backend"]=="tensorcore_fp32_accum"
    assert st["score_definition"]=="negative_unscaled_dot_tensorcore_fp32_accum"
    assert dispatch(q,k,6,selector="qk").tolist()==list(range(6))
    assert dispatch(q,k,6,selector="qk_tc").tolist()==list(range(6))

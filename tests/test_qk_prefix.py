"""QK has its own oracle; no assertion equates QK and Raw L1 algorithms."""
import math
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from src.reference.sparse.qk_prefix import (
    prefix_indices as reference, distances, finish_union)
from src.reference.sparse.sparse_ops import _prefix_indices, _compact_prefix_cache
from src.reference.sparse.config import LLaDASparseConfig, SDARSparseConfig


def oracle(q, k, budget, strict):
    """Independent scalar oracle, exactly representable integer test inputs."""
    length = k.shape[2]
    b = max(0, min(budget, length))
    if not b or b == length:
        return list(range(b))
    g = q.shape[1] // k.shape[1]
    matrix = [[-sum(float(q[0,h,t,c])*float(k[0,h//g,n,c]) for c in range(q.shape[-1]))
               for n in range(length)] for h in range(q.shape[1]) for t in range(q.shape[2])]
    local = max(1, math.ceil(b/len(matrix)))
    union = sorted(set(n for row in matrix
                       for n in sorted(range(length), key=lambda n:(row[n],n))[:local]))
    scores = [min(row[n] for row in matrix) for n in range(length)]
    if len(union) < b:
        remaining = [n for n in range(length) if n not in union]
        union += sorted(remaining, key=lambda n:(scores[n],n))[:b-len(union)]
    elif strict and len(union) > b:
        union = sorted(union, key=lambda n:(scores[n],n))[:b]
    return sorted(union)


@pytest.mark.parametrize("budget", [-1,0,1,2,3,7,8,20])
@pytest.mark.parametrize("strict", [False,True])
def test_cpu_gqa_math_and_budget(budget, strict):
    torch.manual_seed(3)
    q = torch.randint(-3,4,(1,4,3,5)).float()
    k = torch.randint(-3,4,(1,2,8,5)).float()
    stats = {}
    out = reference(q,k,budget,chunk_size=3,strict_budget=strict,selection_stats=stats)
    assert out.tolist() == oracle(q,k,budget,strict)
    assert out.dtype==torch.int64 and out.device==q.device
    assert stats["selector"]=="qk"
    assert stats["budget"] == min(max(budget,0),8)
    assert stats["selected_size"]==out.numel()


def test_direction_not_absolute_cosine_or_group_mean():
    q = torch.tensor([1.,-1.]).reshape(1,2,1,1)
    k = torch.tensor([2.,-3.,1.]).reshape(1,1,3,1)
    assert reference(q,k,1).tolist()==[0,1]  # each Q chooses its own winner
    assert reference(q,k,1,strict_budget=True).tolist()==[1]
    assert reference(q[:,:1],k,1).tolist()==[0]  # abs(dot) would choose1


def test_union_cannot_be_replaced_by_global_topk():
    # Global minima would prefer token1 to token2, but token2 is a query winner.
    q = torch.tensor([1.,0.,0.,1.]).reshape(1,1,2,2)
    k = torch.tensor([[100.,0.],[90.,0.],[0.,1.]]).reshape(1,1,3,2)
    assert reference(q,k,2).tolist()==[0,2]
    assert reference(q,k,2,strict_budget=True).tolist()==[0,2]


@pytest.mark.parametrize("local", [1,2])
def test_all_ties_and_fill(local):
    q=torch.zeros(1,2,3,7); k=torch.zeros(1,1,41,7)
    b=6*local
    for chunk in (1,7,32,256):
        for strict in (False,True):
            out=reference(q,k,b,chunk_size=chunk,strict_budget=strict)
            assert out.tolist()==list(range(b))


def test_strict_union_tie_order():
    stats={}
    out=finish_union(torch.tensor([4,2,3]),torch.zeros(6),2,True,stats,1)
    assert out.tolist()==[2,3]
    assert stats["union_size"]==3


def test_empty_prefix_and_invalid_shapes():
    q=torch.zeros(1,2,1,7); k=torch.zeros(1,1,0,7)
    assert reference(q,k,3).numel()==0
    with pytest.raises(ValueError):reference(q.repeat(2,1,1,1),k,3)
    with pytest.raises(ValueError):reference(q,k.repeat(1,3,1,1),3)
    with pytest.raises(ValueError):reference(q,k[:,:,:,:6],3)
    with pytest.raises(ValueError):_prefix_indices(q,k,3,selector="typo")


def test_config_defaults_and_model_local_dispatch():
    from src.reference.sparse.api import patch_model
    assert LLaDASparseConfig().prefix_selector==SDARSparseConfig().prefix_selector=="raw_l1"
    import src.reference.sparse.sparse_ops as ops
    original_dispatch=ops._prefix_indices
    for family,typ,target in (("sdar","sdar","patch_sdar_model"),("llada","llada2_moe","patch_llada_model")):
        a=SimpleNamespace(config=SimpleNamespace(model_type=typ))
        b=SimpleNamespace(config=SimpleNamespace(model_type=typ))
        with patch("src.reference.sparse.api."+target) as fn:
            patch_model(a,model_name=family,prefix_selector="qk",moe_expert_patch=False)
            assert fn.call_args.kwargs["prefix_selector"]=="qk"
            patch_model(b,model_name=family,moe_expert_patch=False)
            assert fn.call_args.kwargs["prefix_selector"]=="raw_l1"
        assert getattr(a.config,family+"_sparse_config")["prefix_selector"]=="qk"
        assert getattr(b.config,family+"_sparse_config")["prefix_selector"]=="raw_l1"
    assert ops._prefix_indices is original_dispatch
    with pytest.raises(ValueError):
        patch_model(a,model_name="llada",prefix_selector="unknown")


@pytest.mark.parametrize("rescreen", [False,True])
@pytest.mark.parametrize("strict", [False,True])
def test_candidate_mapping_and_shared_layer_pairs(rescreen, strict):
    k=torch.arange(6.).reshape(1,1,6,1);q=torch.ones(1,1,2,1)
    model=SimpleNamespace(config=SimpleNamespace(model_type="sdar",
        sdar_prefix_selector="qk",sdar_prefix_strict_budget=strict,
        sdar_prefix_share_layer_pairs=True),
        model=SimpleNamespace(rotary_emb=lambda q,p:(None,None)),
        _prefix_selection_stats=[])
    cache=SimpleNamespace(to_legacy_cache=lambda:((k,k+10),(k,k+20)))
    old=None if rescreen else (torch.tensor([0,2]),torch.tensor([0,2]))
    with patch("src.reference.sparse.sparse_ops._apply_rotary",lambda q,c,s:q):
        kv,ids=_compact_prefix_cache(model,cache,6,[None,q],None,2,2,
                                     previous_indices=old,previous_length=4)
    assert ids[0].tolist()==ids[1].tolist()==[4,5]
    assert kv[1][1].flatten().tolist()==[24.,25.]
    stats=model._prefix_selection_stats
    assert len(stats)==1 and stats[0]["layer"]==1
    assert stats[0]["candidate_length"]==(6 if rescreen else 4)
    assert stats[0]["prefix_length"]==6 and stats[0]["selector"]=="qk"

def test_two_models_compact_interleaved_without_global_mutation():
    import src.reference.sparse.sparse_ops as ops
    dispatch = ops._prefix_indices
    k=torch.tensor([1.,2.,3.,4.,5.,6.]).reshape(1,1,6,1)
    q=torch.ones(1,1,2,1)
    cache=SimpleNamespace(to_legacy_cache=lambda:((k,k),))
    def make(selector):
        return SimpleNamespace(config=SimpleNamespace(model_type="sdar",
            sdar_prefix_selector=selector,sdar_prefix_strict_budget=True),
            model=SimpleNamespace(rotary_emb=lambda q,p:(None,None)))
    a,b=make("qk"),make("raw_l1")
    def l1(q,k):
        return (q[0,0,:,None,:]-k[0,0,None,:,:]).abs().sum(-1)
    with patch.object(ops,"_apply_rotary",lambda q,c,s:q), patch.object(ops,"adamas_distances",l1):
        for model,expected in [(a,[4,5]),(b,[0,1]),(a,[4,5]),(b,[0,1])]:
            _,indices=_compact_prefix_cache(model,cache,6,[q],None,2,2)
            assert indices[0].tolist()==expected
    assert ops._prefix_indices is dispatch


@pytest.mark.parametrize("selector", ["adamas", "hadamard_qk"])
def test_legacy_selectors_private_dispatch(selector):
    import src.reference.sparse.sparse_ops as ops
    from src.optimized.sparse.prefix import prefix_indices
    q=torch.tensor([0.,10.2,20.5]).reshape(1,1,3,1)
    k=torch.tensor([0.,10.,20.,50.]).reshape(1,1,4,1)
    def dist(q,k):
        return (q.reshape(-1,1).float()[:,None,:]-k[0,0].float()[None,:,:]).abs().sum(-1)
    fn=ops._adamas_prefix_indices if selector=="adamas" else ops._hadamard_qk_prefix_indices
    original=ops._prefix_indices
    with patch.object(ops,"_hadamard_transform",lambda x:x), patch.object(ops,"adamas_distances",dist):
        expected=fn(q,k,2)
        stats={}
        actual=prefix_indices(q,k,2,selection_stats=stats,selector=selector)
        assert torch.equal(expected,actual)
        assert stats["selector"]==selector
        strict=prefix_indices(q,k,2,strict_budget=True,selector=selector)
        assert strict.numel()==2
    assert ops._prefix_indices is original


CUDA=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


@CUDA
@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16])
@pytest.mark.parametrize("length",[1,17,31,32,33,257,1025,2048,2065])
def test_gpu_indices_and_fallback(dtype,length):
    from src.optimized.sparse.qk_prefix import prefix_indices
    torch.manual_seed(length)
    # 3 query rows = partial BQ group; GQA tested separately.
    q=torch.randn(1,1,3,128,device="cuda",dtype=dtype)
    k=torch.randn(1,1,length,128,device="cuda",dtype=dtype)
    for budget in (0,1,3,7,length,length+1):
        for strict in (False,True):
            a,b={},{}
            expected=reference(q,k,budget,strict_budget=strict,selection_stats=a)
            actual=prefix_indices(q,k,budget,strict_budget=strict,selection_stats=b)
            assert torch.equal(actual,expected),(dtype,length,budget,strict)
            assert a==b
            assert torch.equal(actual,prefix_indices(q,k,budget,strict_budget=strict))


@CUDA
@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16])
@pytest.mark.parametrize("tie",[False,True])
def test_exact_scores_gqa_and_strides(dtype,tie):
    from src.optimized.sparse.qk_prefix import summaries,prefix_indices
    torch.manual_seed(99)
    q=torch.randn(1,3,4,256,device="cuda",dtype=dtype).transpose(1,2)[...,::2]
    k=torch.randn(1,2,67,256,device="cuda",dtype=dtype)[...,::2]
    if tie:
        k.zero_();k[:,:,0,0]=1.;k[:,:,32,0]=1.
        # Adjacent representable values exercise near ties too.
        k[:,:,33,0]=1. + torch.finfo(dtype).eps
    m,i,s=summaries(q,k)
    matrix=torch.cat([distances(q[0,h],k[0,h//2]) for h in range(4)],0)
    for chunk in range(m.shape[1]):
        lo,hi=chunk*32,min((chunk+1)*32,k.shape[2])
        scores,idx=matrix[:,lo:hi].min(-1)
        assert torch.equal(m[:,chunk],scores), (m[:,chunk]-scores).abs().max()
        assert torch.equal(i[:,chunk],idx+lo)
    assert torch.equal(s.amin(0),matrix.amin(0))
    for strict in (False,True):
        assert torch.equal(prefix_indices(q,k,5,strict_budget=strict),
                           reference(q,k,5,strict_budget=strict))


@CUDA
@pytest.mark.parametrize("dim",[1,32,64,129])
def test_non128_qk_reference_fallback(dim):
    from src.optimized.sparse.qk_prefix import prefix_indices
    q=torch.randn(1,4,3,dim,device="cuda",dtype=torch.float16)
    k=torch.randn(1,2,73,dim,device="cuda",dtype=torch.float16)
    assert torch.equal(prefix_indices(q,k,8),reference(q,k,8))


@CUDA
def test_large_reference_candidates():
    from src.optimized.sparse.qk_prefix import prefix_indices
    for length in (8192,16384,32768):
        q=torch.zeros(1,1,1,128,device="cuda",dtype=torch.bfloat16);q[...,0]=1
        k=torch.zeros(1,1,length,128,device="cuda",dtype=q.dtype)
        k[0,0,-1,0]=2
        expected=torch.tensor([0,1,2,length-1],device="cuda")
        assert torch.equal(prefix_indices(q,k,4),expected)

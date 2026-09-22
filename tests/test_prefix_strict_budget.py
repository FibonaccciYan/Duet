from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from src.reference.sparse.sparse_ops import _raw_l1_prefix_indices, _distance_prefix_indices, _compact_prefix_cache
from src.reference.sparse.config import LLaDASparseConfig, SDARSparseConfig
from src.reference.sparse.selection_stats import summarize


def distances(q, k):
    q = q.reshape(-1, q.shape[-1]).float()
    k = k[0, 0].float()
    return (q[:, None] - k[None]).abs().sum(-1)


@pytest.mark.parametrize("budget", [0, 2, 3, 4, 9])
def test_budget_and_complete_stats(budget):
    q = torch.tensor([0., 10.2, 20.5]).reshape(1, 1, 3, 1)
    k = torch.tensor([0., 10., 20., 50.]).reshape(1, 1, 4, 1)
    with patch("src.reference.sparse.sparse_ops.adamas_distances", distances):
        stats = {}
        strict = _raw_l1_prefix_indices(q, k, budget, strict_budget=True, selection_stats=stats)
        soft = _raw_l1_prefix_indices(q, k, budget)
        assert strict.numel() == min(budget, 4)
        assert set(stats) >= {"budget", "local_budget", "union_size", "selected_size",
                             "strict_budget", "candidate_length", "bypassed"}
        if budget == 2:
            assert soft.tolist() == [0, 1, 2]
            assert strict.tolist() == [0, 1]
            assert stats["union_size"] == 3
        assert torch.equal(soft, _raw_l1_prefix_indices(q, k, budget, strict_budget=False))


def test_shortfall_and_chunk_invariance():
    q = torch.zeros(1, 1, 3, 1)
    k = torch.arange(5.).reshape(1, 1, 5, 1)
    with patch("src.reference.sparse.sparse_ops.adamas_distances", distances):
        for chunk in (1, 2, 5):
            stats = {}
            out = _distance_prefix_indices(q, k, 2, chunk, strict_budget=True,
                                           selection_stats=stats)
            assert out.tolist() == [0, 1]
            assert stats["union_size"] == 1
            assert stats["selected_size"] == 2


def test_defaults_and_summary():
    assert not LLaDASparseConfig().prefix_strict_budget
    assert not SDARSparseConfig().prefix_strict_budget
    records = [dict(prefix_length=100, candidate_length=10, local_budget=1,
                    budget=2, union_size=u, selected_size=2, strict_budget=True)
               for u in (1, 2, 3)]
    report = summarize(records)
    assert report["overflow"] == report["shortfall"] == 1
    assert report["strict_final_violations"] == 0
    assert report["union_size"]["mean"] == 2
    assert summarize([])["selected_size"] is None


def test_strict_config_forwarding():
    from src.reference.sparse.api import patch_model
    for family, model_type, target in (
            ("llada", "llada2_moe", "patch_llada_model"),
            ("sdar", "sdar", "patch_sdar_model")):
        model = SimpleNamespace(config=SimpleNamespace(model_type=model_type))
        with patch("src.reference.sparse.api." + target) as patched:
            patch_model(model, model_name=family, prefix_strict_budget=True,
                        moe_expert_patch=False)
        assert patched.call_args.kwargs["prefix_strict_budget"] is True
        assert getattr(model.config, family + "_sparse_config")["prefix_strict_budget"]


def test_compactor_candidate_mapping():
    k = torch.arange(6.).reshape(1, 1, 6, 1)
    cache = SimpleNamespace(to_legacy_cache=lambda: ((k, k+10),))
    model = SimpleNamespace(
        config=SimpleNamespace(model_type="sdar", sdar_prefix_strict_budget=True),
        model=SimpleNamespace(rotary_emb=lambda q, pos: (None, None)),
        _prefix_selection_stats=[])
    q = torch.zeros(1, 1, 3, 1)
    seen = {}
    def select(query, candidates, budget, chunk_size, **kwargs):
        seen["keys"] = candidates.flatten().tolist()
        assert kwargs["strict_budget"]
        kwargs["selection_stats"].update(candidate_length=4, budget=2, local_budget=1,
            union_size=3, selected_size=2, strict_budget=True, bypassed=False)
        return torch.tensor([1, 3])
    with patch("src.reference.sparse.sparse_ops._apply_rotary", lambda q, c, s: q), \
         patch("src.reference.sparse.sparse_ops._prefix_indices", select):
        compact, indices = _compact_prefix_cache(
            model, cache, 6, [q], None, 2, 1024,
            previous_indices=(torch.tensor([0, 2]),), previous_length=4)
    assert seen["keys"] == [0., 2., 4., 5.]
    assert indices[0].tolist() == [2, 5]
    assert compact[0][0].flatten().tolist() == [2., 5.]
    assert model._prefix_selection_stats[0]["prefix_length"] == 6


def test_small_dim_v3_safely_uses_reference():
    from src.optimized.sparse.prefix import prefix_indices
    q = torch.zeros(1, 1, 3, 32)
    k = torch.zeros(1, 1, 64, 32)
    with patch("src.optimized.sparse.prefix.reference", return_value=torch.tensor([0])) as ref:
        prefix_indices(q, k, 2, strict_budget=True)
        assert ref.call_args.kwargs["strict_budget"]


def test_compactor_full_prefix_stats():
    k = torch.zeros(1, 1, 2, 1)
    cache = SimpleNamespace(to_legacy_cache=lambda: ((k, k), (k, k)))
    model = SimpleNamespace(config=SimpleNamespace(model_type="sdar",
        sdar_prefix_strict_budget=True, sdar_prefix_share_layer_pairs=True),
        _prefix_selection_stats=[])
    result, positions = _compact_prefix_cache(model, cache, 2, None, None, 5, 1024)
    assert len(result) == 2
    assert len(model._prefix_selection_stats) == 1
    stats = model._prefix_selection_stats[0]
    assert stats["bypassed"] and stats["layer"] == 1 and stats["selected_size"] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("length", [32, 2064])
def test_v3_fused_and_fallback(length):
    from src.optimized.sparse.prefix import prefix_indices
    q = torch.zeros(1,1,3,128,device="cuda",dtype=torch.float16)
    k = torch.zeros(1,1,length,128,device="cuda",dtype=torch.float16)
    q[0,0,:,0] = torch.tensor([0.,10.2,20.5],device="cuda")
    k[0,0,:,0] = torch.arange(length,device="cuda",dtype=torch.float16)*10
    for strict in (False, True):
        a, b = {}, {}
        ref = _raw_l1_prefix_indices(q,k,2,strict_budget=strict,selection_stats=a)
        out = prefix_indices(q,k,2,strict_budget=strict,selection_stats=b)
        assert torch.equal(ref,out)
        assert all(b[name] == value for name, value in a.items())
        assert b["selector"] == "raw_l1"
        assert b["score_definition"] == "raw_l1_legacy"
        assert out.numel() == (2 if strict else 3)

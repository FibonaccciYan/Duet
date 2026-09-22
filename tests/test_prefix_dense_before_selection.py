"""Full shallow KV, preserved pair representatives and multiblock mappings."""
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from src.reference.sparse.sparse_ops import _compact_prefix_cache
from src.reference.sparse.selection_stats import summarize


@pytest.mark.parametrize('family,boundary,shared', [
    ('llada2_moe', 2, False), ('sdar', 6, False), ('sdar', 6, True),
    ('sdar', 5, True), ('sdar', 1, True), ('sdar', 0, True),
    ('sdar', 8, True)])
@pytest.mark.parametrize('refresh', [False, True])
@pytest.mark.parametrize('budget', [2, 20])
def test_cache_semantics(family, boundary, shared, refresh, budget):
    n, length = 8, 12
    keys = [torch.arange(length * 2.).reshape(1, 1, length, 2) + 100*i for i in range(n)]
    cache = SimpleNamespace(to_legacy_cache=lambda: tuple((k, k+1) for k in keys))
    cfg = dict(model_type=family, **{('llada' if family == 'llada2_moe' else family)+'_prefix_strict_budget': True},
               sdar_prefix_share_layer_pairs=shared)
    model = SimpleNamespace(config=SimpleNamespace(**cfg), _prefix_selection_stats=[],
        model=SimpleNamespace(rotary_emb=lambda q,p: (torch.zeros(1), torch.zeros(1))))
    queries = [None if shared and i%2 == 0 else torch.full((1,1,2,2),float(i)) for i in range(n)]
    seen=[]
    def select(q,k,b,c,**kwargs):
        seen.append(int(q[0,0,0,0]))
        kwargs['selection_stats'].update(candidate_length=k.shape[2], budget=b,
            local_budget=1, union_size=b, selected_size=b, strict_budget=True, bypassed=False)
        return torch.tensor([0, k.shape[2]-1])
    previous=tuple(torch.tensor([1,3]) for _ in keys)
    with patch('src.reference.sparse.sparse_ops._apply_rotary',lambda q,c,s:q), \
         patch('src.reference.sparse.sparse_ops._prefix_indices',select):
        result, positions = _compact_prefix_cache(model,cache,length,queries,None,budget,256,
            None if refresh else previous,8,prefix_start_layer=boundary)
    for layer, ((k,v),idx) in enumerate(zip(result,positions)):
        if layer < boundary or budget>=length:
            assert k.data_ptr()==keys[layer].data_ptr()  # zero-copy full KV
            assert torch.equal(idx,torch.arange(length))
        else:
            assert idx.tolist()==([0,11] if refresh else [1,11])
        assert torch.equal(k,keys[layer].index_select(2,idx))
        assert torch.equal(v,(keys[layer]+1).index_select(2,idx))
    expected=[] if budget>=length else [i for i in range(n) if i>=boundary and (not shared or i%2)]
    assert seen==expected
    assert summarize(model._prefix_selection_stats)['strict_final_violations']==0
    dense_rows=[r for r in model._prefix_selection_stats if r.get('budget_applied') is False]
    assert [r['layer'] for r in dense_rows]==list(range(boundary))


@pytest.mark.parametrize('family', ['llada2_moe','sdar'])
def test_default_and_config_forwarding(family):
    from src.reference.sparse.api import patch_model
    from src.reference.sparse.config import LLaDASparseConfig, SDARSparseConfig
    assert not LLaDASparseConfig().prefix_dense_before_query_selection
    assert not SDARSparseConfig().prefix_dense_before_query_selection
    model=SimpleNamespace(config=SimpleNamespace(model_type=family))
    with patch('src.reference.sparse.api.patch_'+('llada' if family == 'llada2_moe' else family)+'_model') as impl:
        patch_model(model,model_name=('llada' if family == 'llada2_moe' else family),moe_expert_patch=False,prefix_dense_before_query_selection=True)
    assert impl.call_args.kwargs['prefix_dense_before_query_selection']
    assert getattr(model.config,('llada' if family == 'llada2_moe' else family)+'_sparse_config')['prefix_dense_before_query_selection']

"""CPU lifecycle tests; GPU numerical/real-generation tests live in performance scripts."""
import torch
from src.optimized.losa.workspace import Workspace


def test_workspace_reuses_matching_buffers_and_separates_shapes():
    ws=Workspace()
    x=torch.randn(4,8)
    a=ws.get("scores",x.shape,x)
    assert ws.get("scores",x.shape,x) is a
    assert ws.get("scores",(8,4),x) is not a
    assert ws.get("scores",x.shape,x,torch.float64) is not a
    other=Workspace()
    assert other.get("scores",x.shape,x) is not a
    assert ws.allocations==3


def test_topk_ties_and_repeated_calls_match_reference():
    ws=Workspace()
    for x in (torch.zeros(32),torch.arange(32).float(),
              torch.tensor([1.,2.,2.,0.]*8)):
        expected=x.topk(5,sorted=False).indices.sort().values
        assert torch.equal(ws.active(x,5),expected)
    allocations=ws.allocations
    for _ in range(4):
        x=torch.randn(32)
        assert torch.equal(ws.active(x,5),x.topk(5,sorted=False).indices.sort().values)
    assert ws.allocations==allocations


def test_page_topk_identical_including_ties():
    ws=Workspace()
    for x in (torch.zeros(5,8,64),torch.randn(5,8,64)):
        assert torch.equal(ws.topk(x,16,"pages"),x.topk(16,dim=-1).indices)

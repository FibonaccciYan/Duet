from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
import src.optimized.losa.generation as g


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(mask_token_id=99, eos_token_id=10)
        self.generation_config = SimpleNamespace(eos_token_id=[10, 11])


def test_resolution():
    m = Model()
    assert g._sdar_eos_ids(m) == (10, 11)
    assert g._sdar_eos_ids(m, 0) == (0,)
    assert g._sdar_eos_ids(m, [12, 12, 10]) == (12, 10)
    assert g._sdar_eos_ids(m, []) == ()
    m.generation_config.eos_token_id = None
    assert g._sdar_eos_ids(m) == (10,)
    m.config.eos_token_id = None
    assert g._sdar_eos_ids(m) == ()


@pytest.mark.parametrize("steps", [16, 32])
@pytest.mark.parametrize("eos", [10, 11])
@pytest.mark.parametrize("early", [True, False])
def test_actual_driver_block_stop_and_trim(steps, eos, early):
    seen = []
    def prefix(model, x, length, positions, **kwargs):
        t = torch.zeros(1,1,length,1)
        return ((t,t),)
    def forward(model, family, ids, mask, positions, **kwargs):
        seen.append(int(positions[0,0]))
        logits = torch.full((*ids.shape,128),-100.)
        logits[...,7] = 100.
        # Prompt contains EOS too; it must not stop generation or be returned.
        logits[:,3,7] = -100.
        logits[:,3,eos] = 100.
        t = torch.zeros(1,1,int(positions[0,0])+ids.shape[1],1)
        return SimpleNamespace(logits=logits,
            past_key_values=SimpleNamespace(to_legacy_cache=lambda:((t,t),))), []
    with patch.object(g,"build_sdar_prefix_cache",prefix), patch.object(g,"model_forward",forward):
        out=g.block_diffusion_generate(Model(),family="sdar",
            inputs=torch.full((1,32),eos,dtype=torch.long),gen_length=64,
            block_length=32,steps=steps,mask_id=99,temperature=0.,
            remasking_strategy="sequential",eos_early_stop=early,use_losa=False)
    assert out.tokens.shape==(1,4 if early else 64)
    assert out.tokens[0,3]==eos
    assert (64 in seen) == (not early)


def test_earliest_of_multiple_ids():
    assert g._sdar_eos_positions(torch.tensor([7,11,7,10]),(10,11)).tolist()==[1,3]

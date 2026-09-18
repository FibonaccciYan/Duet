import types
import unittest
import torch
from src.kernels.optimized.rope_runtime import install,restore

def apply_rotary_pos_emb(q,k,cos,sin,position_ids=None,unsqueeze_dim=1):
    return q*cos.unsqueeze(unsqueeze_dim),k*cos.unsqueeze(unsqueeze_dim)

def _apply_rotary(q,cos,sin):return q*cos.unsqueeze(1)

def direct(self,q,k,c,s):return apply_rotary_pos_emb(q,k,c,s)

def sequential(self,query,key,cos,sin):
    query=_apply_rotary(query,cos,sin)
    key=_apply_rotary(key,cos,sin)
    return query,key

class RopeRuntimeTest(unittest.TestCase):
 def check(self,function):
  attn=types.SimpleNamespace();attn.forward=types.MethodType(function,attn)
  original=attn.forward;globals_before=dict(function.__globals__)
  model=types.SimpleNamespace(config=types.SimpleNamespace(model_type='sdar'),
        model=types.SimpleNamespace(layers=[types.SimpleNamespace(self_attn=attn)]))
  args=(torch.randn(1,2,3,128),torch.randn(1,1,3,128),torch.randn(1,3,128),torch.zeros(1,3,128))
  expected=original(*args)
  self.assertEqual(install(model),1)
  actual=attn.forward(*args)
  self.assertTrue(all(torch.equal(a,b) for a,b in zip(actual,expected)))
  self.assertIs(function.__globals__['apply_rotary_pos_emb'],globals_before['apply_rotary_pos_emb'])
  self.assertIs(function.__globals__['_apply_rotary'],globals_before['_apply_rotary'])
  self.assertEqual(install(model),0)
  restore(model)
  self.assertIs(attn.forward,original)
  self.assertFalse(attn._versioned_rope_enabled)
 def test_direct_fallback_and_restore(self):self.check(direct)
 def test_sequential_fallback_and_restore(self):self.check(sequential)
if __name__=='__main__':unittest.main()

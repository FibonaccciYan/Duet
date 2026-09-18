import types
import unittest
import torch
from transformers.cache_utils import DynamicCache
from src.kernels.optimized.prefix_cache_view import PrefixViewCache, install

def _cached_forward(prefix):
    return DynamicCache.from_legacy_cache(prefix)

def _driver(self,prefix):
    return _cached_forward(prefix)

class PrefixCacheViewTest(unittest.TestCase):
 def test_append_and_source_ownership(self):
  prefix=tuple((torch.randn(1,2,16,8),torch.randn(1,2,16,8)) for _ in range(2))
  snapshots=tuple((k.clone(),v.clone()) for k,v in prefix)
  actual=PrefixViewCache.from_legacy_cache(prefix)
  ref=DynamicCache.from_legacy_cache(prefix)
  for i,(k,v) in enumerate(prefix):
   self.assertEqual(actual[i][0].data_ptr(),k.data_ptr())
   self.assertEqual(actual[i][1].data_ptr(),v.data_ptr())
   q,r=torch.randn(1,2,4,8),torch.randn(1,2,4,8)
   a,b=actual.update(q,r,i);x,y=ref.update(q,r,i)
   self.assertTrue(torch.equal(a,x) and torch.equal(b,y))
   self.assertTrue(torch.equal(k,snapshots[i][0]) and torch.equal(v,snapshots[i][1]))
 def test_install_does_not_mutate_original_helpers(self):
  model=types.SimpleNamespace();model.generate=types.MethodType(_driver,model)
  other=types.SimpleNamespace();other.generate=types.MethodType(_driver,other)
  prefix=((torch.randn(1,2,16,8),torch.randn(1,2,16,8)),)
  install(model)
  self.assertIsInstance(model.generate(prefix),PrefixViewCache)
  self.assertIs(type(other.generate(prefix)),DynamicCache)
  self.assertIs(_cached_forward.__globals__['DynamicCache'],DynamicCache)
 def test_empty(self):
  self.assertEqual(PrefixViewCache.from_legacy_cache(None).get_seq_length(),0)
  self.assertEqual(PrefixViewCache.from_legacy_cache(()).get_seq_length(),0)
if __name__=='__main__':unittest.main()

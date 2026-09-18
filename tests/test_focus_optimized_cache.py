import unittest
from unittest.mock import patch
import torch
from transformers.cache_utils import DynamicCache
from src.optimized.focus.model import _make_cache

class CacheOwnerTest(unittest.TestCase):
 def test_existing_owners_do_not_construct_copy(self):
  owner=object()
  with patch('src.optimized.focus.model.DynamicCache',side_effect=AssertionError('unneeded allocation')):
   self.assertIs(_make_cache((),block_cache=owner),owner)
   self.assertIs(_make_cache((),append_cache=owner),owner)
  with self.assertRaises(ValueError):_make_cache((),owner,owner)
 def test_view_append_preserves_prefix(self):
  prefix=tuple((torch.randn(1,2,16,8),torch.randn(1,2,16,8)) for _ in range(3))
  saved=tuple((k.clone(),v.clone()) for k,v in prefix)
  cache=_make_cache(prefix);ref=DynamicCache.from_legacy_cache(prefix)
  for i,(k,v) in enumerate(prefix):
   self.assertEqual(cache[i][0].data_ptr(),k.data_ptr())
   self.assertEqual(cache[i][1].data_ptr(),v.data_ptr())
   q,r=torch.randn(1,2,4,8),torch.randn(1,2,4,8)
   a,b=cache.update(q,r,i);x,y=ref.update(q,r,i)
   self.assertTrue(torch.equal(a,x) and torch.equal(b,y))
   self.assertTrue(torch.equal(k,saved[i][0]) and torch.equal(v,saved[i][1]))
if __name__=='__main__':unittest.main()

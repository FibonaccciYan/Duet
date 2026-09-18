"""Check zero-copy prefix ownership and DynamicCache append compatibility."""
import unittest
import torch
from transformers.cache_utils import DynamicCache
from src.optimized.losa.generation import prefix_cache_view

class PrefixViewTest(unittest.TestCase):
    def test_alias_append_and_original_ownership(self):
        prefix=tuple((torch.randn(1,2,16,8),torch.randn(1,2,16,8)) for _ in range(3))
        saved=tuple((k.clone(),v.clone()) for k,v in prefix)
        actual=prefix_cache_view(prefix)
        reference=DynamicCache.from_legacy_cache(prefix)
        for i,(k,v) in enumerate(prefix):
            self.assertEqual(actual.layers[i].keys.data_ptr(),k.data_ptr())
            self.assertEqual(actual.layers[i].values.data_ptr(),v.data_ptr())
            for _ in range(2):
                q,r=torch.randn(1,2,4,8),torch.randn(1,2,4,8)
                ak,av=actual.update(q,r,i)
                bk,bv=reference.update(q,r,i)
                self.assertTrue(torch.equal(ak,bk) and torch.equal(av,bv))
            self.assertTrue(torch.equal(k,saved[i][0]) and torch.equal(v,saved[i][1]))
        self.assertEqual(actual.get_seq_length(),24)
    def test_empty(self):
        self.assertEqual(prefix_cache_view(()).get_seq_length(),0)

if __name__=='__main__': unittest.main()

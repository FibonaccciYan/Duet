import unittest
import torch
from transformers.cache_utils import DynamicCache
from src.kernels.optimized.block_scratch_cache import BlockScratchCache


class ScratchCacheTest(unittest.TestCase):
    def test_variable_suffix_replaces_not_appends(self):
        prefix=tuple((torch.randn(1,2,37,8),torch.randn(1,2,37,8)) for _ in range(3))
        original=tuple((k.clone(),v.clone()) for k,v in prefix)
        cache=BlockScratchCache(prefix,32)
        pointers={}
        for count in (32,17,5,28,1,32):
            reference=DynamicCache.from_legacy_cache(prefix)
            for layer in range(3):
                k=torch.randn(1,count,2,8).transpose(1,2)
                v=torch.randn_like(k)
                expected=reference.update(k,v,layer)
                actual=cache.update(k,v,layer)
                for a,b in zip(expected,actual): self.assertTrue(torch.equal(a,b))
                self.assertEqual(actual[0].shape[-2],37+count)
                if layer in pointers:self.assertEqual(actual[0].data_ptr(),pointers[layer])
                pointers[layer]=actual[0].data_ptr()
            for (k,v),(pk,pv) in zip(prefix,original):
                self.assertTrue(torch.equal(k,pk) and torch.equal(v,pv))

    def test_zero_prefix_and_capacity(self):
        cache=BlockScratchCache((),32)
        k=torch.ones(1,2,5,8)
        self.assertEqual(cache.update(k,k,0)[0].shape[-2],5)
        with self.assertRaises(ValueError):
            cache.update(torch.ones(1,2,33,8),torch.ones(1,2,33,8),0)


if __name__=="__main__":unittest.main()

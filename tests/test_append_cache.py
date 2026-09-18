import unittest
import torch
from src.kernels.optimized.append_cache import AppendCache


class AppendCacheTest(unittest.TestCase):
    def test_growth_independent_layers_and_old_prefix_views(self):
        cache = AppendCache(initial_capacity=4)
        references = [[], []]
        snapshots = []
        with torch.inference_mode():
            for count in (3, 1, 5, 7):
                for layer in (0, 1):
                    # Noncontiguous model-style input.
                    key = torch.arange(count*2*8).view(1,count,2,8).transpose(1,2).float()+layer
                    value = key + 0.5
                    references[layer].append((key.clone(), value.clone()))
                    k,v = cache.update(key,value,layer)
                    expected_k = torch.cat([x[0] for x in references[layer]],dim=2)
                    expected_v = torch.cat([x[1] for x in references[layer]],dim=2)
                    self.assertTrue(torch.equal(k, expected_k))
                    self.assertTrue(torch.equal(v, expected_v))
                    self.assertEqual(cache.get_seq_length(layer), expected_k.shape[2])
                    snapshots.append((k,k.clone(),v,v.clone()))
                for k,expected_k,v,expected_v in snapshots:
                    self.assertTrue(torch.equal(k, expected_k))
                    self.assertTrue(torch.equal(v, expected_v))
        self.assertEqual(len(cache.to_legacy_cache()), 2)

    def test_in_capacity_preserves_storage_pointer(self):
        cache = AppendCache(initial_capacity=16)
        with torch.inference_mode():
            x=torch.ones(1,2,4,8)
            k,_=cache.update(x,x,0)
            pointer=k.data_ptr()
            k,_=cache.update(x,x,0)
            self.assertEqual(pointer,k.data_ptr())

    def test_grad_inputs_rejected(self):
        cache=AppendCache()
        x=torch.ones(1,1,1,8,requires_grad=True)
        with self.assertRaises(ValueError):
            cache.update(x,x,0)


if __name__=="__main__":
    unittest.main()

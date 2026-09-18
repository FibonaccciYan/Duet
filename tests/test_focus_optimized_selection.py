import unittest
import torch
from src.reference.focus.algorithm import select_retained_positions as reference
from src.optimized.focus.algorithm import select_retained_positions as optimized


class FocusOptimizedSelectionTest(unittest.TestCase):
    def check_cases(self, device):
        torch.manual_seed(17)
        for length in (1,5,17,32):
            for density in (0.0,0.25,0.75,1.0):
                tokens=torch.arange(length,device=device)
                masks=torch.rand(length,device=device)<density
                tokens=tokens.masked_fill(masks,99)
                n=int(masks.sum())
                first=torch.randn(n,device=device)
                for second in (first.clone(),torch.randn_like(first)):
                    for alpha,average,progress in ((1.,1.,-1),(1.5,2.,length//2),(3.,100.,length)):
                        args=(tokens,99,first,second)
                        kw=dict(alpha=alpha,average_decoded_tokens=average,block_progress=progress)
                        self.assertTrue(torch.equal(reference(*args,**kw),optimized(*args,**kw)))

    def test_cpu_all_rules(self):
        self.check_cases("cpu")

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA required")
    def test_cuda_all_rules(self):
        self.check_cases("cuda")

    def test_invalid_inputs(self):
        for function in (reference,optimized):
            with self.assertRaises(ValueError):
                function(torch.ones(1,4),99,torch.empty(0),torch.empty(0),
                         alpha=1.5,average_decoded_tokens=1.)
            with self.assertRaises(ValueError):
                function(torch.full((4,),99),99,torch.ones(3),torch.ones(4),
                         alpha=1.5,average_decoded_tokens=1.)
            with self.assertRaises(ValueError):
                function(torch.ones(4),99,torch.empty(0),torch.empty(0),
                         alpha=0.5,average_decoded_tokens=1.)


if __name__=="__main__":
    unittest.main()

import unittest
import torch
from scripts.original.performance.full_step_timing import instrument_sdar

def sample_loop(cur_x,side_effects):
    num_block=3
    for step in range(4):
        mask_index=cur_x==0
        finished=not bool(mask_index.any())
        if finished:
            side_effects.append('persist')
            break
        transfer_index=mask_index
        cur_x[transfer_index]=1
    return cur_x

def unsupported_loop(x):return x

class Clock:
    def __init__(self,x):self.x=x;self.events=[]
    def begin(self,block,step):
        self.events.append(('begin',block,step,self.x.clone()))
        return block,step
    def end(self,stamp,phase):self.events.append((phase,*stamp,self.x.clone()))

class FullStepTimingTest(unittest.TestCase):
    def test_observes_scatter_and_preserves_behavior(self):
        actual=torch.tensor([[0,2,0]]);expected=actual.clone()
        clock=Clock(actual);effects=[];refeffects=[]
        wrapped=instrument_sdar(sample_loop,clock)
        self.assertTrue(torch.equal(wrapped(actual,effects),sample_loop(expected,refeffects)))
        self.assertEqual(effects,refeffects)
        self.assertEqual([e[0] for e in clock.events],['begin','denoise','begin','finalize'])
        self.assertTrue(torch.equal(clock.events[1][3],torch.tensor([[1,2,1]])))
    def test_rejects_unknown_loop(self):
        with self.assertRaises(ValueError):instrument_sdar(unsupported_loop,Clock(torch.zeros(1)))
if __name__=='__main__':unittest.main()

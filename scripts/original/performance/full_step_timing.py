"""Benchmark-only SDAR iteration timing without editing original drivers.

Instrument a private AST clone of the inspected function. The only inserted
operations are clock calls before mask computation and after token scatter,
or after final cache persistence. Fail closed if the expected loop changes.
"""
import ast
import inspect
import textwrap
import time
import torch

class StepClock:
    def __init__(self):self.events=[]
    def begin(self,block,step):
        begin=torch.cuda.Event(enable_timing=True)
        begin.record()
        return block,step,begin,time.perf_counter()
    def end(self,stamp,phase):
        end=torch.cuda.Event(enable_timing=True)
        end.record()
        self.events.append((*stamp,end,time.perf_counter(),phase))
    def rows(self):
        return [dict(block=b,step=s,phase=p,cuda_ms=start.elapsed_time(end),
                     host_ms=(he-hs)*1000)
                for b,s,start,hs,end,he,p in self.events]

def instrument_sdar(function,clock):
    function=inspect.unwrap(function)
    tree=ast.parse(textwrap.dedent(inspect.getsource(function)))
    fn=tree.body[0]
    if not isinstance(fn,ast.FunctionDef):raise ValueError('expected a function')
    fn.decorator_list=[]
    loops=[n for n in ast.walk(fn) if isinstance(n,ast.For)
           and isinstance(n.target,ast.Name) and n.target.id=='step'
           and any(isinstance(c,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='mask_index'
                   for t in c.targets) for c in n.body)]
    if len(loops)!=1:raise ValueError('expected exactly one SDAR denoising loop')
    loop=loops[0]
    assignments=[n for n in loop.body if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name)
                         and t.value.id=='cur_x' and isinstance(t.slice,ast.Name)
                         and t.slice.id=='transfer_index' for t in n.targets)]
    finished=[n for n in loop.body if isinstance(n,ast.If)
              and isinstance(n.test,ast.Name) and n.test.id=='finished']
    if len(assignments)!=1 or len(finished)!=1:raise ValueError('SDAR loop structure changed')
    if not isinstance(finished[0].body[-1],ast.Break):raise ValueError('unexpected finalization')
    loop.body.insert(0,ast.parse('_timing_stamp = _full_step_clock.begin(num_block, step)').body[0])
    loop.body.insert(loop.body.index(assignments[0])+1,
                     ast.parse('_full_step_clock.end(_timing_stamp, "denoise")').body[0])
    finished[0].body.insert(-1,ast.parse('_full_step_clock.end(_timing_stamp, "finalize")').body[0])
    namespace={**function.__globals__,'_full_step_clock':clock}
    exec(compile(ast.fix_missing_locations(tree),'<sdar-full-step-timing>','exec'),namespace)
    return namespace[function.__name__]

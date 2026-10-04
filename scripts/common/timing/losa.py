"""Timing instrumentation for the optimized LoSA generation driver."""
import ast
import contextlib
import inspect
import textwrap

import torch


class Clock:
    def __init__(self, timing=False):
        self.timing = timing
        self.counts = {}
        self.cached = 0
        self.events = []
        self.block = None

    def wrap_transfer(self, fn):
        def wrapped(*a, **kw):
            result = fn(*a, **kw)
            before = self.counts.get(self.block, 0)
            self.counts[self.block] = before + 1
            self.cached += int(before > 0)
            return result
        return wrapped

    @contextlib.contextmanager
    def iteration(self):
        before = self.counts.get(self.block, 0)
        if self.timing:
            a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            a.record()
        try:
            yield
        finally:
            if self.timing and self.counts.get(self.block, 0) > before:
                b.record()
                self.events.append((self.block, before, a, b))

    def rows(self):
        return [dict(block=block, loop_iteration=index, steady=index >= 1,
                     cuda_ms=a.elapsed_time(b)) for block, index, a, b in self.events]


def clone_losa(generate, clock):
    f = inspect.unwrap(generate)
    tree = ast.parse(textwrap.dedent(inspect.getsource(f)))
    fn = tree.body[0]
    fn.decorator_list = []
    blocks = [n for n in ast.walk(fn) if isinstance(n, ast.For)
              and isinstance(n.target, ast.Name) and n.target.id == "block_idx"]
    loops = [n for n in ast.walk(fn) if isinstance(n, ast.While)]
    assert len(blocks) == len(loops) == 1
    blocks[0].body.insert(0, ast.parse("_clock.block = block_idx").body[0])
    if clock.timing:
        wrap = ast.parse("with _clock.iteration():\n pass").body[0]
        wrap.body = loops[0].body
        loops[0].body = [wrap]
    ns = {**f.__globals__, "_clock": clock}
    for key in ("transfer_llada", "transfer_sdar"):
        ns[key] = clock.wrap_transfer(ns[key])
    exec(compile(ast.fix_missing_locations(tree), "<narrative80-losa>", "exec"), ns)
    cloned = torch.inference_mode()(ns[f.__name__])
    def call(*a, **kw):
        return cloned(*a, **kw).tokens
    return call

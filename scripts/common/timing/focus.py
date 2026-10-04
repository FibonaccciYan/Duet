"""Timing instrumentation for the optimized FOCUS generation driver."""
import ast
import contextlib
import inspect
import textwrap

import torch


class Clock:
    def __init__(self, timing=False, event_factory=None):
        self.timing, self.block = timing, None
        self.counts, self.events = {}, []
        self.event_factory = event_factory or (
            lambda: torch.cuda.Event(enable_timing=True))

    def wrap_transfer(self, function):
        def call(*args, **kwargs):
            output = function(*args, **kwargs)
            self.counts[self.block] = self.counts.get(self.block, 0) + 1
            return output
        return call

    @contextlib.contextmanager
    def iteration(self):
        block = self.block
        index = self.counts.get(block, 0)
        if self.timing:
            begin, end = self.event_factory(), self.event_factory()
            begin.record()
        try:
            yield
        finally:
            # Handles LLaDA's final editing pass that breaks after transfer.
            # Excludes loop-exit checks without an executed transfer.
            after = self.counts.get(block, 0)
            if after != index:
                if after != index + 1:
                    raise RuntimeError("more than one transfer in an iteration")
                if self.timing:
                    end.record()
                    self.events.append((block, index, begin, end))

    def rows(self):
        return [dict(block=block, loop_iteration=index, noninitial=index >= 1,
                     cuda_ms=a.elapsed_time(b))
                for block, index, a, b in self.events]


def clone_driver(function, clock):
    f = inspect.unwrap(function)
    tree = ast.parse(textwrap.dedent(inspect.getsource(f)))
    fn = tree.body[0]
    fn.decorator_list = []
    blocks = [n for n in ast.walk(fn) if isinstance(n, ast.For)
              and isinstance(n.target, ast.Name) and n.target.id == "block_idx"]
    loops = [n for n in ast.walk(fn) if isinstance(n, ast.While)]
    if len(blocks) != 1 or len(loops) != 1:
        raise RuntimeError("FOCUS driver structure changed")
    blocks[0].body.insert(0, ast.parse("_clock.block = block_idx").body[0])
    if clock.timing:
        wrapper = ast.parse("with _clock.iteration():\n pass").body[0]
        wrapper.body = loops[0].body
        loops[0].body = [wrapper]
    ns = {**f.__globals__, "_clock": clock}
    for key in ("_selected_llada_transfer", "_selected_sdar_transfer"):
        ns[key] = clock.wrap_transfer(ns[key])
    exec(compile(ast.fix_missing_locations(tree), "<focus-full-dataset>", "exec"), ns)
    return torch.inference_mode()(ns[f.__name__])

"""Timing instrumentation for the dense and sparse block-diffusion drivers."""
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

    def transfer(self):
        self.counts[self.block] = self.counts.get(self.block, 0) + 1

    @contextlib.contextmanager
    def iteration(self, index):
        before = self.cached
        if self.timing:
            a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            a.record()
        yield
        if self.timing and self.cached != before:
            b.record()
            self.events.append((self.block, index, a, b))

    def rows(self):
        return [dict(block=block, loop_iteration=index, steady=index >= 2,
                     cuda_ms=a.elapsed_time(b)) for block, index, a, b in self.events]


def clone_sdar(generate, clock):
    """Private SDAR block driver: transfer counts exclude cache-finalize pass."""
    f = inspect.unwrap(generate)
    if hasattr(f, "__func__"):
        f = f.__func__
    original = inspect.unwrap(f.__globals__["block_diffusion_generate"])
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    fn = tree.body[0]
    fn.decorator_list = []
    blocks = [n for n in ast.walk(fn) if isinstance(n, ast.For)
              and isinstance(n.target, ast.Name) and n.target.id == "num_block"]
    loops = [n for n in ast.walk(fn) if isinstance(n, ast.For)
             and isinstance(n.target, ast.Name) and n.target.id == "step"]
    assert len(blocks) == len(loops) == 1
    blocks[0].body.insert(0, ast.parse("_clock.block = num_block").body[0])
    scatter = [s for s in loops[0].body if isinstance(s, ast.Assign)
               and any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                       and t.value.id == "cur_x" and isinstance(t.slice, ast.Name)
                       and t.slice.id == "transfer_index" for t in s.targets)]
    assert len(scatter) == 1
    pos = loops[0].body.index(scatter[0]) + 1
    loops[0].body[pos:pos] = ast.parse("_clock.cached += 1\n_clock.transfer()").body
    if clock.timing:
        wrap = ast.parse("with _clock.iteration(step):\n pass").body[0]
        wrap.body = loops[0].body
        loops[0].body = [wrap]
    ns = {**original.__globals__, "_clock": clock}
    exec(compile(ast.fix_missing_locations(tree), "<narrative80-sdar>", "exec"), ns)
    import types
    driver = types.FunctionType(
        f.__code__, {**f.__globals__, "block_diffusion_generate": ns[original.__name__]},
        f.__name__, f.__defaults__, f.__closure__)
    driver.__kwdefaults__ = f.__kwdefaults__
    return torch.inference_mode()(driver)


def clone_driver(generate, clock):
    """Count actual transfers; time whole cached iterations in a separate pass."""
    f = inspect.unwrap(generate)
    if hasattr(f, "__func__"):
        f = f.__func__
    tree = ast.parse(textwrap.dedent(inspect.getsource(f)))
    fn = tree.body[0]
    fn.decorator_list = []
    blocks = [n for n in ast.walk(fn) if isinstance(n, ast.For)
              and isinstance(n.target, ast.Name) and n.target.id == "block_idx"]
    loops = [n for n in ast.walk(fn) if isinstance(n, ast.For)
             and isinstance(n.target, ast.Name) and n.target.id == "_"]
    assert len(blocks) == len(loops) == 1
    blocks[0].body.insert(0, ast.parse("_clock.block = block_idx").body[0])
    if clock.timing:
        wrap = ast.parse("with _clock.iteration(_):\n pass").body[0]
        wrap.body = loops[0].body
        loops[0].body = [wrap]
    transfer = f.__globals__["_transfer_tokens"]
    forward = f.__globals__["_cached_forward"]

    def counted_transfer(*a, **kw):
        result = transfer(*a, **kw)
        clock.transfer()
        return result

    def counted_forward(*a, **kw):
        result = forward(*a, **kw)
        clock.cached += 1
        return result

    ns = {**f.__globals__, "_clock": clock,
          "_transfer_tokens": counted_transfer, "_cached_forward": counted_forward,
          "_trim_to_first_eos": lambda generated, eos_id: generated}
    exec(compile(ast.fix_missing_locations(tree), "<narrative80-counted>", "exec"), ns)
    return torch.inference_mode()(ns[f.__name__])

"""Optional same-block replay. Capturing never advances algorithm state.

The graph is tied to one immutable prefix, one set of layer-state buffers, and
one input shape. It cannot be reused across generation blocks or model versions.
"""
import time
import torch


class BlockForwardGraph:
    def __init__(self, forward, args, kwargs):
        context = kwargs.get("losa_context_kwargs")
        if not context or not context["states"] or context.get("trace_detail"):
            raise ValueError("capture requires initialized LoSA states and trace_detail=False")
        self.states = context["states"]
        self.input = args[2].clone()
        self.args = (*args[:2], self.input, *args[3:])
        self.kwargs = kwargs
        self.forward = forward
        self.graph = torch.cuda.CUDAGraph()
        snapshot = self.snapshot()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        start = time.perf_counter()
        try:
            # Dispatch primes the kernels/workspaces with an actual eager
            # sparse step. Do not repeat two whole forwards per capture.
            # synchronize before capture: no unfinished allocator/kernel work.
            torch.cuda.synchronize()
            with torch.cuda.graph(self.graph, stream=stream):
                self.result = forward(*self.args, **self.kwargs)
        finally:
            torch.cuda.current_stream().wait_stream(stream)
            self.restore(snapshot)
        torch.cuda.synchronize()
        self.capture_seconds = time.perf_counter() - start

    def snapshot(self):
        return {
            key: tuple(getattr(state, name).clone() for name in
                       ("query", "prefix_output", "prefix_lse"))
            for key, state in self.states.items()
        }

    def restore(self, snapshot):
        for key, tensors in snapshot.items():
            for name, tensor in zip(("query", "prefix_output", "prefix_lse"), tensors):
                getattr(self.states[key], name).copy_(tensor)

    def replay(self, input_ids):
        if input_ids.shape != self.input.shape:
            raise ValueError("graph input shape changed")
        self.input.copy_(input_ids)
        self.graph.replay()
        return self.result

    def verify_replay(self, input_ids):
        """Check one evolving step, including all persistent layer states."""
        before = self.snapshot()
        self.input.copy_(input_ids)
        eager, _ = self.forward(*self.args, **self.kwargs)
        expected = eager.logits.clone()
        after = self.snapshot()
        selections = {}
        tails = {}
        for key, state in self.states.items():
            workspace = getattr(state, "workspace", None)
            if workspace is not None:
                selections[key] = {name: tensor.clone()
                                   for name, tensor in workspace.buffers.items()
                                   if name[0] in ("active_sorted", "pages")}
            tails[key] = tuple(getattr(state, name)[..., -input_ids.shape[1]:, :].clone()
                               for name in ("combined_key", "combined_value")
                               if hasattr(state, name))
        self.restore(before)
        actual, trace = self.replay(input_ids)
        torch.testing.assert_close(actual.logits, expected, rtol=0, atol=0)
        for key, tensors in after.items():
            for name, tensor in zip(("query", "prefix_output", "prefix_lse"), tensors):
                torch.testing.assert_close(getattr(self.states[key], name), tensor, rtol=0, atol=0)
        for key, values in selections.items():
            for name, tensor in values.items():
                torch.testing.assert_close(self.states[key].workspace.buffers[name], tensor,
                                           rtol=0, atol=0)
        for key, values in tails.items():
            for name, tensor in zip(("combined_key", "combined_value"), values):
                torch.testing.assert_close(
                    getattr(self.states[key], name)[..., -input_ids.shape[1]:, :],
                    tensor, rtol=0, atol=0)
        return actual, trace

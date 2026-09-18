"""Version-local graph dispatch: static state/workspace per generation block.

No cross-block state reuse. Capture time remains in generation wall time.
This saves Python dispatch on replay, not the mandatory dense initialization.
"""
from .graph import BlockForwardGraph


def run(eager, model, family, input_ids, attention_mask, position_ids, **kwargs):
    context = kwargs.get("losa_context_kwargs")
    enabled = bool(getattr(model, "_losa_optimized_graph_replay", False))
    if not enabled or not context or not context["states"] or context.get("trace_detail"):
        return eager(model, family, input_ids, attention_mask, position_ids, **kwargs)
    if getattr(model, "_losa_optimized_primed_states", None) is not context["states"]:
        model._losa_optimized_primed_states = context["states"]
        model._losa_optimized_graph_stats["priming_steps"] += 1
        return eager(model, family, input_ids, attention_mask, position_ids, **kwargs)
    runner = getattr(model, "_losa_optimized_graph", None)
    if runner is None or runner.states is not context["states"]:
        runner = BlockForwardGraph(
            eager, (model, family, input_ids, attention_mask, position_ids), kwargs)
        model._losa_optimized_graph = runner
        stats = getattr(model, "_losa_optimized_graph_stats", None)
        if stats is None:
            stats = model._losa_optimized_graph_stats = dict(captures=0, capture_seconds=0., replays=0)
        stats["captures"] += 1
        stats["capture_seconds"] += runner.capture_seconds
    model._losa_optimized_graph_stats["replays"] += 1
    if getattr(model, "_losa_optimized_verify_graph", False):
        return runner.verify_replay(input_ids)
    return runner.replay(input_ids)

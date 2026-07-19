"""Regression: PoC decode must actually run through CUDA graphs (default mode),
and eager mode must run with none — proven from the profiler trace, not log lines.

This guards criterion 3 (cudagraph engaged + speedup): if a refactor silently
dropped PoC off the captured-graph path, decode would fall back to eager and these
counts would collapse. The trace `cudaGraphLaunch` count is the objective signal.
"""
import pytest

from tests.poc._graph import count_tail_graph_launches, profile_poc_request


@pytest.mark.integration
def test_decode_runs_through_cudagraph():
    """Default (cudagraph) mode: a decode PoC request replays captured graphs."""
    total, traces = profile_poc_request(max_tokens=8, prof_dir="/tmp/poc_prof_cg_on")
    assert traces, "no profiler trace produced (VLLM_TORCH_PROFILER_DIR unsupported?)"
    assert total > 0, (
        f"decode PoC ran with ZERO cudaGraphLaunch — not graphed (got {total}). "
        f"PoC fell off vLLM's captured-graph path.")


@pytest.mark.integration
@pytest.mark.skip(reason="tail-graph live replay is shelved: not a net win (bottleneck "
                         "is input prep, not the snap) and it can't produce the margin/"
                         "q-vectors the margin gate + vector channel need. Re-enable with "
                         "POC_TAIL_CUDAGRAPH=1 once the tail is reworked — see "
                         "benchmarks/poc/CUDAGRAPH_POC_TAIL_STEPS.md.")
def test_decode_poc_tail_runs_through_its_own_cudagraph():
    """A nonzero total (above) only proves SOME graph replayed during a PoC
    request — the model forward is graphed via the native inline reflection
    path regardless of the post-processing tail (vllm/poc/tail_cudagraph.py).
    This asserts the tail graph SPECIFICALLY replayed, closing the coverage
    gap CUDAGRAPH_POC_TAIL_STEPS.md flagged: attribute cudaGraphLaunch events
    to the tail graph via its record_function-labeled replay span."""
    total, traces = profile_poc_request(max_tokens=16, prof_dir="/tmp/poc_prof_cg_tail")
    assert traces, "no profiler trace produced (VLLM_TORCH_PROFILER_DIR unsupported?)"
    tail_total = sum(count_tail_graph_launches(t) for t in traces)
    assert tail_total > 0, (
        f"decode PoC's post-processing tail ran with ZERO cudaGraphLaunch "
        f"attributable to PoCTailGraphManager (got {tail_total} of {total} total "
        f"launches) — the tail fell back to eager (or was never captured).")


@pytest.mark.integration
def test_eager_runs_without_cudagraph():
    """--enforce-eager: the same request replays NO graphs (clean baseline). Proves
    the graph in the default run is real, and that eager still works."""
    total, traces = profile_poc_request(server_extra_args=["--enforce-eager"],
                                        max_tokens=8, prof_dir="/tmp/poc_prof_cg_off")
    assert traces, "no profiler trace produced"
    assert total == 0, (
        f"eager run unexpectedly replayed {total} cudaGraphLaunch — "
        f"--enforce-eager should disable cudagraph entirely.")

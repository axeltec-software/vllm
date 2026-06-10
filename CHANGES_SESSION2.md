# Source code changes — session 2 (2026-06-09)

Source commits: `1e5a5c4`, `4dd31711`.  
The table below summarises what is in the **current working tree** relative to `im/cuda-graph-poc`.

| Change | File | Status |
|---|---|---|
| Falsy check for empty PoC output | `generate_queue.py` | **In code** |
| `{}` sentinel for "PoC ran, no artifacts" | `scheduler.py` | **In code** |
| `_poc_direct_output` for pure-PoC batch queue path | `gpu_model_runner.py` | **In code** |
| Workspace size 394 MB → 2 GiB | `poc_model_runner.py` | **In code** |
| Re-plan after warmup before graph capture | `poc_model_runner.py` | **In code** |
| Cache-hit: full re-plan + workspace swap + `synchronize` + replay | `poc_model_runner.py` | ~~Reverted~~ (slows CG to eager speed) |
| `use_cuda_graph=True` wrappers with stable GPU buffers | `poc_model_runner.py` | ~~Reverted~~ (tied to the item above) |
| Rename `_poc_graph_pool` → `_poc_prefill_graph_pool` | `poc_model_runner.py` | ~~Reverted~~ |

---

## `vllm/poc/generate_queue.py` — falsy check instead of `is None`

```diff
- if poc_out is None:
+ if not poc_out:  # None or empty dict (PoC ran but no artifact)
```

**Why:** The scheduler was changed to use `{}` (empty dict) instead of `None` to signal "PoC ran but produced no artifacts" (see `scheduler.py` below).  The old `is None` check treated `{}` as a valid artifact object and tried to unpack it, causing a downstream error.  The falsy `if not poc_out` treats both `None` (PoC did not run) and `{}` (PoC ran, no output) as "skip this nonce".

---

## `vllm/v1/core/sched/scheduler.py` — sentinel value for "PoC ran, no artifacts"

```diff
- poc_output = None
- if hasattr(model_runner_output, 'poc_outputs') and model_runner_output.poc_outputs:
+ poc_output: dict | None = {}  # empty dict = PoC ran but no artifacts
+ if hasattr(model_runner_output, 'poc_outputs') and model_runner_output.poc_outputs is not None:
```

**Why:** Two distinct states needed to be expressed:
- `None` — PoC did not execute this step (not a PoC request, or no `poc_outputs` field).
- `{}` (empty dict) — PoC executed but returned no artifacts (batch exceeded reservation, or other refusal).

Previously both mapped to `None`.  The old truthy check `if poc_outputs:` also skipped a valid but empty `poc_outputs` dict; the explicit `is not None` is correct.

---

## `vllm/v1/worker/gpu_model_runner.py` — `_poc_direct_output` for pure-PoC batches

```python
# __init__:
self._poc_direct_output: ModelRunnerOutput | None = None
```

```python
# execute_model() — pure-PoC path:
poc_output = ModelRunnerOutput(...)
self._poc_direct_output = poc_output
return poc_output
```

```python
# sample_tokens():
if self.execute_model_state is None:
    poc_out = self._poc_direct_output
    if poc_out is not None:
        self._poc_direct_output = None
        return poc_out
    # original PP non-final rank path...
```

**Why:** vLLM's async batch queue calls `execute_model()` and then `sample_tokens()` as separate steps.  For normal requests, `execute_model()` saves intermediate state into `execute_model_state` and `sample_tokens()` reads it back.  For pure-PoC batches, `execute_model()` returns a complete `ModelRunnerOutput` directly and never sets `execute_model_state`.  When `sample_tokens()` was subsequently called, `execute_model_state is None` was treated as the pipeline-parallel non-final-rank case and `None` was returned, triggering `RuntimeError("unexpected error")` in `step_with_batch_queue`.

The fix stores the complete output in `_poc_direct_output` immediately.  `sample_tokens()` checks this field first and returns it, bypassing the normal state-restoration path.

---

## `vllm/poc/poc_model_runner.py` — workspace size 394 MB → 2 GiB

```diff
- poc_ws = torch.zeros_like(b._workspace_buffer)
+ poc_ws = torch.zeros(
+     FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT,  # 2 GiB
+     dtype=torch.uint8,
+     device=device,
+ )
```

**Why:** `torch.zeros_like` copies the default inference-engine workspace size (~394 MB).  Large PoC batches need ~833 MB for a single FlashInfer intermediate buffer (`batch_prefill_tmp_v`), causing a buffer overflow crash:

```
RuntimeError: Buffer overflow when allocating memory for batch_prefill_tmp_v
with size 873463808 and alignment 16, but only 413138944 bytes available.
```

`FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT` (2 GiB) is the constant vLLM already uses for fixed-batch CUDA-graph workloads in the regular inference path.

---

## `vllm/poc/poc_model_runner.py` — re-plan after warmup, before graph capture

```python
# warmup run (corrupts poc_ws):
warmup_out = model(...)
torch.cuda.synchronize()
warmup_hidden = (...).clone()

# Re-plan: restore poc_ws to a valid plan state before capture.
poc_attn_meta, poc_slot_map = _create_v1_attn_metadata(
    batch_size, seq_len, block_size, device, worker
)

# Capture (poc_ws is now in a valid plan state):
with torch.cuda.graph(graph, pool=worker._poc_graph_pool):
    captured_out = model(...)
```

**Why:** The warmup `model(...)` call uses `poc_ws` as scratch space, overwriting the scheduling data that `plan()` placed there.  If graph capture starts immediately after warmup, the FlashInfer kernels are recorded with the workspace in an undefined post-run state.  The re-plan restores `poc_ws` to a valid plan state so the captured kernel sequence is correct.

**Performance note:** This runs only once per `(batch_size, seq_len)` shape (during graph capture, not during replay), so it adds no overhead to the hot path.

---

## Reverted: cache-hit full re-plan + `synchronize` + replay

This change from commit `4dd31711` was reverted because it made CUDA graph replay nearly as slow as eager execution.

The original change added `torch.cuda.synchronize()` before every `graph.replay()` in the cache-hit path.  `synchronize()` forces the CPU to block until all pending GPU operations complete, fully serialising CPU and GPU execution and eliminating the pipelining benefit of CUDA graphs.

Additionally, it called `_create_v1_attn_metadata()` (which invokes `plan()`) on every cache-hit replay, adding Python overhead and CUDA kernel launches to each PoC request after the first.

The current code reverts to the `im/cuda-graph-poc` approach: restore `paged_kv_indices` from a snapshot and return `(graph, captured_hidden)` for the caller to replay.

---

## Status in current code

| Scenario | Status |
|---|---|
| PoC only, 1st request | Graph captured; warmup result returned |
| PoC only, 2nd+ request | Graph replayed via `im/cuda-graph-poc` path |
| PoC → chat → PoC (3rd request NaN) | Needs test; may be fixed by re-plan after warmup |
| Mixed batch (PoC + completions together) | Open issue — see `POC_WORK_CONTEXT.md` |
| Decode CUDA graph | Working (`_poc_direct_output` fix) |
| Large-batch profiling crash | Fixed (workspace 2 GiB) |

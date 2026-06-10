# Source code changes — session 3

Base commit: `e8820f62` — "poc: KV-bound mixed decode PoC on v15 (cudagraph + reservation, no collective_rpc)"  
Head commit: `5f32e97b` — "Reverted some changes that slowed down the CG replay."

4 files changed, 35 insertions, 5 deletions.

---

## `vllm/poc/generate_queue.py` — falsy check instead of `is None`

```diff
- if poc_out is None:
+ if not poc_out:  # None or empty dict (PoC ran but no artifact)
```

**Why:** The scheduler was changed (see below) to initialise `poc_output` to `{}` (empty dict) rather than `None` to distinguish two states:

- `None` — PoC was not part of this scheduling step at all.
- `{}` — PoC ran but produced no artifacts (e.g. the batch was refused by the KV reservation guard).

With the old `is None` check, an empty dict `{}` from a refused batch was treated as a valid artifact object, and the code below it tried to call `.get()` on it, causing a downstream error.  The falsy check `if not poc_out` correctly treats both `None` and `{}` as "skip this nonce — no artifact available".

---

## `vllm/v1/core/sched/scheduler.py` — sentinel value for "PoC ran, no artifacts"

```diff
- poc_output = None
- if hasattr(model_runner_output, 'poc_outputs') and model_runner_output.poc_outputs:
+ poc_output: dict | None = {}  # empty dict = PoC ran but no artifacts
+ if hasattr(model_runner_output, 'poc_outputs') and model_runner_output.poc_outputs is not None:
```

**Why:** The default `poc_output = None` made it impossible to tell whether the PoC forward pass ran at all or whether it ran and found nothing to return.  Two distinct states are needed:

| Value | Meaning |
|---|---|
| `None` | PoC did not execute this step (request is not PoC, or `poc_outputs` field absent). |
| `{}` (empty dict) | PoC executed but returned no artifacts. The batch was gracefully refused (e.g. `poc_blocks_needed > poc_reserved`, or all nonces produced NaN). |
| `{"nonces": [...], "vectors": [...]}` | PoC executed and produced valid artifacts. |

The `{}` default propagates to `generate_queue.py` where the falsy check turns it into a `None` return for that nonce, signalling failure without crashing.

The condition change from `if poc_outputs:` (truthy) to `if poc_outputs is not None:` is needed because an empty `poc_outputs` dict would be falsy under the old check and skip the lookup even when the batch ran. The explicit `is not None` passes for any dict, including `{}`.

---

## `vllm/v1/worker/gpu_model_runner.py` — `_poc_direct_output` for pure-PoC batches in the async queue

### Field added to `__init__`

```python
self._poc_direct_output: ModelRunnerOutput | None = None
```

### `execute_model()` — store output before returning

```diff
- return ModelRunnerOutput(
-     req_ids=..., req_id_to_index=..., sampled_token_ids=...,
-     ...poc_outputs=poc_outputs_dict,
- )
+ poc_output = ModelRunnerOutput(
+     req_ids=..., req_id_to_index=..., sampled_token_ids=...,
+     ...poc_outputs=poc_outputs_dict,
+ )
+ self._poc_direct_output = poc_output
+ return poc_output
```

### `sample_tokens()` — return stored output when `execute_model_state` is absent

```python
if self.execute_model_state is None:
    poc_out = self._poc_direct_output
    if poc_out is not None:
        self._poc_direct_output = None
        return poc_out
    # original PP non-final rank path continues here...
```

**Why:** vLLM's async batch queue (`step_with_batch_queue`) calls `execute_model()` and `sample_tokens()` as two separate steps in the async pipeline.  For normal (non-PoC) batches, `execute_model()` saves intermediate GPU state into `execute_model_state` and `sample_tokens()` uses it to finalise sampling and return a `ModelRunnerOutput`.

For pure-PoC batches, `execute_model()` constructs the complete `ModelRunnerOutput` directly (PoC has no sampling step) and returns it immediately, never setting `execute_model_state`.  When `sample_tokens()` was then called by the queue, `execute_model_state is None` was treated as the pipeline-parallel non-final-rank case, and the function returned `None`.  `step_with_batch_queue` interpreted a `None` model output as an unexpected error and raised `RuntimeError("unexpected error")`.

The fix stores the complete `ModelRunnerOutput` in `_poc_direct_output` before `execute_model()` returns.  `sample_tokens()` checks this field first: if set, it clears it and returns the stored output, giving `step_with_batch_queue` the non-`None` result it expects.  The field is cleared after each read so it is not returned twice.

---

## `vllm/poc/poc_model_runner.py` — two fixes in the graph capture path

Both changes are in `_get_or_capture_poc_graph`, in the first-call (capture) branch only.  They have no effect on cache-hit replays and therefore do not affect CUDA graph replay performance.

### 1. Workspace size 394 MB → 2 GiB

```diff
+ from vllm.v1.attention.backends.flashinfer import (
+     FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT,
+ )
  ...
- poc_ws = torch.zeros_like(b._workspace_buffer)
+ poc_ws = torch.zeros(
+     FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT,
+     dtype=torch.uint8,
+     device=device,
+ )
```

**Why:** `torch.zeros_like(b._workspace_buffer)` copies the size of the inference-engine workspace, which defaults to ~394 MB (controlled by `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE`).  A PoC prefill batch is much larger than a single chat request — e.g. 32 nonces × 256 tokens = 8192 tokens.  FlashInfer needs ~833 MB for a single intermediate buffer (`batch_prefill_tmp_v`) at that scale, causing a buffer overflow crash when the 394 MB workspace is used:

```
RuntimeError: Buffer overflow when allocating memory for batch_prefill_tmp_v
with size 873463808 and alignment 16, but only 413138944 bytes available.
```

`FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT` is the 2 GiB constant vLLM already allocates for fixed-batch CUDA-graph workloads in the regular inference path.  It is large enough for any realistic PoC configuration.

### 2. Re-plan after warmup, before graph capture

```python
# warmup pass (corrupts poc_ws scheduling data):
warmup_out = model(input_ids=None, ...)
torch.cuda.synchronize()
warmup_hidden = (...).clone()

# NEW: restore poc_ws to a valid plan state before capture
poc_attn_meta, poc_slot_map = _create_v1_attn_metadata(
    batch_size, seq_len, block_size, device, worker
)

# capture (poc_ws is now in a valid plan state):
with torch.cuda.graph(graph, pool=worker._poc_graph_pool):
    captured_out = model(...)
```

**Why:** FlashInfer's `plan()` writes attention scheduling data (indptr tables, dispatch parameters) into `poc_ws`.  The warmup `model(...)` call uses `poc_ws` as scratch space for intermediate computations and overwrites that scheduling data.  If graph capture starts immediately after warmup, the FlashInfer kernels in the captured graph are recorded reading from `poc_ws` in its post-run (corrupted) state.

On the first replay after capture, `poc_ws` would still be in that corrupted state, so the FlashInfer kernels would compute incorrect attention → wrong hidden states → incorrect or NaN artifacts.

The second `_create_v1_attn_metadata` call re-runs `plan()`, which restores `poc_ws` to a valid scheduling state.  Graph capture then records the kernels reading from a well-defined workspace, and the first replay reads the same valid data.

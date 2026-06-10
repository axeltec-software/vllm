# Source code changes — session 2 (2026-06-09)

Commits:
- `1e5a5c4` — "Fixed CUDA Graph for mixed batches in decode. For prefill still doesn't work."
- `4dd31711` — "Now both prefill and decode are captured into cuda graphs, but this variant doesn't work for mixed batches."

---

## Commit 1 — `1e5a5c4`

Four files changed.  The goal was to make the decode CUDA graph work correctly in mixed-batch mode (PoC + normal completions in the same scheduler step).

### 1. `vllm/poc/generate_queue.py` — falsy check instead of `is None`

```diff
- if poc_out is None:
+ if not poc_out:  # None or empty dict (PoC ran but no artifact)
```

**What changed:** The PoC-ran-but-no-artifact signal was changed from `None` to `{}` (empty dict) in the scheduler (see below).  The old `is None` check treated `{}` as a valid artifact object and tried to unpack it, causing a downstream error.  The falsy check `if not poc_out` treats both `None` (PoC did not run at all) and `{}` (PoC ran but produced no output) as "skip this nonce", which is the correct behavior.

---

### 2. `vllm/v1/core/sched/scheduler.py` — sentinel value for "PoC ran, no artifacts"

```diff
- poc_output = None
- if hasattr(model_runner_output, 'poc_outputs') and model_runner_output.poc_outputs:
+ poc_output: dict | None = {}  # empty dict = PoC ran but no artifacts
+ if hasattr(model_runner_output, 'poc_outputs') and model_runner_output.poc_outputs is not None:
```

**What changed:** Two distinct states needed to be expressed:
- `None` — PoC did not execute this step (not a PoC request, or no poc_outputs field).
- `{}` (empty dict) — PoC executed but returned no artifacts (batch exceeded reservation, or other refusal).

Previously both mapped to `None`, making them indistinguishable.  The old truthy check `if poc_outputs:` would also skip a valid but empty `poc_outputs` dict; the explicit `is not None` check is correct.

The `{}` sentinel lets `generate_queue.py` return `None` for the nonce (indicating failure) without confusing it with "the PoC forward pass never ran".

---

### 3. `vllm/v1/worker/gpu_model_runner.py` — `_poc_direct_output` for pure-PoC batches

```python
# Added to __init__:
self._poc_direct_output: ModelRunnerOutput | None = None
```

```python
# execute_model(): pure-PoC path now stores the output before returning it
poc_output = ModelRunnerOutput(...)
self._poc_direct_output = poc_output
return poc_output
```

```python
# sample_tokens(): retrieves it when execute_model_state is absent
if self.execute_model_state is None:
    poc_out = self._poc_direct_output
    if poc_out is not None:
        self._poc_direct_output = None
        return poc_out
    # original PP non-final rank path...
```

**Why this was needed:** vLLM's async batch queue calls `execute_model()` followed by `sample_tokens()` as separate steps.  For normal requests, `execute_model()` saves intermediate state into `execute_model_state` and `sample_tokens()` reads it back.  For pure-PoC batches, `execute_model()` returns a complete `ModelRunnerOutput` directly — it never sets `execute_model_state`.  When `sample_tokens()` was subsequently called, `execute_model_state is None` was treated as the pipeline-parallel non-final-rank case, causing the function to return `None`, which triggered `RuntimeError("unexpected error")` in `step_with_batch_queue`.

The fix stores the complete output in `_poc_direct_output` immediately after building it.  `sample_tokens()` checks this field first and returns it, bypassing the normal state-restoration path.

---

### 4. `vllm/poc/poc_model_runner.py` — prefill cache-hit path falls back to eager

```python
# cache-hit: run eagerly instead of replaying the CUDA graph
eager_meta, eager_slot = _create_v1_attn_metadata(
    batch_size, seq_len, block_size, device, worker
)
model = getattr(worker, "model_runner", worker).model
with set_forward_context(eager_meta, ...):
    with poc_forward_context():
        out = model(input_ids=None, positions=positions_buf, inputs_embeds=embeds_buf)
return None, (out[0] if isinstance(out, tuple) else out)
```

**Why this was needed (and why it is only a temporary workaround):** The prefill CUDA graph had a workspace-state issue.  FlashInfer's `plan()` writes the attention schedule into `poc_ws`.  `run()` (the forward pass) uses `poc_ws` as scratch space and overwrites that schedule.  The original code tried to replay the graph on cache-hit without re-planning, which meant the baked-in FlashInfer kernels were reading garbage from `poc_ws`.

This commit chose the conservative workaround: bypass the graph entirely on cache-hit and run eagerly.  The graph still captures correctly on the first call; only subsequent calls (cache-hits) skip it.  Commit 2 replaces this workaround with a proper replay that re-plans before every replay.

---

## Commit 2 — `4dd31711`

One file changed (`vllm/poc/poc_model_runner.py`).  The goal was to restore full CUDA graph usage for the prefill path — capture once, replay correctly on every subsequent call.

### 1. Cache-hit replay: full re-plan + synchronize before replay

The eager workaround from commit 1 was replaced with a proper replay sequence:

```python
graph, captured_hidden, poc_attn_meta, poc_slot_map, poc_workspaces, poc_prefill_wrappers, _kv_snap = cache[key]
builders = list(_iter_attn_builders(worker))

# Step 1: swap in poc_ws and poc_pw so that _create_v1_attn_metadata writes
#         into the PoC workspace, not the inference-engine workspace.
orig_workspaces = [b._get_workspace_buffer() for b in builders]
orig_prefill_wrappers = [b._prefill_wrapper for b in builders]
for b, poc_pw, poc_ws in zip(builders, poc_prefill_wrappers, poc_workspaces):
    b.set_workspace_buffer(poc_ws)
    b._prefill_wrapper = poc_pw

# Step 2: re-plan into poc_ws with current PoC parameters.
_create_v1_attn_metadata(batch_size, seq_len, block_size, device, worker)

# Step 3: restore inference-engine workspace and wrapper.
for b, ows, opw in zip(builders, orig_workspaces, orig_prefill_wrappers):
    b.set_workspace_buffer(ows)
    b._prefill_wrapper = opw

# Step 4: synchronize — FlashInfer plan() may issue non-blocking H2D/D2D
#         copies for the stable paged_kv buffers; guarantee they land before
#         graph.replay() reads from those addresses.
torch.cuda.synchronize()

# Step 5: replay with the forward context that was active at capture time.
with set_forward_context(poc_attn_meta, vllm_config, ...):
    with poc_forward_context():
        graph.replay()

return None, captured_hidden
```

**Why the re-plan is essential:** `run()` writes intermediate results into `poc_ws` during every forward pass (both warmup and each subsequent replay), overwriting the scheduling data that `plan()` placed there.  Before the next replay, `poc_ws` must be restored to a valid plan state.  Without this re-plan, the FlashInfer kernels baked into the graph read stale/random bytes from `poc_ws` → incorrect attention outputs → NaN in hidden states.

**Why `use_cuda_graph=True` wrappers are essential for the re-plan to work:**
When `plan()` is called with a standard (non-CUDA-graph) wrapper, FlashInfer allocates new GPU tensors for `paged_kv_indices`, `paged_kv_indptr`, etc. on each call.  The CUDA graph bakes in the GPU addresses from capture time.  After capture, those original tensors may be freed by PyTorch's allocator and reused for other data.  On replay, the graph reads garbage → NaN.

With `use_cuda_graph=True`, the wrapper pre-allocates these tensors once at construction time, at stable fixed addresses.  On each `plan()` call, FlashInfer copies the new data *into* those stable buffers (not into newly allocated ones).  The graph's baked-in addresses always point to valid, up-to-date data on every replay.

**Why `torch.cuda.synchronize()` is needed:** `plan()` issues some non-blocking CUDA copies (H2D for host-side indptr arrays, D2D for device-side paged_kv indices).  In vLLM's multi-stream setup these could be in-flight when `graph.replay()` starts on the compute stream.  The synchronize ensures all copy operations have completed and the buffers contain the new plan data before the graph kernels read from them.

---

### 2. Capture path: dedicated `use_cuda_graph=True` wrappers with stable GPU buffers

```python
# Per-builder: allocate four stable GPU buffers up front.
qo_indptr_buf            = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
paged_kv_indptr_buf      = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
paged_kv_indices_buf     = torch.zeros(total_blocks,   dtype=torch.int32, device=device)
paged_kv_last_page_len_buf = torch.zeros(batch_size,   dtype=torch.int32, device=device)

poc_pw = BatchPrefillWithPagedKVCacheWrapper(
    poc_ws, kv_cache_layout,
    use_cuda_graph=True,
    qo_indptr_buf=qo_indptr_buf,
    paged_kv_indptr_buf=paged_kv_indptr_buf,
    paged_kv_indices_buf=paged_kv_indices_buf,
    paged_kv_last_page_len_buf=paged_kv_last_page_len_buf,
)
b._prefill_wrapper = poc_pw
poc_prefill_wrappers.append(poc_pw)
```

The wrappers and their backing buffers are stored in `cache[key]` alongside the graph.  They must remain alive for the lifetime of the cached graph because the graph's recorded CUDA ops read from those GPU addresses.

---

### 3. Capture path: workspace size 394 MB → 2 GiB

```diff
- poc_ws = torch.zeros_like(b._workspace_buffer)   # copies ~394 MB
+ poc_ws = torch.zeros(
+     FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT,   # 2 GiB
+     dtype=torch.uint8,
+     device=device,
+ )
```

**Why:** `torch.zeros_like` copies the default inference-engine workspace size (controlled by `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE`, default 394 MB).  A 32-nonce PoC batch with 256 token sequence length needs ~833 MB for just one FlashInfer intermediate buffer (`batch_prefill_tmp_v`).  The 394 MB workspace overflows, crashing the server:

```
RuntimeError: Buffer overflow when allocating memory for batch_prefill_tmp_v
with size 873463808 and alignment 16, but only 413138944 bytes available.
```

`FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT` is the 2 GiB constant that vLLM already uses for fixed-batch (CUDA-graph-mode) workloads in the regular inference path — it is large enough for any realistic PoC configuration.

---

### 4. Capture path: re-plan after warmup, before graph capture

```python
# Run warmup pass
warmup_out = model(input_ids=None, positions=positions_buf, inputs_embeds=embeds_buf)
torch.cuda.synchronize()
warmup_hidden = (...).clone()

# Re-plan: warmup run() corrupted poc_ws; restore valid plan state before capture.
poc_attn_meta, poc_slot_map = _create_v1_attn_metadata(
    batch_size, seq_len, block_size, device, worker
)

# Now capture
with torch.cuda.graph(graph, pool=worker._poc_prefill_graph_pool):
    captured_out = model(...)
```

**Why:** The warmup forward pass writes into `poc_ws` as scratch.  If capture starts immediately after warmup, the graph kernels are recorded with a post-run (corrupted) workspace state.  The re-plan restores `poc_ws` to the exact state expected by the FlashInfer kernels, so the captured kernel sequence is correct from the first replay.

---

### 5. Rename `_poc_graph_pool` → `_poc_prefill_graph_pool`

Minor naming change for clarity, to distinguish from the decode graph pool (`_poc_decode_graph_pool`) when both exist on the same worker.

---

## Status after these two commits

| Scenario | Status |
|---|---|
| PoC only (no chat), 1st request | Captured eagerly, graph stored |
| PoC only, 2nd+ request | Graph replayed (commit 2 fix) |
| PoC → chat → PoC (3rd request NaN) | Should be fixed by re-plan + synchronize; needs test |
| Mixed batch (PoC + completions together) | **Broken** — commit 2 title says "doesn't work for mixed batches"; see open issue in `POC_WORK_CONTEXT.md` |
| Decode CUDA graph | Working (commit 1 fixed the _poc_direct_output path) |
| Profiling large batches | Fixed (workspace overflow resolved) |

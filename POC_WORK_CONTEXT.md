# PoC CUDA Graph Work Context

Branch: `poc-mixed-cudagraph`  
Base: vLLM v0.15.x (V1 architecture, FlashInfer attention backend)

---

## What this branch does

Implements KV-bound mixed-decode PoC (Proof of Compute) with CUDA graph acceleration for the prefill and decode steps. The PoC runs batched forward passes on reserved KV blocks (blocks 0..poc_reserved), applies per-layer Householder reflections via registered forward hooks, and extracts sphere-projected hidden states as cryptographic artifacts.

The CUDA graph is gated by `VLLM_POC_CUDAGRAPH=1` (defaults off). When disabled, every PoC call runs eagerly.

---

## Key files

- `vllm/poc/poc_model_runner.py` — all PoC graph capture/replay logic, `execute_poc_forward`
- `vllm/poc/layer_hooks.py` — `LayerHouseholderHook`, `poc_forward_context`
- `vllm/poc/gpu_random.py` — `apply_householder`, `generate_inputs`, etc.
- `vllm/v1/attention/backends/flashinfer.py` — `FlashInferMetadataBuilder.build()`, workspace management
- `run_request.sh` — test script (PoC + chat interleaved)
- `run_profile_server.sh` / `run_profile_requests.sh` — profiling scripts

---

## How the CUDA graph works

### Capture (first call for a given `(batch_size, seq_len)`)

1. **Isolated workspace**: For each attention builder `b`, save `b._workspace_buffer` (ows) and `b._prefill_wrapper` (opw). Allocate `poc_ws` at `FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT` (2 GB). Swap both in: `b.set_workspace_buffer(poc_ws)`, `b._prefill_wrapper = poc_pw`.

2. **`poc_pw` construction**: `BatchPrefillWithPagedKVCacheWrapper(poc_ws, layout, use_cuda_graph=True, qo_indptr_buf=..., paged_kv_indices_buf=..., ...)`. The `use_cuda_graph=True` flag makes `plan()` copy paged_kv data into pre-allocated stable GPU buffers (fixed addresses baked into the graph) instead of allocating new tensors each call.

3. **First plan + warmup**: `_create_v1_attn_metadata(...)` → `builder.build()` → `poc_pw.plan()` writes scheduling data into `poc_ws` and the stable paged_kv buffers. Run `model(warmup)` eagerly — `run()` executes and may use `poc_ws` as scratch.

4. **Re-plan before capture**: Call `_create_v1_attn_metadata(...)` again to restore `poc_ws` to a valid plan state. This is critical because `run()` during warmup may have overwritten the scheduling data in `poc_ws`.

5. **Graph capture**: `torch.cuda.graph(graph, pool=worker._poc_prefill_graph_pool)` around `model(capture)`. All CUDA kernels — attention (`poc_pw.run()` or TRTLLM kernel), FFN, norms, per-layer Householder hooks — are recorded.

6. **Restore + store**: Restore `b._workspace_buffer = ows`, `b._prefill_wrapper = opw`. Store in `worker._poc_cuda_graphs[(batch_size, seq_len)]`: `(graph, captured_hidden, poc_attn_meta, poc_slot_map, poc_workspaces, poc_prefill_wrappers, kv_snapshots)`.

   `kv_snapshots` = clone of `b.paged_kv_indices.gpu[:total_blocks]` at capture time = `[0..total_blocks-1]`.

### Replay (subsequent calls, same `(batch_size, seq_len)`)

1. Swap in `poc_ws` and `poc_pw` for each builder.
2. Call `_create_v1_attn_metadata(...)` — re-plans `poc_ws` with fresh PoC parameters. This is essential to restore `poc_ws` from its post-run state (left by the previous replay) and to refresh the stable paged_kv buffers via `poc_pw.plan()`.
3. Restore `b._workspace_buffer = ows`, `b._prefill_wrapper = opw`.
4. `torch.cuda.synchronize()` — ensures all non-blocking H2D/D2D copies from `plan()` have landed before the graph kernels read from those buffers.
5. `with set_forward_context(poc_attn_meta, ...)`: `graph.replay()`.

### What is captured in the graph

- All FlashInfer prefill attention kernels (reads `poc_ws`, `paged_kv_indices_buf`, `kv_cache`)
- `reshape_and_cache_flash` (writes Q/K/V to KV cache using `poc_slot_map`)
- All FFN, norm, and other model CUDA kernels
- Per-layer Householder hooks: `apply_householder(hidden, v)` ops (reads `reflection_vectors[i]` at stable addresses, writes in-place to hidden states)

### Hooks and block_hash rotation

`LayerHouseholderHook` is registered once per model. Reflection vectors are pre-allocated at fixed GPU addresses. On `block_hash` change, `update_block_hash()` writes new values in-place — the graph remains valid because addresses are stable. The hooks fire during capture because `poc_forward_context()` is active (sets `_poc_forward_active_flag = True`); on graph replay Python code does not re-run so the flag state is irrelevant.

---

## Changes made in this session

### 1. Fixed workspace buffer size (profiling crash)

**File**: `poc_model_runner.py` line ~542  
**Bug**: `poc_ws = torch.zeros_like(b._workspace_buffer)` copies the default 394 MB workspace. Large PoC batches need ~833 MB for a single FlashInfer intermediate buffer (`batch_prefill_tmp_v`), causing:
```
RuntimeError: Buffer overflow when allocating memory for batch_prefill_tmp_v
with size 873463808 and alignment 16, but only 413138944 bytes available.
```
**Fix**: Allocate at `FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT` (2 GB):
```python
from vllm.v1.attention.backends.flashinfer import FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT
poc_ws = torch.zeros(FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT, dtype=torch.uint8, device=device)
```

### 2. Fixed cache-hit replay path regression

**Bug**: A previous attempt replaced `_create_v1_attn_metadata` in the cache-hit path with direct `poc_pw.plan()` calls using pre-computed arrays. This broke the 2nd consecutive PoC call (even without chat requests), causing NaN in all hidden states. Root cause: calling `plan()` directly bypasses some side effects of `builder.build()` that are needed for correct workspace setup.

**Fix**: Reverted to the `_create_v1_attn_metadata`-based approach in the cache-hit path. Added workspace swap (`set_workspace_buffer(poc_ws)`) before the call and a `torch.cuda.synchronize()` after, to match the capture path structure and eliminate any potential race between non-blocking plan copies and graph replay.

The cache-hit path now:
```python
orig_workspaces = [b._get_workspace_buffer() for b in builders]
orig_prefill_wrappers = [b._prefill_wrapper for b in builders]
for b, poc_pw, poc_ws in zip(builders, poc_prefill_wrappers, poc_workspaces):
    b.set_workspace_buffer(poc_ws)
    b._prefill_wrapper = poc_pw
_create_v1_attn_metadata(batch_size, seq_len, block_size, device, worker)
for b, ows, opw in zip(builders, orig_workspaces, orig_prefill_wrappers):
    b.set_workspace_buffer(ows)
    b._prefill_wrapper = opw
torch.cuda.synchronize()
with set_forward_context(poc_attn_meta, ...):
    with poc_forward_context():
        graph.replay()
```

---

## Open / unsolved issues

### Issue 1: NaN on 3rd PoC request after interleaved chat (original bug)

**Symptom**: With `run_request.sh` running `PoC → chat → PoC → chat → PoC`, the 3rd PoC request (2nd cache-hit replay) produces NaN in all 32/32 hidden states. The 1st (capture) and 2nd (1st replay) are OK.

**Status**: Not reproduced in recent testing after the regression fix and `synchronize()` addition. May have been fixed by those changes, or may still be present. Needs a full test run with `run_request.sh` (chat lines uncommented) to confirm.

**What was investigated**:
- After a chat request: `builder.paged_kv_indices.gpu` is overwritten with chat block indices, `b._prefill_wrapper` may be set to a new chat wrapper. The re-plan via `_create_v1_attn_metadata` overwrites these back to PoC parameters before replay, so this should not cause the issue.
- `poc_ws` is isolated (separate from `b._workspace_buffer`). Chat requests do not write to `poc_ws`.
- The double-plan (warmup + re-plan before capture) ensures the graph is captured with valid `poc_ws` state.
- A timing/race condition between FlashInfer's non-blocking plan copies and `graph.replay()` was hypothesised as the root cause; `torch.cuda.synchronize()` was added as a fix.
- The TRTLLM vs FlashInfer path for large batches (≥8k tokens) was checked; both paths use stable addresses in the cached metadata.

**Most likely remaining cause** (if still present): Unknown. All static-analysis paths lead to correct behavior. Actual GPU-level debugging (e.g., CUDA error checking, `CUDA_LAUNCH_BLOCKING=1`) is needed to confirm.

### Issue 2: Incorrect artifacts for mixed PoC + completions batch

**Symptom**: When PoC requests and regular chat completion requests are scheduled in the same batch (mixed batching), PoC artifacts may be incorrect.

**Root cause**: The PoC CUDA graph path is a *separate* forward pass that runs outside the normal vLLM scheduler path (via `execute_poc_forward`). However, the per-layer Householder hooks (`LayerHouseholderHook`) are registered permanently on the model. In mixed-batch mode, the hook uses a position mask (`poc_forward_context_with_mask`) to apply transformations only to PoC token positions. This mask mode is **explicitly incompatible with CUDA graphs** — the hook detects `poc_mask is not None` and takes a non-in-place code path that CUDA graph capture does not support:

```python
# layer_hooks.py _create_hook():
if poc_mask is not None:
    # Mask mode: eager only. Do not trigger during graph capture/replay.
    v = self.reflection_vectors[layer_idx]
    return self._apply_selective_transform(output, poc_mask, v)
```

When the CUDA graph is replayed, Python hooks do not run (only captured CUDA ops replay). This means:
- During the PoC CUDA graph replay, hooks fire (they were captured) — the Householder reflections apply to ALL hidden states (all token positions), regardless of any mask. This is correct for pure-PoC batches.
- For mixed batches going through the *normal* scheduler path (not the PoC graph path), the mask mode hooks apply transformations only to PoC positions. However, if the PoC CUDA graph was captured *before* mixing started, and then a mixed batch triggers a graph replay, the graph's captured ops transform ALL positions rather than just PoC positions.

**Impact**: The PoC graph is only triggered via `execute_poc_forward` (a separate code path, not the normal batch scheduler path), so the two paths do not directly collide. Mixed batching affects the *normal inference path*, which uses live `poc_forward_context_with_mask`. Pure PoC batches use the CUDA graph path with `poc_forward_context` (all-positions mode).

**Status**: The interaction between mixed-batch inference and the PoC CUDA graph has not been fully tested. The risk is that if any mixed-batch forward triggers the PoC CUDA graph replay (e.g., if `execute_poc_forward` is called while a mixed batch is in flight), the graph would apply Householder transformations to ALL tokens, corrupting chat completion outputs.

**Suggested fix direction**: Ensure `execute_poc_forward` is never called concurrently with normal inference. The current design uses `collective_rpc` to run PoC between scheduler ticks; this serialization should prevent simultaneous execution, but this needs to be verified.

---

## Branch comparison: `im/cuda-graph-poc` vs `poc-mixed-cudagraph`

The `im/cuda-graph-poc` remote branch takes a different approach:

| Aspect | `im/cuda-graph-poc` | `poc-mixed-cudagraph` (current) |
|---|---|---|
| `poc_ws` size | `zeros_like` → 394 MB **bug** | `zeros(2 GB)` fixed |
| Wrapper | Standard (no `use_cuda_graph=True`) | `use_cuda_graph=True` + private stable buffers |
| Re-plan between warmup and capture | Missing | Present (second `_create_v1_attn_metadata`) |
| Re-plan before replay | Missing (only `paged_kv_indices` restored) | Present (full re-plan + synchronize) |
| Replay context metadata | Static `attn_metadata` (wrong wrapper ref) | `poc_attn_meta` from cache (correct) |
| Decode re-plan | Per-step re-plan ✓ | Per-step re-plan ✓ |

`im/cuda-graph-poc` relies on FlashInfer's float workspace scheduling data surviving `run()` (i.e., scratch and scheduling regions do not overlap). If that assumption holds, the simpler no-re-plan approach works. If it doesn't, every replay after the first produces wrong results. Our branch is conservative and re-plans before every replay to avoid the question.

---

## Test commands

```bash
# Start server (with CUDA graph enabled)
VLLM_POC_CUDAGRAPH=1 bash run_request.sh   # currently has chat commented out

# With chat interleaved (uncomment lines in run_request.sh):
#   python random_prompt_completion.py --server http://localhost:8005
# Then: PoC → chat → PoC → chat → PoC  (3rd PoC should not NaN)

# Profiling
bash run_profile_server.sh &
bash run_profile_requests.sh
```

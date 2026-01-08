# PoC Scheduler Integration - Branch Documentation

## Branch Purpose

This branch (`fix/poc-scheduler-integration`) integrates Proof of Compute (PoC) nonce verification into vLLM's scheduler, allowing PoC requests to be processed alongside chat requests through a unified code path.

**Goal:** Make vLLM treat PoC nonce verification as regular requests with lower priority, without significant performance loss.

## Architecture Change

### Before (Separate Paths)

```
Chat Request ──► Scheduler ──► Worker ──► GPU
                    │
                    │ (waits)
                    ▼
PoC Request ───► collective_rpc ──► GPU (BLOCKS EVERYTHING)
```

**Problems:**
- PoC blocks chat (no priority scheduling)
- Manual batch size tuning required
- Cannot share GPU efficiently with chat

### After (Unified Path)

```
Chat Request ──┐
  (priority=0) │
               ├──► Scheduler ──► Worker ──► GPU
PoC Request ───┘
  (priority=1)
```

**Benefits:**
- Chat always wins (priority=0 > priority=1)
- Automatic batching by scheduler
- Mixed batches maximize GPU utilization
- Single code path for all requests

## Fixes Applied

### Fix 1: k-dim Alignment (Critical)

**File:** `vllm/worker/model_runner.py:2042`

```python
# BEFORE (wrong)
POC_PICK_K_DIMS = 64

# AFTER (correct)
POC_PICK_K_DIMS = 12  # Must match poc_model_runner.py
```

**Why:** Different k produces different distances. Validators use k=12, so integration must match.

### Fix 2: Priority Value

**File:** `vllm/poc/routes.py:465, 540`

```python
# BEFORE
priority=0  # Same as chat

# AFTER
priority=1  # Lower than chat (0) - PoC yields to inference
```

**Why:** Priority 0 = highest. PoC should yield to chat requests.

### Fix 3: Pipeline Parallel Rank Check

**File:** `vllm/worker/model_runner.py` (start of `_compute_poc_distances`)

```python
# ADDED
from vllm.distributed import get_pp_group

if not get_pp_group().is_last_rank:
    return {}  # Only compute on last rank
```

**Why:** Only last PP rank has final hidden states. Other ranks have intermediate tensors.

### Fix 4: Batch Position Mapping

**File:** `vllm/worker/model_runner.py` (in `_compute_poc_distances`)

```python
# BEFORE (wrong - assumed dict order = batch order)
for idx, (request_id, poc_params) in enumerate(poc_params_map.items()):
    start_pos = seq_start_positions[idx]

# AFTER (correct - lookup actual position)
request_id_to_batch_pos = {req_id: pos for pos, req_id in enumerate(request_ids_to_seq_ids.keys())}
for request_id, poc_params in poc_params_map.items():
    batch_pos = request_id_to_batch_pos.get(request_id)
    start_pos = seq_start_positions[batch_pos]
```

**Why:** In mixed batches, PoC requests may be at any position, not necessarily 0, 1, 2...

### Fix 5: Chunked Prefill Exemption

**File:** `vllm/core/scheduler.py:2038`

```python
# BEFORE
if enable_chunking and len(seqs) == 1:

# AFTER
if enable_chunking and len(seqs) == 1 and seq_group.poc_params is None:
```

**Why:** PoC needs full prefill for correct distance computation. Chunking produces intermediate hidden states.

## Experiments to Perform

### 1. Smoke Test (Basic Functionality)

```bash
cd ~/gonka-vllm-poc
pip install -e .
VLLM_USE_V1=0 python scripts/poc_smoke_test.py
```

**Expected:** All checks pass (determinism, valid distances)

### 2. Integration Tests (All Fixes)

```bash
VLLM_USE_V1=0 python scripts/poc_integration_tests.py --model Qwen/Qwen2.5-7B-Instruct
```

**Tests:**
- Mixed batch (chat + PoC together) → validates Fix 4
- Priority scheduling (chat before PoC) → validates Fix 2
- Chunked prefill (long PoC not chunked) → validates Fix 5
- Determinism (same nonce = same distance)

### 3. Distance Consistency Test

Compare distances between scheduler integration and original `collective_rpc` path:

```python
# Run same nonces through both paths
# Scheduler path:
engine.add_request(request_id, prompt, params=poc_params)

# Original path:
execute_poc_forward(worker, block_hash, public_key, nonces, ...)

# Distances should match (within float tolerance)
```

### 4. Mixed Workload Performance

```bash
# Start vLLM server with PoC support
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --enable-chunked-prefill

# In parallel:
# Terminal 1: Send chat requests
# Terminal 2: Send PoC nonces

# Measure:
# - Chat latency with/without PoC load
# - PoC throughput (nonces/sec)
# - GPU utilization
```

### 5. Multi-GPU Test (if available)

```bash
# TP=2 test
python scripts/poc_integration_tests.py --tensor-parallel-size 2

# PP=2 test (validates Fix 3)
python scripts/poc_integration_tests.py --pipeline-parallel-size 2
```

## Files Modified

| File | Changes |
|------|---------|
| `vllm/poc/routes.py` | priority=0 → priority=1 |
| `vllm/worker/model_runner.py` | k-dim fix, PP rank check, batch position mapping |
| `vllm/core/scheduler.py` | PoC exempt from chunked prefill |

## Distance Computation Reference

```
distance = ||y_k - target||₂

Where:
1. Generate input embeddings from (block_hash, public_key, nonce)
2. Run model forward pass → hidden_states
3. Take last token: last_hidden = hidden_states[-1]
4. Normalize: last_hidden /= ||last_hidden||
5. Pick k=12 random dimensions → x_k
6. Apply Haar rotation: y_k = Q @ x_k
7. Normalize: y_k /= ||y_k||
8. Generate target unit vector
9. distance = ||y_k - target||

Valid nonce: distance < r_target (typically ~1.4-1.5)
```

## Related Documentation

- Full analysis: `~/gonka-setup-docs/dev/POC-SCHEDULER-INTEGRATION.md`
- Original design: commit `ee7acf91` (notes.md)
- Original implementation: commit `e452beb6`

# Decode-PoC + chat co-existence — architecture (vLLM 0.25 branch)

How decode-PoC (Proof-of-Compute) runs **alongside** normal chat inference,
and how it reaches the gonka ML node and chain. Read top-to-bottom =
request lifecycle.

## Two request types
- **Chat** — normal request: prefill, decode, sample a token per step, return
  text. Uses vLLM's standard sampler + output path.
- **PoC** — per-nonce *deterministic* forward. Each decode step snaps the
  hidden state to a discrete codebook index `sphere_k`, chained (`prev_k`
  seeds the next step). Produces a **trajectory** (`k_points_steps`) and
  seeded vector picks, **not text**, delivered on a separate channel
  (`poc_outputs`).

**Core invariant: PoC produces no token.** It must never depend on or
pollute the token sampler/output path.

## Implementation homes
The PoC implementation exists in two byte-identical homes: the `gonka_poc`
plugin (production path) and this in-tree package (fallback). Engine seams
resolve through `poc/dispatch.py` — plugin when installed, in-tree
otherwise — and log which is active. Wire types (`poc_params.py`) are never
dispatched: one class per process.

## Request lifecycle (by file)

| Stage | File | Role |
|---|---|---|
| Entry | `entrypoints/openai/api_server.py` | registers PoC routes (`add_api_route`, not `include_router` — FastAPI's `_IncludedRouter` breaks prometheus route naming) |
| Route | `poc/routes.py` | `/api/v1/pow/generate` → builds `PoCParams`, fans out via generate_queue |
| Fan-out | `poc/generate_queue.py` | one engine request per nonce, `generate(poc_params=...)`, no sampling_params; assembles artifact from the emit-once `poc_output` |
| Request | `v1/engine/async_llm.py` | builds the engine request with a **unique per-nonce cache salt** (prefix caching must never share KV across nonces) |
| Admission | `poc/admission.py` ← seam in `v1/core/sched/scheduler.py` | which nonces enter this step's batch: poc_share budget, KV-derived AUTO cap (admitting beyond pool capacity livelocks), decode-only mixing. PoC gets paged KV like chat |
| Async cap | `v1/core/sched/scheduler.py` | PoC rows are **exempt** from the async output-placeholder finish check — they never produce sampled tokens, so token-count logic has a false premise; artifact presence is the sole finish authority |
| Runner bridge | `poc/runner_bridge.py` ← seams in `v1/worker/gpu_model_runner.py` (V1) and `v1/worker/gpu/model_runner.py` (V2) | builds PoC row views from `SchedulerOutput`, publishes per-row block hashes + routing state, excludes PoC rows from sampling. Hooks fire **before** the cudagraph replay branch |
| Transforms | `poc/native.py` | attached once at model load, **inside the compiled forward**: seeded embeddings (PoCEmbedding), Householder reflections (PoCLayer), seeded MoE routing (PoCRouter, gate+experts discovery, hard-fail on unreadable expert metadata), sphere snap on final norm (PoCSnap). Class-level dispatch — torch.compile traces class forwards; instance patches are ignored and module replacement breaks the compiled parameter map |
| Chain/emit | `poc/mixed_decode.py` | k-chain step→step, seeded vector picks, artifact assembled on-device, **emit-once** guard (pipelined steps must not re-emit) |
| Deliver | `scheduler.update_from_output` → `v1/engine/output_processor.py` | PoC on its own branch: drain `poc_outputs`, artifact-driven finish; chat reads its sampled token |
| ML node | gonka `decentralized-api` `pow_v2_routes.py` (Go) | proxy: validates v2 body, forwards to vLLM `/api/v1/pow/*` |
| Chain | gonka `poc/decode.go` (Go) | trajectory packed in `PoCArtifactV2.Vector` (no proto change); teacher-forced validation, `fraud = mismatch_rate > p_mismatch` |

## PoC off the token path (why the output rules exist)
The scheduler reads a chat token as
`sampled_token_ids[req_id_to_index[req_id]]`; PoC is handled on its own
branch and never reads that array. Two rules keep async safe:
- **Never renumber**: PoC rows stay in the output arrays at their natural
  input_batch index. Renumbering to a chat-only `0..n` is self-consistent
  in sync but desyncs in async (tokens resolve later from the full GPU
  tensor) — chat then reads a PoC row's ignored token.
- **Snapshot, don't read live**: anything `sample_tokens` needs about this
  step's batch comes from the `execute_model` snapshot
  (`ExecuteModelState`); the next step's `execute_model` rebuilds live
  state before this step's sampling runs.

## Invariants
- PoC never enters the token sampler/output path.
- Async scheduling is the production configuration; sync-only results do
  not count as verification.
- All PoC forwards run inside captured CUDA graphs — transforms, snap and
  routing are in-graph; zero eager fallback.
- Pure dynamic KV; chat KV untouched by PoC; unique cache salt per nonce.
- Seeded MoE routing is part of the algorithm, not a toggle: expert
  selection must not read the noise-prone hidden state.
- Determinism: cudagraph ≡ eager, but **attention-backend-specific** (pin
  the backend for cross-node validation).

## Tests (this branch)
- `tests/poc/unit/` — admission, runner-bridge views, native attach guards
  (loud-fail contracts), residual contract pins (each seam behavior tied to
  the incident it prevents).
- `tests/poc/integration/test_live_decode_trajectory.py` — multi-nonce full
  trajectories, cross-round determinism, and the cross-version golden
  trajectories (dense + MoE): any change there is a consensus change.
- `benchmarks/poc_perf/` — throughput, cross-validation matrix, and chat
  quality under concurrent PoC.

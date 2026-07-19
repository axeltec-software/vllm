# Step-by-step: bring the decode-PoC post-processing tail into CUDA graph capture

Implements lever #2 from [CUDAGRAPH_MOE.md](CUDAGRAPH_MOE.md) ("Bring the PoC
snap/chain into (or adjacent to) the captured region"). Grounded in the current
code (line numbers as of this checkout) — re-verify against `git blame` before
relying on them if this file has aged.

## Status: implemented and correctness-verified, but it does NOT deliver a
## measured speedup — read this before extending it

Steps 1-8 below were implemented as written (`vllm/poc/tail_cudagraph.py`,
plus the wiring in `mixed_decode.py`/`gpu_model_runner.py`), and step 10's
separation-preserving bar was verified three times: two full
`run_scope.sh` separation sweeps (`allenai/OLMoE-1B-7B-0924-Instruct` vs its
FP8 quant; `stelterlab/DeepSeek-R1-Distill-Qwen-14B-AWQ` vs a GPTQ-4bit quant)
both matched their pre-change baselines within fp-noise on every honest/fraud
pair, and the tail graph is confirmed to actually replay (not silently fall
back) via `graph_inspect.py`'s `count_tail_graph_launches()` (step 8, now
implemented — see the note there).

**But step 9's payoff never materialized.** Same-session, matched-conditions
A/B (`git stash` toggling this exact implementation on/off, immediately
sequential, same box, same request shape) on three models:

| Model | Baseline | With this tail graph | Delta |
|---|---|---|---|
| OLMoE-1B-7B (small MoE) | 463→472 steps/s (1.02×) | 463→474 steps/s (1.02×) | ~0.4%, noise |
| DeepSeek-R1-Distill-Qwen-14B (dense) | 352→453 steps/s (1.29×) | 352→453 steps/s (1.29×) | 0%, identical |
| gpt-oss-20b (larger MoE) | 1158.7 steps/s | 1171.2 steps/s | ~1.1%, noise |

None of these show a reproducible win. (An earlier, informal comparison on
gpt-oss-20b — different `perfomance_nonces.py` invocations run hours apart —
suggested a dramatic ~52% improvement; that number does not survive a fair,
matched-conditions rematch and was a methodology artifact, not a real effect.
Don't trust cross-session timing comparisons on this harness without
controlling for `--duration`/`--warmup` and box state.)

**Root cause, found by directly instrumenting both code paths** (see
`vllm/poc/mixed_decode.py`'s `POC_FORCE_EAGER_TAIL` env-var toggle and the
`record_function` labels `poc_tail_graph_path` / `poc_tail_eager_path` /
`poc_tail_graph_replay`, profiled on gpt-oss-20b, 128 decode-step samples):

| Span | Avg time/call |
|---|---|
| Old eager tail, no graph at all | 6.78ms |
| New graph path (buffer writes + replay + readback) | 7.46ms |
| — of which just `graph.replay()` | 0.15ms |

The graph replay itself is fast and correct. **But it was never the
bottleneck.** ~97% of the tail's cost is in eager code that's IDENTICAL in
both paths — `hidden_states[idxs]` (indexing with a Python list),
`torch.cat([...])` assembling `base_seeds`/`prev_k` from per-request
`[1]`-shaped tensors, and `torch.tensor([...], device=device)` building the
steps tensor from a Python list, every decode step, for up to
`poc_max_batch_size` concurrent requests. Step 2 below identifies this exact
block as "the capturable half" and estimates it at "~6-8 kernel launches" per
`CUDAGRAPH_MOE.md` — that estimate is what's wrong. It's not 6-8 cheap
kernels; it's ~7ms of something (likely a host-device sync inside
`torch.tensor(list, device='cuda')` and/or the list-indexed gather — not
fully root-caused down to the exact CUDA call). Graphing the *downstream
math* (murmur-pick → matmul-vs-codebook → argmax) did nothing about it, and
the `index_copy_`/`index_select` calls step 3-5 add to feed the graph's
buffers are pure additional cost on top — hence the new path being slightly
*slower* than the old one, not faster.

**What a real fix needs, that this implementation doesn't do:** eliminate
the per-step `torch.cat`/`torch.tensor(list)` reassembly itself, not just
graph what consumes its output. `base_seeds`/`prev_k_t` would need to live as
persistent, already-in-place buffer rows from the moment a request is
allocated a slot (written once/updated in-place per step, the same pattern
`PoCNativeState` uses for the reflection path) so there's nothing to
gather-and-concatenate fresh every step — just a direct read. That's a
bigger structural change than steps 1-8 below (which only touch the
downstream math), touching `PoCDecodeState`/`PoCMixedDecodeManager`
(`mixed_decode.py:137-211`) directly rather than adding a manager alongside
them. Anyone picking this up should prototype *that* first — with the same
`POC_FORCE_EAGER_TAIL`-style instrumentation this file's status section used
— before assuming a captured graph helps at all.

The rest of this document is left as-written (steps 1-8, 10) because it's
accurate documentation of what was built and correctly identifies the capture
mechanics, invariants, and verification approach — just not aimed at the
part of the pipeline that actually costs time.

## 0. Before you start — re-verify the baseline

`CUDAGRAPH_MOE.md` describes decode graphs as keyed by `(batch_size, seq_len,
step)`. That is **not** what the current code does: `BatchDescriptor`
(`vllm/forward_context.py:31-59`) only carries `num_tokens`, `num_reqs`,
`uniform`, `has_lora`, `num_active_loras` — no `seq_len`/`step` dimension.
Per-position KV state is threaded through via in-place-updated attention
metadata (slot mappings, block tables), not by capturing a new graph per step.

Also: `vllm/poc/native.py`'s own docstring says it already replaced "the
un-capturable Python forward-hook and the hand-rolled CUDA-graph capture" for
the **inline per-layer reflection** (the Householder transform applied inside
each decoder layer). That path — and the router-force / embedding-substitution
wrappers alongside it — already rides the model's captured graph today via
`PoCLayerWrapper`/`PoCRouterWrapper`/`PoCEmbeddingWrapper`.

So before writing any code: run `python benchmarks/poc/graph_inspect.py`
against your target model and confirm what's *already* graphed vs. not, rather
than assuming lever #1 in `CUDAGRAPH_MOE.md` is still outstanding. The gap this
guide closes is specifically the **post-forward** tail in
`process_poc_outputs_from_hidden` — the per-step sphere-snap bookkeeping that
runs *after* `self.model(...)` returns, which is confirmed still eager (see
step 1).

## 1. Confirm the current capture boundary

The entire captured region is one call: `self.model(...)`.

- `self.model` is wrapped once at load time — `gpu_model_runner.py:5194`:
  ```python
  if cudagraph_mode.has_full_cudagraphs() and not self.parallel_config.use_ubatching:
      self.model = CUDAGraphWrapper(self.model, self.vllm_config, runtime_mode=CUDAGraphMode.FULL)
  ```
- `CUDAGraphWrapper.__call__` (`vllm/compilation/cuda_graph.py:233-339`) is the
  actual `torch.cuda.CUDAGraph()` / `torch.cuda.graph(cudagraph, pool=...,
  stream=...)` capture-and-replay logic, keyed by `BatchDescriptor` via
  `forward_context.batch_descriptor`.
- The PoC tail is called well **after** this, outside any forward-context /
  graph-replay state — `gpu_model_runner.py:4549-4577`:
  ```python
  poc_outputs_dict = None
  if mixed_batch_info:
      ...
      elif mixed_batch_info.get('is_mixed'):
          poc_metadata = mixed_batch_info.get('poc_metadata')
          if poc_metadata and hidden_states is not None:
              poc_outputs_dict = mixed_decode.process_poc_outputs_from_hidden(
                  self, hidden_states, poc_metadata
              )
  ```
  This runs after `_bookkeeping_sync` and after the `with
  set_forward_context(...): ...` block has already exited (`gpu_model_runner.py`
  ends that context at line 4254) — plain eager Python, every decode step.

**Verify this yourself** before changing anything: put a breakpoint or a
`print(torch.cuda.is_current_stream_capturing())` at
`mixed_decode.py:process_poc_outputs_from_hidden`'s entry — it should read
`False` even during a decode step whose model forward replayed a graph.

## 2. Split the tail into a capturable half and a must-stay-eager half

Full function: `vllm/poc/mixed_decode.py:405-548`. The decode block
(`~L497-511`) is the capturable candidate:

```python
lh = hidden_states[idxs].float()                                   # fresh gather, dynamic B
lh = lh / (lh.norm(dim=-1, keepdim=True) + 1e-8)
base_seeds = torch.cat([m['decode_state'].base_seeds for m in decode_metas])  # fresh cat
prev_k = torch.cat([m['decode_state'].prev_k_t for m in decode_metas])        # fresh cat
steps = torch.tensor([m['decode_step'] for m in decode_metas], ...)           # fresh host upload
sph = random_pick_indices_gpu(base_seeds, prev_k, steps, H, SPHERE_DIM, device)
k_all, bad_all = snap_with_guard(project_to_sphere(torch.gather(lh, 1, sph)), codebook)
```
This is on-device, no `.item()`/`.tolist()`, and (per `CUDAGRAPH_MOE.md`'s
count) ~6-8 kernel launches. It is the thing you're trying to graph.

**Correction (see Status section at top):** that "~6-8 kernel launches"
estimate is not what's expensive here. Measured directly: this whole block
costs ~6.8ms/call regardless of whether it's graphed, and the
murmur-pick/matmul/argmax math (the part a graph actually captures) is only
~0.15ms of that. The `hidden_states[idxs]` / `torch.cat([...])` /
`torch.tensor([...], device=device)` lines — kept in this block because
they're what a graph's fixed-address inputs need to be filled from — are the
real cost, and graphing what comes after them doesn't touch it.

The rest (`~L513-546`) must **stay eager**, unconditionally:
```python
for i, meta in enumerate(decode_metas):        # length varies call-to-call
    ...
    if step >= st.max_tokens:                  # data-dependent branch
        k_points = torch.cat(st.k_steps_t).tolist()   # CPU sync
        n_mismatches = int(st.mismatch_t.item())       # CPU sync
        n_nan = int(st.n_nan_t.item())                  # CPU sync
        ...
        get_decode_manager(runner).free(meta['req_id'])
```
A variable-length Python loop and data-dependent `.item()`/`.tolist()` calls
are fundamentally not capturable — don't try. The refactor's job is to make
the capturable half feed this eager half through stable buffers instead of
being fused with it.

## 3. Make the capturable half's inputs/outputs fixed-size and address-stable

CUDA graphs require every captured kernel's inputs and outputs to live at the
**same memory address on every replay**. Today, `lh`, `base_seeds`, `prev_k`,
and `steps` are all freshly allocated (via gather / `torch.cat` / a fresh
`torch.tensor(...)`) with a batch dimension `B = len(decode_metas)` that
varies step to step. Neither the address nor the shape is stable — this is
the actual blocker, not the math itself.

Follow the pattern `PoCNativeState` already uses for the reflection path
(`vllm/poc/native.py:157-272`) — persistent, fixed-`max_batch`-sized buffers,
updated in place before each replay:

```python
self.register_buffer("poc_v", v, persistent=False)      # native.py:77, fixed shape
...
buf[row].copy_(vs[i].to(buf.dtype))                       # native.py:230, in-place update
self.mask[:n].copy_(row_mask)                              # native.py:272, live mask each step
```

Concretely, for the tail:
- Allocate persistent buffers sized `[MAX_POC_DECODE_BATCH, H]` (hidden gather
  target), `[MAX_POC_DECODE_BATCH]` (base_seeds, prev_k, steps, and the
  `k_all`/`bad_all` outputs), once, on the runner or a small state object
  alongside `PoCDecodeState`/`get_decode_manager(runner)`.
- Replace `hidden_states[idxs]` with an `index_copy_`/`copy_` into the
  persistent `[MAX_POC_DECODE_BATCH, H]` buffer (source indices still vary,
  but the *destination* tensor's address doesn't).
- Replace the two `torch.cat([...])` calls (`base_seeds`, `prev_k`) with
  per-request `copy_` writes into persistent `[MAX_POC_DECODE_BATCH]` buffers
  — same technique `set_routing` uses at `native.py:236-264`.
- Replace the fresh `steps = torch.tensor([...], device=device)` host upload
  with a `copy_` into a persistent int64 buffer (or maintain it entirely
  on-device by incrementing a per-slot counter instead of re-uploading from
  Python each step).
- Zero-pad unused rows (`B < MAX_POC_DECODE_BATCH`) with a boolean mask,
  mirroring `native.py`'s `torch.where(mask, transformed, x)` trick
  (`native.py:66`) — so the graph always runs at the fixed `MAX_POC_DECODE_BATCH`
  shape and inactive rows are masked out rather than shrinking the batch.

## 4. Capture the tail as a *separate*, adjacent graph — don't extend the shared chat graph

The chat `CUDAGraphWrapper`'s graph is shared with every pure-chat batch of
the same `BatchDescriptor` (`num_tokens`/`num_reqs`/`uniform`/lora only — no
PoC dimension, `vllm/forward_context.py:31-59`). Splicing PoC-tail math into
that shared graph risks capturing it for chat-only replays too, or capturing
it against the wrong KV/shape context — exactly the class of bug
`CUDAGRAPH_MOE.md`'s invariant section warns about.

Instead, follow the existing precedent for an independent, adjacent graph:
`EncoderCudaGraphManager` (`vllm/v1/worker/encoder_cudagraph.py:50`), which
owns its own `torch.cuda.CUDAGraph()`, is captured from `capture_model()`
right after the main capture loop (`gpu_model_runner.py:6372-6374`):
```python
if self.encoder_cudagraph_manager is not None:
    self.encoder_cudagraph_manager.capture()
```
and is replayed independently alongside the main model graph.

Build an analogous `PoCTailCudaGraphManager`:
- Owns one `torch.cuda.CUDAGraph()` (or a small pool keyed by the padded PoC
  decode-batch size, if you support more than one size — start with a single
  fixed `MAX_POC_DECODE_BATCH` to keep the first version simple).
- `capture()`: run the buffer-writing + `decode_metas` math block from step 3
  once, inside `with torch.cuda.graph(self._graph, pool=..., stream=...):`,
  during `capture_model()` (wire it in next to the encoder-graph capture
  call above).
- `replay()`: `self._graph.replay()`.

## 5. Wire the replay into the runner, between the model call and the eager tail

In `gpu_model_runner.py`, right after `hidden_states = model_output` (the
model-forward call, ~L4262) and **before** the current
`process_poc_outputs_from_hidden` call site (~L4566):
1. Write the current step's `idxs`/`base_seeds`/`prev_k`/`steps` into the
   persistent buffers from step 3 (this part is cheap Python-side bookkeeping,
   stays eager — it's just picking which rows go where; the actual matmuls
   happen in the graph).
2. Call `self.poc_tail_graph_manager.replay()`.
3. Read `k_all`, `bad_all` back out of the persistent output buffers (stable
   address, so this is just an ordinary tensor read — no different from
   reading `entry.output` after a normal `CUDAGraphWrapper` replay).
4. Hand `k_all`/`bad_all` to the now-slimmed-down eager tail (step 2's
   `for i, meta in enumerate(decode_metas): ...` loop + emit-once branch),
   which no longer needs to *compute* `k_all`/`bad_all` itself — only consume
   them.

## 6. Preserve the "decide once, before padding" invariant — don't skip this

This is the non-negotiable item (#4) in `CUDAGRAPH_MOE.md`, and it's the part
most likely to reintroduce the prior wrong-KV bug if rushed.

- The eager-vs-graph / padded-shape decision is made once, at
  `_determine_batch_execution_and_padding` (`gpu_model_runner.py:4014-4027`,
  via `CudagraphDispatcher.dispatch`, `vllm/v1/cudagraph_dispatcher.py:234`) —
  **before** `poc_position_mask`/`poc_metadata` are even computed
  (`gpu_model_runner.py:4142-4191`).
- That decision's output shape drives both `_build_attention_metadata`
  (`gpu_model_runner.py:4110-4128`, `pad_attn = cudagraph_mode ==
  CUDAGraphMode.FULL`) and slot-mapping construction
  (`gpu_model_runner.py:4099-4108`).
- **Rule:** any new PoC-tail-graph dispatch decision (e.g. "should we replay
  the PoC-tail graph this step, given `poc_req_ids`") must be derived from
  the *same* `cudagraph_mode`/`batch_descriptor` chosen at L4014-4027 — never
  re-decided afterward, and never allowed to change what
  `_build_attention_metadata` builds. If the PoC-tail graph needs a mode of
  its own (e.g. it should never replay under `--enforce-eager`), extend the
  existing guard at `gpu_model_runner.py:4196-4200`:
  ```python
  if poc_position_mask is not None and self.model_config.enforce_eager:
      assert cudagraph_mode == CUDAGraphMode.NONE, (
          "PoC batch must be eager under enforce_eager, "
          f"got {cudagraph_mode}"
      )
  ```
  rather than adding a second, independent check elsewhere that could
  disagree with it.
- KV correctness for PoC rows today is "pure dynamic" — no reserved-block
  mechanism (`vllm/poc/mixed_decode.py:1-8`, `setup_decode_poc` at
  `mixed_decode.py:214-244`). The PoC-tail graph you're adding only touches
  `hidden_states`/sphere-codebook math, not KV/attention — so as long as you
  don't touch attention metadata or block tables from the new manager, you
  cannot reintroduce the wrong-KV bug. Keep it that way: the tail graph's
  inputs should be exactly "this step's last-hidden slice + per-row PoC
  decode state," nothing KV-shaped.

## 7. Warm up the tail graph, don't let it capture-on-first-use

Mirror how chat graphs are pre-captured in `capture_model()`
(`gpu_model_runner.py:6314-6402`) rather than captured lazily on the first
live request — a lazy first capture would (a) spike latency on whichever PoC
request happens to trigger it, and (b) is a common source of "worked in my
test, broke in production" bugs if the buffer addresses captured during that
ad hoc first call don't match what later code assumes. Call
`self.poc_tail_graph_manager.capture()` from `capture_model()`, next to the
encoder-graph capture call (`gpu_model_runner.py:6372-6374`), using a dummy
decode-PoC batch to exercise the real code path during warmup.

## 8. Strengthen the coverage signal before declaring this done

**Implemented.** `benchmarks/poc/graph_inspect.py` / `tests/poc/_graph.py`'s
`count_graph_launches` originally only proved *some* graph replayed during a
PoC request (true regardless of this work, since the native-inline
reflection path is already graphed) — not that the tail specifically was.
Fixed by labeling `PoCTailGraphManager.run()`'s `graph.replay()` call with
`record_function(POC_TAIL_GRAPH_REPLAY_LABEL)` (`vllm/poc/tail_cudagraph.py`)
and adding `count_tail_graph_launches()` (`tests/poc/_graph.py`), which
attributes `cudaGraphLaunch` events to the tail graph specifically by
timestamp-containment within that labeled span. `graph_inspect.py` now
reports both totals, and
`tests/poc/integration/test_cudagraph_engaged.py::test_decode_poc_tail_runs_through_its_own_cudagraph`
asserts the tail-specific count is nonzero. Verified on real hardware: `tail
= 8` for an 8-decode-step request, exactly one launch per step.

This signal proves the graph replays correctly — it does NOT prove the
replay is worth anything. See the Status section at the top: it isn't, on
its own, because the graphed portion was never the bottleneck.

## 9. Measure the actual payoff

**Done — see the Status section at the top.** Measured with matched-conditions
same-session A/B (`git stash` toggling this implementation, not cross-session
comparisons — those are unreliable on this harness, see Status) on
`allenai/OLMoE-1B-7B-0924-Instruct` (small MoE), `openai/gpt-oss-20b` (larger
MoE), and `stelterlab/DeepSeek-R1-Distill-Qwen-14B-AWQ` (dense, for
contrast). **No reproducible improvement on any of them** — the PoC ratio
does not move relative to baseline. Root cause: the graphed math was never
the bottleneck (see Status). If re-measuring after a real fix (see Status'
"what a real fix needs"), use the same protocol: `git stash` to toggle the
change on the *same* box in the *same* session, immediately sequential runs,
identical `--duration`/`--warmup`/nonces on both sides — do not compare
numbers taken hours apart.

## 10. Confirm separation still holds — this is not a byte-identity requirement

Per `CUDAGRAPH_MOE.md`: "Acceptance is separation-preserving, NOT
byte-identity." A PoC-aware graph will differ from the eager path at the
floating-point level (different kernel fusion/scheduling) — that's expected
and validation already tolerates fp drift. The actual gate is honest/fraud
separation. Before merging, run the scope harness's separation experiment
(`benchmarks/poc/scope/run_scope.sh <honest> <fraud>`, or `--perf-only` first
to confirm the speedup, then a full run) and confirm honest mismatch rates
stay below the calibrated threshold and fraud rates stay above it, same as
any other change to the decode-PoC path — see
[HOWTO.md](HOWTO.md) and [SESSION_LAYOUT.md](SESSION_LAYOUT.md) for how to run
and read that report.

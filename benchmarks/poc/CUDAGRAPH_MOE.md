# Decode-PoC cudagraph coverage on MoE models — issue + next steps

## Problem
On **MoE** models, decode-PoC gets a much smaller cudagraph speedup than plain
inference on the same server. The PoC per-step post-processing (sphere-snap +
chain + next-step seed) runs **outside** the captured graph, so once the (graphed)
MoE forward is fast, that un-graphed tail dominates and caps the PoC speedup.

## Evidence (measured, MiniMax-M2.7, `--mode both`, cudagraph vs eager)
| GPU | pure-inference cg/eager | decode-PoC cg/eager |
|---|---|---|
| A100 | 4.47× | **1.70×** |
| H100 | 5.63× | **1.90×** |

Pure inference (chat) gets 4.5–5.6× from cudagraph; decode-PoC on the *same server*
only 1.7–1.9×. (Two secondary caveats: the PoC perf window under-saturates for slow
big models — `total_req` 32–64 vs chat 128–192, so re-measure with longer
`--duration`/more nonces; and the perf card's attention-backend label is derived from
the profile name, not the stamped `attention_backend` — unreliable on backends the
profile doesn't name, e.g. sm_103.)

## Root cause (confirmed in code)
- `process_poc_outputs_from_hidden(...)` is called on the forward **output**
  `hidden_states` in `vllm/v1/worker/gpu_model_runner.py` (~L4566), i.e. **after**
  the model call. The cudagraph captures only the model forward, so the whole PoC
  tail is un-graphed.
- The tail itself is already **batched + on-device + emit-once**
  (`vllm/poc/mixed_decode.py`, `process_poc_outputs_from_hidden`, decode block ~L500):
  gather last hidden `[B,H]` → normalize → `random_pick_indices_gpu` → `project_to_sphere`
  → `snap_with_guard` (matmul vs the 16-point codebook + argmax) → device-accumulate.
  So it's not per-request Python and not CPU-synced — but it is **~6–8 un-graphed
  kernel launches per decode step**, which cudagraph cannot remove.

## Why it's MoE-specific (and why this wasn't visible before)
- **Dense** 7B/235B are **compute-bound** → cudagraph (which only removes kernel-launch
  overhead) buys ≈ nothing for *any* workload (measured: 7B w8a16 chat cudagraph ≈ +1%
  at batch 32). The un-graphed PoC tail was therefore invisible, and cudagraphing PoC
  was deliberately shelved as low-value for dense models.
- **MoE** is **launch-bound** (hundreds of small expert kernels per step) → cudagraph
  helps chat a lot (4.5–5.6×). Now the un-graphed PoC tail is the bottleneck. The
  dense-only "not worth it" conclusion does **not** transfer to MoE.

## Levers / next steps (in rough priority)
1. **Collapse the per-step decode graphs (P2).** The decode graph is keyed by
   `(batch_size, seq_len, step)` → up to `max_tokens` graphs (VRAM blowup / likely
   past cudagraph OOMs). Capture **one** graph per `(batch_size, seq_len)` at max
   context and vary per step via `seq_lens`/`paged_kv_last_page_len` + in-place
   block/slot buffers (vLLM's normal decode-graph pattern). Biggest structural win.
2. **Bring the PoC snap/chain into (or adjacent to) the captured region.** The snap is
   a fixed sequence of shape-stable ops (gather → normalize → project → matmul vs a
   constant codebook → argmax → device-accumulate) on a fixed `[B,H]` — a good
   candidate for capture. If a fused capture is too invasive, at least reduce the tail
   to the fewest possible kernel launches (fuse gather+normalize+project; keep the
   codebook resident).
3. **If per-step vector *averaging* is added later** (a separability experiment, N
   Householder reflections per step): batch the N reflections into **one** stacked op,
   not a loop — otherwise it multiplies the un-graphed tail and makes this worse.
4. **Non-negotiable invariant:** the shape the forward runs MUST equal the shape the
   attention metadata was built for; decide eager-vs-graph **once, before padding**.
   Violating this caused prior multi-nonce crashes and a wrong-KV bug. A PoC graph must
   bake in the **reserved-KV block-tables** correctly (replaying the *chat* graph for a
   PoC batch reads the wrong KV → systematically wrong artifacts, not just fp drift).

## How to verify
- **Coverage:** `benchmarks/poc/graph_inspect.py` (+ `tests/poc/_graph.py`) counts
  `cudaGraphLaunch` per PoC request — decode > 0 = graphed, eager == 0. Regression test:
  `tests/poc/integration/test_cudagraph_engaged.py`.
- **Payoff:** re-run `perfomance_nonces.py --mode both` on a **MoE** model (saturated:
  longer `--duration`/more nonces) and compare `decode-PoC cg/eager` to
  `pure-inference cg/eager`. Success = the PoC ratio approaches the pure-inference ratio.
- **Acceptance is separation-preserving, NOT byte-identity.** A dedicated PoC-aware
  graph differs from eager only at the fp level, which validation tolerates; the gate is
  "honest/fraud separation still holds under the graph," not bit-for-bit equality.

## Where to develop
Small MoE that fits a 20 GB card for local iteration: **`allenai/OLMoE-1B-7B-0924-Instruct`**
(~13.8 GB bf16, genuine 64-expert MoE → launch-bound, reproduces the effect without renting HW).

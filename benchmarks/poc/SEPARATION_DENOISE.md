# Lowering the cross-HW honest floor — mismatch-based denoising (two ideas to try)

## Problem
Cross-HW separation is marginal for a *good* quant. Measured (MiniMax-M2.7, one AWQ
fraud, three validators):

| validator | honest same-HW | honest cross-HW | fraud | @ p_mismatch=0.1 |
|---|---|---|---|---|
| A100 | 1.25% | ~6.7% | 9.5% | **MISSED** |
| H100 | 0.56% | — | 11.4% | caught |
| B300 | 0.05% | ~7.0% | 33.9%* | caught |

Two facts: (1) the **cross-HW honest floor is ~7%** — only ~2.7 pt below the A100 fraud
(9.4%), so no fixed threshold cleanly separates on A100; (2) the fraud magnitude is
**hardware-dependent** (9.5%→33.9%, ~3.6×), so a single global `p_mismatch` is not robust.
(\*B300 33.9% is under review — its actual attention backend needs confirming.)

## Root cause (NOT input seeding)
The decode inputs are **already seeded per-nonce** (`gpu_random.py`,
`f"{block_hash}_{public_key}_nonce{nonce}"`). The ~7% is **not** a seeding gap — it's the
**model forward producing slightly different hidden states across GPU archs** (fp /
transcendental / kernel differences), and the **hard snap** to the nearest of 16 codebook
points **amplifies** that tiny jitter into full `sphere_k` boundary flips. Proof:
**same-HW** floor 0.05–2.7% (a GPU agrees with itself) vs **cross-HW** ~7%. A seed cannot
fix this — it controls the model *input*, not the GPU's floating-point *output*.

Framed as SNR: noise `|δ_hw|` (hardware) vs signal `|δ_q|` (quantization/fraud). Production
sits near **SNR≈1** on A100 (the blind spot). Goal: **reduce `δ_hw` without touching `δ_q`.**
Insight: `δ_hw` is isotropic/random, `δ_q` is structured → averaging suppresses the noise
far faster than the signal.

## HARD CONSTRAINT
Keep the metric **mismatch-based**: discrete `sphere_k` mismatch count + fixed `p_mismatch`,
`k-id` still drives the chain, on-chain acceptance rule unchanged. **No** continuous
vector-distance metric. Both ideas below denoise the vector **before** the existing snap;
the emitted artifact stays a discrete per-step `sphere_k`.

## Current decode snap (baseline)
`vllm/poc/mixed_decode.py` `process_poc_outputs_from_hidden` (decode block ~L500):
gather seeded sphere dims → `project_to_sphere` → `snap_with_guard` (matmul vs the
16-point codebook + argmax). **No Householder reflection is applied in the decode path
today** — both ideas add one.

## Idea 1 — per-nonce Householder reflection seed  (FREE; try FIRST)
Apply **one** Householder reflection to the hidden state before the snap, seeded
**per-nonce** (not one seed per block). The mismatch **rate is already an average over the
~N nonces**, so per-nonce-randomized reflections make the honest jitter flip nonces
**independently** → the rate denoises for free; a *shared* per-block reflection lets a
systematic jitter direction flip the *same* boundary across many nonces (correlated →
inflated honest rate). Fraud's structured bias flips regardless of reflection → fraud rate
preserved.
- **Compute: ~zero** — one reflection/nonce (2 dot-products + a scaled subtract,
  O(hidden_dim)); no extra forwards, no cudagraph impact.
- **Expected:** lower honest cross-HW floor, fraud unchanged → wider margin at fixed
  `p_mismatch`. (Offline SNR plots with a per-nonce seed show `δ_hw` dropping ~7×.)

## Idea 2 — per-step average of N reflections  (fallback; costs cudagraph)
Per step, project the hidden state through **N** different Householder reflections, average
the pre-snap coordinate, **then snap the average** → one denoised `sphere_k`. Denoises
*within* a step (∝1/√N), still discrete/mismatch-based.
- **Compute:** the forward stays 1×; adds **N** cheap post-forward projections. Tiny FLOPs,
  but they live in the **un-graphed tail** (see `CUDAGRAPH_MOE.md`) → N× the launch
  overhead. **Batch the N reflections into one stacked op**, never a Python loop, or it
  worsens the MoE cudagraph ratio.
- Use only if Idea 1's per-nonce averaging isn't enough.

## How to evaluate (mismatch-based, no protocol change)
1. Implement Idea 1 (per-nonce reflection seed) in the decode snap.
2. On a **MoE** model that fits a 20 GB card — `allenai/OLMoE-1B-7B-0924-Instruct` — measure
   the **honest cross-HW mismatch rate** (or a same-model, different-seed proxy for jitter)
   and the **fraud mismatch rate**, before vs after.
3. Success = honest floor drops materially while fraud holds → the fixed-`p_mismatch`
   verdict passes with margin (the report already prints per-row `MISSED`/`false-pos` at a
   chosen `--p-mismatch`).
4. If the margin is still thin, add Idea 2 (batched N-average) and repeat.

## Security note
`k-id` continues to seed the chain (sequential-execution guarantee is unchanged). The
reflection only rotates the hidden state before the snap to reduce boundary flips — it does
not change what the prover must compute (still requires running the model).

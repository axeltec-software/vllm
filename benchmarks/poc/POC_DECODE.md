## PoC-decode

**Problem:** A language model answers a request in two phases: it **reads** the prompt once (the *prefill* phase), then **writes** the reply one token at a time (the *decode* phase), which runs once for every token of the answer. For any real request the writing phase is where the overwhelming majority of the compute goes. Yet today's PoC measures a node's contribution from prefill only: it performs a single prefill forward, generates no tokens, and never runs or checks the decode phase. That leaves the largest and most valuable part of a node's work unproven, and opens the door to configurations tuned to score well on PoC while serving real inference slowly, so a node can be fast at the proof and slow at the work the network actually needs. The proof and the paid job drift apart.

**Goal:** Align PoC with inference. The proof should measure and run like the real workload, both in *what* it proves (the decode phase, where most compute happens) and in *how* it runs (at the same performance as production inference), so that a node cannot be fast at PoC and slow at inference, and its measured contribution matches the job the network is actually paid to do.

**Task:** Extend the PoC procedure to cover decode steps, and run it under CUDA graphs (the same mechanism production decode relies on) so proof performance aligns with real inference performance.


## Theory

**Today's PoC (prefill only).** After the prompt is read (prefill), the model holds a hidden state. PoC selects `k_dim=12` dimensions from that hidden state, forms a small vector, transforms it, and returns it as the proof. It then stops: no tokens are generated and the decode phase is never run (`max_tokens = 0`). The existing procedure turns the prefill hidden state into a single fingerprint vector and nothing more.

```
  prompt
    │
    ▼
 ┌─────────┐
 │ prefill │
 └────┬────┘
      │ hidden state
      ▼
  pick 12 dims ──► transform ──► PROOF (one vector)
      │
      ▼
  stop  (decode phase is never run)
```

**PoC-decode** carries the same idea into every decode step, using a sphere.

*Setup.* A unit sphere is prepared with `k` points spread evenly over its surface. The points are derived deterministically from the current block, so the prover and every validator use the identical set for that round.

*Starting from prefill.* The chain is seeded from the prefill hidden state, the same hidden state today's PoC already uses. Instead of stopping at one returned vector, `sphere_dim` (currently 256) of its dimensions are selected, the resulting vector is projected onto the sphere, and the nearest of the `k` points is recorded. This first point is what connects the reading phase to the decode chain that follows.

*Each decode step.* As the model writes each token, the same operation runs on that step's hidden state: select `sphere_dim` dimensions, project the vector onto the sphere, and snap to the nearest point. The chosen point then (a) guides that step and (b) determines which dimensions are read at the *next* step.

Because each step's selection depends on the previous step's point, the steps interlock and must be computed in order; there is no shortcut that skips the actual generation.

```
  prompt
    │
    ▼
 ┌─────────┐   hidden
 │ prefill │──────────► pick dims ──► project on sphere ──► snap ──► k0
 └─────────┘                                                         │
      ┌──── each point k(i) chooses which dims step (i+1) reads ─────┘
      ▼
 ┌──────────┐  hidden
 │ decode 1 │──────────► pick ──► project ──► snap ──► k1
 └────┬─────┘
      ▼
 ┌──────────┐  hidden
 │ decode 2 │──────────► pick ──► project ──► snap ──► k2
 └────┬─────┘
      ▼
     ...  ─────────────────────────────────────────────► kN

  PROOF  = trajectory [k0, k1, k2, … , kN]
  BACKUP = raw pre-snap vector saved for every step (insurance)
```

### Why sample this way, instead of comparing plain inference

A model's plain output is a weak proof on its own: it is short, easy to copy, and reveals nothing about whether the internal computation was actually performed. K-trajectory sampling turns the run *itself* into the evidence. It is a **reproducible, uniform, chained randomization** layered on top of ordinary inference, chosen for three properties:

- **Reproducible.** The reference points and the sampled indices are derived deterministically from the block, so an honest validator running the same model obtains the same trajectory. The model's normal output is not changed.
- **Uniform.** Randomizing *which* hidden-state dimensions are read, instead of fixing them, makes the chosen k-points come out near-uniform across the sphere. There is no small, predictable target an attacker could reproduce cheaply without doing the real work.
- **Chained.** Because each step's sampling depends on the previous step's result, the trajectory has to be produced in sequence by actually running the model step by step. It cannot be precomputed, cached, or shortcut.

Together these make the trajectory a fingerprint of the genuine computation: cheap to check, hard to fake, and read out of inference without altering it.

An interactive 3D sphere plot is available [here](https://axeltec-software.github.io/ReportsHelper/poc-decode/sphere_projection.html), with example code in `notebooks/poc-sphere-projection.ipynb`. K-point selection statistics across different block hashes are available as a [chart](https://axeltec-software.github.io/ReportsHelper/poc-decode/nearest_k_point_dist.html).

**Final output:** an array of selected k-point IDs for every decode step (the trajectory). This sequence is the proof-of-computation for the whole answer. Alongside it, the raw projected vector at each step (before snapping) is retained per token as a finer-grained backup artifact.

### Validation

The method operates in two modes: inference and validation.

- **Inference mode** uses the pipeline described above and produces the trajectory.
- **Validation mode** replays it: the request carries the `k_point_ids` obtained during inference; at each decode step the validator's model produces its own `k_point_id` and compares it against the one from inference. If they differ, a mismatch is recorded, and the inference `k_point_id` is used to continue the chain so the two runs stay aligned.

**Main idea:** both servers return an array of selected k-point IDs for every step. By comparing the two arrays we count how many steps diverged. An honest node (the same model, weights and arithmetic) reproduces almost the whole trajectory; a fraudulent one (a cheaper model, lower precision, or no real computation) drifts onto different points and produces significantly more mismatches.


## Fraud resistance

- The hidden-state indices measured at each step are chosen randomly (seeded by the point from the previous step), which yields a near-uniform k-point distribution and removes any fixed target an attacker could aim at.
- Hardcoding the k-point instead of deriving it from real computation was verified not to be viable: hardcoded variants consistently diverge from the honest pipeline.
- Even two honest runs never match to the last bit, because different GPUs round the arithmetic slightly differently. The k-trajectory metric is measured across the whole run (hundreds of decode steps), so this small per-step noise averages out and the honest mismatch rate stays low and well clear of fraud (see Results).


## Results

The PoC-decode procedure is model-agnostic. The results below are shown for one representative model, **MiniMax-M2**, purely as an illustration of how honest and fraudulent work separate and how the overhead behaves across hardware. The same procedure applies to any model the network serves. This particular run spans four GPU platforms (B300, H100, H200, A100) with the full route window (256):

- **Honest setup:** the same model on *different* hardware (e.g. a run produced on one GPU type validated on another).
- **Fraud setup:** a different quantization of the model (AWQ) validated against the reference model.

### Separation

Honest and fraudulent work are separated by the **k-trajectory metric**: the share of decode steps whose k-point disagrees between the prover and the validator. An honest node reproduces almost the entire trajectory, so its mismatch rate is low; a fraudulent one drifts onto different points, so its rate is high.

| Metric | Value |
| --- | --- |
| Honest mismatch rate | 0% to ~10.4% |
| Fraud mismatch rate | ~13.6% to ~14.2% |
| Acceptance threshold | ≈ 11.98% |

Per pair, the average k-trajectory mismatch (%) with each validator (rows) re-checking each prover (columns):

**Honest (same model):**

| validator ↓ / prover → | B300 | H100 | H200 | A100 |
| --- | --- | --- | --- | --- |
| **B300** | 0.00 | 7.76 | 8.05 | 10.41 |
| **H100** | 7.50 | 0.09 | 7.40 | 9.92 |
| **H200** | 8.07 | 7.41 | 0.08 | 10.03 |

**Fraud (cheaper model, AWQ prover):**

| validator ↓ / prover → | B300 | H100 | H200 |
| --- | --- | --- | --- |
| **B300** | 14.20 | | 13.90 |
| **H100** | 14.00 | 13.57 | 13.84 |
| **H200** | 14.08 | 13.73 | 13.67 |

The diagonal is same-hardware validation and sits near 0%. Every honest cell stays below the ≈11.98% threshold and every fraud cell above it, with a clear gap between the two.

As insurance, a second, continuous artifact is recorded for **every generated token within each nonce**: the projected vector *before* it is snapped to the nearest point. This is the very same kind of vector, compared in the very same way, that the current prefill-based PoC already relies on; the decode channel simply produces one per generated token instead of one per request. Keeping it alongside the discrete k-trajectory means the network always has a proven, familiar fallback: if a case is ever borderline on the k-metric there is additional information to rely on, and on its own this vector also separates honest from fraud.

### Performance and fairness across platforms

PoC is run in its most performant configuration, the same one production inference uses: **CUDA-graph execution** and **large batches**. This is a fairness measure, not only a speed one. When the proof runs as efficiently as real inference, there is no gap between "being fast at PoC" and "being fast at the actual job," so a node cannot tune its setup to score well on the proof while serving inference slowly. Closing that gap removes the room for PoC manipulation and makes reward distribution track genuine inference capacity.

Measured as the **overhead ratio** (proof throughput divided by the node's normal chat throughput), this comes out nearly identical on every platform:

| Platform | Proof (nonce/s) | Chat (req/s) | Overhead ratio |
| --- | --- | --- | --- |
| 1×B300 | 14.86 | 19.16 | 0.775 |
| 4×H100 | 17.62 | 24.10 | 0.731 |
| 4×A100 | 10.65 | 14.93 | 0.713 |
| 2×H200 | 12.93 | 19.84 | 0.652 |

The ratios span only **×1.19** from lowest to highest, so the proof leans on every platform about equally and no hardware is unfairly favored.

A core deliverable of this project was re-building PoC as a **CUDA-graph-based algorithm**. Previously the proof ran step-by-step (eager); now the whole PoC path (prefill and decode) executes inside CUDA graphs, exactly as production inference does. Compared with the step-by-step path, this speeds a node up by up to **~7×** on chat and **~3.5×** on the proof path (measured on B300), which is what lets the proof run at inference speed rather than competing with it.

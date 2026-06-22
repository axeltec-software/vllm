# FA vs FI cross-backend investigation (PARKED)

## Finding
Honest cross-backend (FlashAttn↔FlashInfer) mismatch is ~16.7% on 7B but only ~4.5%
on 235B. Root cause: the sphere snap (16 codebook points in 256-dim) has a tiny
top1-top2 margin (median 0.024) -> it is a hair-trigger amplifier. Mismatch rate is a
fixed function of the relative hidden perturbation ε. FA↔FI perturbs the 7B hidden
~1.2% (3× the engine's 0.4%) -> 17%; on 235B FA↔FI perturbs only ~0.3% -> 4.5%.
NOT a bug — the metric is designed to be sensitive (fraud at ε~2.5% -> 34%).

## Constraint: cannot pin backend (heterogeneous nodes; validator ≠ prover backend).

## Recommendation: TOP-2 TOLERANCE (backend-robust, no pinning, keeps fraud separation)
Match if prover's sphere_k ∈ validator's TOP-2 nearest codebook points.
  hard snap:  engine 6%  · FA↔FI 18% · fraud 34%   (thin)
  top-2:      engine 0.7%· FA↔FI 4%  · fraud 15%   (clean ~3.5×, p_mismatch ~8-10%)  ★
This is enforced-sampling-with-k-candidates applied to the cheap hidden snap (no LM head).
Two artifact options: (A) unchanged (store top-1, validator checks top-2 membership);
(B) store top-2 per step (~2 ints/step, symmetric overlap check).

## TODO to productionize
- Prototype (A) in the validation path (collect.py/validate: compute top-2, mismatch =
  ref_k ∉ top-2). Re-run existing 7B FA/FI references -> confirm 18%->~4% on real data.
- Direct confirmation (when a GPU is free): measure real FA-vs-FI hidden ε on 7B,
  map through fa_fi_sensitivity_sim.py, check it predicts ~17%.
- Decide top-2 vs top-3 by the fraud-margin requirement per model.
See sim: benchmarks/poc/fa_fi_sensitivity_sim.py

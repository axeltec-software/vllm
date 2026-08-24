#!/usr/bin/env python3
"""Artifact-derived acceptance thresholds for the PoC separation channels.

Both separation channels — the discrete k-mismatch RATE and the continuous VECTOR
distance — put honest runs low and fraud runs high, so their acceptance line can be
computed OFFLINE from the per-nonce scores already sitting in the validation
artifacts. No GPU, no re-validation.

  Do NOT confuse this with the margin gate tau. tau is applied per decode step INSIDE
  the validator, so it changes the artifacts themselves — you can only find it by
  re-validating on the GPU at each candidate value (benchmarks/poc/scope/
  calibrate_tau.sh). These thresholds, by contrast, are just read off the finished
  numbers: same function, same rule, for k and for vector.

One source of truth, used by simplify_report.py (the acceptance-line row + chart
lines) and vector_separation.py (offline analysis). Pure python, no deps.
"""
import math


def auc(honest, fraud):
    """P(a fraud nonce scores strictly above a honest one), ties = 0.5.
    1.0 = perfectly separable, 0.5 = indistinguishable. None if a side is empty."""
    if not honest or not fraud:
        return None
    wins = sum((1.0 if f > h else 0.5 if f == h else 0.0) for h in honest for f in fraud)
    return wins / (len(honest) * len(fraud))


def recommend(honest, fraud):
    """The acceptance line separating honest (low) from fraud (high), read off the
    per-nonce scores: the log-midpoint of the gap — geometric mean of the worst honest
    and the best fraud — when both are positive, else the arithmetic midpoint. Returns
    None if either side is empty. Works for ANY 'higher = more fraud' score, so the k
    RATE and the vector DISTANCE go through the identical rule."""
    if not honest or not fraud:
        return None
    hmax, fmin = max(honest), min(fraud)
    if hmax > 0 and fmin > 0:
        return math.sqrt(hmax * fmin)
    return (hmax + fmin) / 2.0


def separation(honest, fraud):
    """Full offline summary for one channel from its per-nonce scores: the recommended
    threshold, AUC, the honest/fraud edges, the gap ratio, and whether they overlap."""
    if not honest or not fraud:
        return None
    hmax, fmin = max(honest), min(fraud)
    return {
        "threshold": recommend(honest, fraud),
        "auc": auc(honest, fraud),
        "honest_max": hmax,
        "fraud_min": fmin,
        "gap": (fmin / hmax) if hmax > 0 else math.inf,
        "overlap": hmax >= fmin,
    }

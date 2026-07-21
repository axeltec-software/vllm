"""The ONE artifact-derived acceptance-line calculator (benchmarks/poc/thresholds.py).

Both separation channels (discrete k-rate, continuous vector distance) put honest low
and fraud high, so their acceptance line is read off the per-nonce scores by the same
rule — geometric mean of the worst honest and the best fraud. Distinct from tau (which
needs GPU re-validation). Pure math, no deps.
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "benchmarks", "poc"))
import thresholds  # noqa: E402


def test_recommend_is_geomean_of_worst_honest_and_best_fraud():
    # honest ≤ 0.2, fraud ≥ 10 → line = sqrt(0.2 * 10) = sqrt(2)
    t = thresholds.recommend([0.05, 0.2, 0.0], [10.0, 20.0, 11.0])
    assert math.isclose(t, math.sqrt(0.2 * 10.0), rel_tol=1e-9)
    # and it sits strictly between the two clusters
    assert max(0.05, 0.2, 0.0) < t < min(10.0, 20.0, 11.0)


def test_recommend_falls_back_to_midpoint_when_an_edge_is_zero():
    # worst honest is exactly 0 → geomean degenerate → arithmetic midpoint
    assert thresholds.recommend([0.0, 0.0], [4.0, 6.0]) == pytest.approx(2.0)


def test_recommend_none_when_a_side_is_empty():
    assert thresholds.recommend([], [1.0]) is None
    assert thresholds.recommend([1.0], []) is None


def test_same_rule_serves_k_rate_and_vector_distance():
    # identical shape, different scales — one function, no special-casing
    k = thresholds.recommend([0.0, 0.23], [10.6])            # percents
    v = thresholds.recommend([2.5e-5, 4.4e-3], [1.0e-1])     # cosine distances
    assert math.isclose(k, math.sqrt(0.23 * 10.6), rel_tol=1e-9)
    assert math.isclose(v, math.sqrt(4.4e-3 * 1.0e-1), rel_tol=1e-9)


def test_auc_is_one_when_perfectly_separable_and_half_when_identical():
    assert thresholds.auc([1.0, 2.0], [3.0, 4.0]) == 1.0
    assert thresholds.auc([1.0, 1.0], [1.0, 1.0]) == 0.5     # all ties


def test_separation_summary_flags_overlap_and_gap():
    clean = thresholds.separation([0.1, 0.2], [10.0, 20.0])
    assert clean["overlap"] is False and clean["gap"] == pytest.approx(50.0) and clean["auc"] == 1.0
    dirty = thresholds.separation([0.1, 5.0], [3.0, 20.0])   # honest_max 5 > fraud_min 3
    assert dirty["overlap"] is True


if __name__ == "__main__":
    import sys as _s
    _s.exit(pytest.main([__file__, "-q"]))

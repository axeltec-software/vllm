"""Throughput must only count nonces that did the FULL work — pure logic, no GPU/server.

perfomance_nonces credits every completed nonce as (max_tokens+1) decode steps. If a nonce
finishes SHORT (e.g. deferred/truncated past a batch cap) that accounting inflates steps/s —
this is exactly how a capped PoC batch once reported an impossible 422 nonce/min. The guard
excludes short trajectories from the count, so a benchmark can no longer report a
plausible-looking number for work that never happened."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks" / "poc"))
from perfomance_nonces import _full_trajectories as F  # noqa: E402

MT = 256
FULL = list(range(MT + 1))          # k0 + one k per decode step


def _resp(*trajs):
    return {"artifacts": [{"k_points_steps": t} for t in trajs]}


def test_full_trajectory_accepted():
    assert F(_resp(FULL), MT) is True
    assert F(_resp(FULL, FULL, FULL), MT) is True


def test_short_trajectory_rejected():
    assert F(_resp(FULL[:10]), MT) is False          # truncated
    assert F(_resp(FULL, FULL[:-1]), MT) is False    # one short among full ones
    assert F(_resp([]), MT) is False                 # empty (pool-exhaustion shape)


def test_missing_or_empty_artifacts_rejected():
    assert F({"artifacts": []}, MT) is False
    assert F({}, MT) is False
    assert F(None, MT) is False


def test_prefill_only_has_no_trajectory_to_check():
    # max_tokens == 0 -> prefill-only, no decode chain; must not be flagged short.
    assert F(_resp([]), 0) is True
    assert F({}, 0) is True

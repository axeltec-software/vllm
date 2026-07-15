"""Windowed/sliced vector artifacts (poc_vector_artifacts).

Wire slices are raw (NOT renormalized); the scorer truncates both sides to
the common leading dims and renormalizes, so full-debug and windowed
artifacts interoperate. Pure CPU math.
"""
import numpy as np

from vllm.poc.data import decode_vector, encode_vector
from vllm.poc.mixed_decode import (_vector_artifact_cfg, encode_sph_slices,
                                   keep_q_step)
from vllm.poc.validation import score_vector_channel


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _traj(vecs, dim=None):
    """Encode prefill + decode-step slices, optionally wire-truncated to dim."""
    rows = [_unit(np.ones(256))] + [_unit(v) for v in vecs]
    if dim is not None:
        rows = [r[:dim] for r in rows]           # raw slice, no renorm (wire)
    return [encode_vector(r) for r in rows]


def test_config_defaults_off_and_sized():
    from vllm.config import CacheConfig
    cc = CacheConfig()
    assert cc.poc_vector_artifacts is False
    assert cc.poc_vector_artifact_steps == 64
    assert cc.poc_vector_artifact_dim == 32


def test_cfg_helper_disabled_without_engine_config():
    class Bare:
        pass
    # only element [0] is load-bearing: with enabled=False every consumer
    # gates on it first, steps/dim are dead
    assert _vector_artifact_cfg(Bare())[0] is False


def test_cfg_helper_reads_cache_config():
    class CC:
        poc_vector_artifacts = True
        poc_vector_artifact_steps = 16
        poc_vector_artifact_dim = 8

    class VC:
        cache_config = CC()

    class Runner:
        vllm_config = VC()

    assert _vector_artifact_cfg(Runner()) == (True, 16, 8)


def test_windowed_slices_score_like_full_vectors():
    """Same trajectories shipped full-256D vs wire-sliced to 32 dims must give
    the same distance up to the slice's own geometry (identical vectors -> 0)."""
    rng = np.random.default_rng(3)
    steps = [rng.standard_normal(256) for _ in range(4)]
    full = [{"nonce": 0, "sph_values_steps": _traj(steps)}]
    sliced = [{"nonce": 0, "sph_values_steps": _traj(steps, dim=32)}]
    s_full = score_vector_channel(full, {0: _traj(steps)})
    s_sliced = score_vector_channel(sliced, {0: _traj(steps, dim=32)})
    assert s_full["mean_dist"] < 1e-5
    assert s_sliced["mean_dist"] < 1e-5
    assert s_sliced["per_nonce"][0]["n_steps_scored"] == 4


def test_mixed_dims_truncate_to_common_leading_slice():
    """A full-256D side scored against a 32-dim windowed side must equal the
    both-sides-32 score (min-dim truncation + renorm)."""
    rng = np.random.default_rng(5)
    ref_steps = [rng.standard_normal(256) for _ in range(3)]
    own_steps = [v + 0.05 * rng.standard_normal(256) for v in ref_steps]
    both32 = score_vector_channel(
        [{"nonce": 0, "sph_values_steps": _traj(own_steps, dim=32)}],
        {0: _traj(ref_steps, dim=32)})
    mixed = score_vector_channel(
        [{"nonce": 0, "sph_values_steps": _traj(own_steps)}],
        {0: _traj(ref_steps, dim=32)})
    assert abs(both32["mean_dist"] - mixed["mean_dist"]) < 1e-4


def test_scorer_renormalizes_raw_slices():
    """Wire slices are not unit vectors; cosine must be computed on the
    renormalized slices, not the raw dot product."""
    v = np.zeros(256, dtype=np.float32)
    v[0], v[32] = 0.6, 0.8                       # unit in 256-D, |slice|=0.6
    ref = [{"nonce": 0, "sph_values_steps": _traj([v], dim=32)}]
    s = score_vector_channel(ref, {0: _traj([v], dim=32)})
    # identical slices: renormalized cosine distance is 0 (raw dot were 0.36)
    assert s["mean_dist"] < 1e-6


def test_window_shorter_than_full_traj_scores_available_steps():
    """Validator with a full trajectory vs a prover artifact windowed to 2
    decode steps: only the common window is scored."""
    rng = np.random.default_rng(9)
    steps = [rng.standard_normal(256) for _ in range(6)]
    own_full = [{"nonce": 0, "sph_values_steps": _traj(steps)}]
    ref_window = {0: _traj(steps[:2], dim=32)}
    s = score_vector_channel(own_full, ref_window)
    assert s["per_nonce"][0]["n_steps_scored"] == 2
    assert s["mean_dist"] < 1e-5


def test_window_keeps_prefill_plus_exactly_va_steps():
    kept = sum(keep_q_step(s, False, True, 64) for s in range(1, 257))
    assert kept == 64                       # + the prefill point kept at setup
    assert all(keep_q_step(s, True, False, 0) for s in range(1, 257))  # debug
    assert not keep_q_step(1, False, False, 64)                        # both off


def test_window_longer_than_chain_degrades_to_full_trajectory():
    assert sum(keep_q_step(s, False, True, 1000) for s in range(1, 257)) == 256


def test_emitter_ships_raw_unrenormalized_slices():
    v = np.zeros((1, 256), dtype=np.float32)
    v[0, 0], v[0, 32] = 0.6, 0.8            # unit in 256-D, |leading 32| = 0.6
    wire = encode_sph_slices(v, debug=False, va_on=True, va_dim=32)
    row = decode_vector(wire[0])
    assert row.shape == (32,)
    assert abs(float(np.linalg.norm(row)) - 0.6) < 1e-3   # raw, NOT renormalized


def test_emitter_dim_beyond_sphere_degrades_to_full_width():
    v = np.random.default_rng(1).standard_normal((2, 256)).astype(np.float32)
    wire = encode_sph_slices(v, debug=False, va_on=True, va_dim=512)
    assert decode_vector(wire[0]).shape == (256,)
    full = encode_sph_slices(v, debug=True, va_on=False, va_dim=32)
    assert decode_vector(full[0]).shape == (256,)          # debug: full width


def test_zero_norm_slice_is_a_fault_not_a_distance():
    good = _unit(np.ones(32))
    zero = np.zeros(32, dtype=np.float32)
    traj = [encode_vector(_unit(np.ones(32))), encode_vector(zero),
            encode_vector(good)]
    s = score_vector_channel([{"nonce": 0, "sph_values_steps": traj}], {0: traj})
    e = s["per_nonce"][0]
    assert e["n_bad_steps"] == 1
    assert e["n_steps_scored"] == 1
    assert e["mean_dist"] < 1e-6
    assert s["n_bad_steps_total"] == 1

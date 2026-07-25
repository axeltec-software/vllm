"""Pre-snap vector artifacts (poc_vector_artifacts).

Emission stores one vector per step (prefill + every decode step), each a SEEDED
k_dim-coordinate pick of the SPHERE_DIM unit vector — the same sampler prefill
uses (random_pick_indices), NOT a leading slice — so prover and validator select
identical coords from the identical step vector. Wire rows are raw (NOT
renormalized); the scorer renormalizes both sides. Pure CPU math.
"""
import numpy as np
import torch

from vllm.poc.data import decode_vector, encode_vector
from vllm.poc.gpu_random import random_pick_indices
from vllm.poc.mixed_decode import (_vector_artifact_cfg, encode_sph_slices,
                                   keep_q_step)
from vllm.poc.sphere import SPHERE_DIM
from vllm.poc.validation import score_vector_channel

BH, PK, NONCE, KDIM = "blk", "pk", 7, 12
_CPU = torch.device("cpu")


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _traj(vecs, dim=None):
    """Encode prefill + decode-step slices, optionally wire-truncated to dim."""
    rows = [_unit(np.ones(256))] + [_unit(v) for v in vecs]
    if dim is not None:
        rows = [r[:dim] for r in rows]           # raw slice, no renorm (wire)
    return [encode_vector(r) for r in rows]


def _q(T, seed=0):
    return np.random.default_rng(seed).standard_normal(
        (T, SPHERE_DIM)).astype(np.float32)


def _fp16(a):
    return np.asarray(a).astype('<f2').astype(np.float32)   # wire round-trip


# --- config / gating -------------------------------------------------------

def test_config_default_off_and_knobs_retired():
    from vllm.config import CacheConfig
    cc = CacheConfig()
    assert cc.poc_vector_artifacts is False
    # dim = k_dim and window = every step now, so the old knobs are gone
    assert not hasattr(cc, "poc_vector_artifact_steps")
    assert not hasattr(cc, "poc_vector_artifact_dim")


def test_cfg_helper_disabled_without_engine_config():
    class Bare:
        pass
    assert _vector_artifact_cfg(Bare()) is False


def test_cfg_helper_reads_cache_config():
    class CC:
        poc_vector_artifacts = True

    class VC:
        cache_config = CC()

    class Runner:
        vllm_config = VC()

    assert _vector_artifact_cfg(Runner()) is True


def test_keep_q_step_keeps_every_step():
    # aligned to the PoC step count: all steps under va_on OR debug, none when off
    assert all(keep_q_step(s, False, True) for s in range(0, 257))
    assert all(keep_q_step(s, True, False) for s in range(0, 257))
    assert not any(keep_q_step(s, False, False) for s in range(0, 257))


# --- emitter: seeded k_dim pick, every step --------------------------------

def test_emitter_covers_every_step_at_k_dim_width():
    q = _q(5)
    wire = encode_sph_slices(q, BH, PK, NONCE, KDIM, debug=False)
    assert len(wire) == 5                              # one vector per step
    for w in wire:
        assert decode_vector(w).shape == (KDIM,)       # width == k_dim, not 32


def test_emitter_uses_seeded_pick_not_leading_slice():
    q = _q(3, seed=1)
    wire = encode_sph_slices(q, BH, PK, NONCE, KDIM, debug=False)
    for step, w in enumerate(wire):
        idx = random_pick_indices(
            BH, PK, [NONCE], SPHERE_DIM, KDIM, _CPU, step=step)[0].numpy()
        # exactly the seeded coords of this step's vector (fp16 round-trip)
        np.testing.assert_array_equal(decode_vector(w), _fp16(q[step][idx]))
        # ...and NOT the leading k_dim coords
        assert not np.array_equal(np.sort(idx), np.arange(KDIM))


def test_prover_and_validator_pick_identical_coords():
    q = _q(4, seed=2)                                  # same seed inputs both sides
    assert (encode_sph_slices(q, BH, PK, NONCE, KDIM, debug=False)
            == encode_sph_slices(q, BH, PK, NONCE, KDIM, debug=False))


def test_pick_varies_by_step():
    # the seed mixes the step, so consecutive steps pick different coords
    i0 = random_pick_indices(BH, PK, [NONCE], SPHERE_DIM, KDIM, _CPU, step=0)[0].tolist()
    i1 = random_pick_indices(BH, PK, [NONCE], SPHERE_DIM, KDIM, _CPU, step=1)[0].tolist()
    assert sorted(i0) != sorted(i1)


def test_pick_varies_by_nonce_and_block():
    q = _q(3, seed=4)
    a = encode_sph_slices(q, BH, PK, NONCE, KDIM, debug=False)
    assert a != encode_sph_slices(q, BH, PK, NONCE + 1, KDIM, debug=False)
    assert a != encode_sph_slices(q, "other", PK, NONCE, KDIM, debug=False)


def test_debug_ships_full_width_unpicked():
    q = _q(2, seed=5)
    wire = encode_sph_slices(q, BH, PK, NONCE, KDIM, debug=True)
    assert len(wire) == 2
    for step, w in enumerate(wire):
        row = decode_vector(w)
        assert row.shape == (SPHERE_DIM,)              # full trajectory under debug
        np.testing.assert_array_equal(row, _fp16(q[step]))


def test_emitter_ships_raw_unrenormalized_slices():
    # a picked slice of a unit vector is generally NOT itself unit — the scorer
    # renormalizes; the wire must carry the raw values.
    q = _unit(np.arange(1, SPHERE_DIM + 1))[None, :].astype(np.float32)
    wire = encode_sph_slices(q, BH, PK, NONCE, KDIM, debug=False)
    idx = random_pick_indices(BH, PK, [NONCE], SPHERE_DIM, KDIM, _CPU, step=0)[0].numpy()
    row = decode_vector(wire[0])
    assert abs(float(np.linalg.norm(row))
               - float(np.linalg.norm(_fp16(q[0][idx])))) < 1e-3


# --- scorer interop (independent of the emitter) ---------------------------

def test_windowed_slices_score_like_full_vectors():
    """Same trajectories shipped full-256D vs wire-sliced must give the same
    distance up to the slice's own geometry (identical vectors -> 0)."""
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
    """A full-256D side scored against a narrower windowed side must equal the
    both-sides-narrow score (min-dim truncation + renorm)."""
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

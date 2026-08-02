"""Batched PoC RNG must be per-row identical to the serial per-nonce path.

Guards the prefill-synth batching (generate_inputs now calls _batched_normal once
instead of a per-nonce _normal loop) — a perf change that MUST NOT alter artifacts.
CPU-runnable (no GPU)."""
import torch

from vllm.poc.gpu_random import (
    _normal, _batched_normal, _seed_from_string, generate_inputs)


def test_batched_normal_matches_per_seed():
    dev = torch.device("cpu")
    seeds = [_seed_from_string(f"deadbeef_cafebabe_nonce{n}") for n in range(6)]
    n = 257
    batched = _batched_normal(seeds, n, dev)          # [B, n]
    for i, s in enumerate(seeds):
        per = _normal(s, n, dev)                       # [n]
        assert torch.equal(batched[i], per), f"row {i} diverged: batched != _normal"


def test_generate_inputs_matches_serial_loop():
    dev = torch.device("cpu")
    bh, pk = "deadbeef" * 8, "cafebabe" * 8
    nonces = [0, 1, 2, 3]
    dim, seq_len = 64, 8
    got = generate_inputs(bh, pk, nonces, dim, seq_len, dev, dtype=torch.float16)
    # reference: the old serial per-nonce computation
    ref = torch.empty(len(nonces), seq_len, dim, dtype=torch.float16)
    for i, nz in enumerate(nonces):
        seed = _seed_from_string(f"{bh}_{pk}_nonce{nz}")
        ref[i] = _normal(seed, seq_len * dim, dev).view(seq_len, dim).to(torch.float16)
    assert torch.equal(got, ref), "batched generate_inputs != serial per-nonce loop"

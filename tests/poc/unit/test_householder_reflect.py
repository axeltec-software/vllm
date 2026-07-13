"""Unit spec for the per-nonce Householder reflection (sphere-snap denoise).

Pure CPU, no GPU/server. Properties that must hold for it to be a valid, reproducible
pre-snap rotation: norm-preserving, deterministic, per-nonce-distinct, off==identity.
Run: vllm-v0.20/.venv/bin/python -m pytest tests/poc/unit/test_householder_reflect.py -q
"""
import torch

from vllm.poc.gpu_random import poc_householder_reflect

CPU = torch.device("cpu")
K = 256  # SPHERE_DIM


def test_norm_preserving():
    x = torch.randn(4, K)
    for n in (1, 2, 3):
        y = poc_householder_reflect("bh", "pk", [0, 1, 2, 3], x, CPU, n)
        assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-4), n


def test_deterministic():
    x = torch.randn(3, K)
    a = poc_householder_reflect("bh", "pk", [0, 1, 2], x, CPU, 2)
    b = poc_householder_reflect("bh", "pk", [0, 1, 2], x, CPU, 2)
    assert torch.equal(a, b)


def test_per_nonce_distinct():
    # same input row, two different nonces -> different reflected vectors
    x = torch.randn(1, K).repeat(2, 1)
    y = poc_householder_reflect("bh", "pk", [0, 1], x, CPU, 1)
    assert not torch.allclose(y[0], y[1])


def test_block_scoped():
    # same nonce, different block_hash -> different reflection (seed includes block+key)
    x = torch.randn(1, K)
    a = poc_householder_reflect("blockA", "pk", [7], x, CPU, 1)
    b = poc_householder_reflect("blockB", "pk", [7], x, CPU, 1)
    assert not torch.allclose(a, b)


def test_off_is_identity():
    x = torch.randn(2, K)
    assert torch.equal(poc_householder_reflect("bh", "pk", [0, 1], x, CPU, 0), x)

"""Unit tests for poc_graph_bucket — pad a PoC batch up to the nearest cudagraph
capture bucket (capped at poc_max_batch_size). Pure logic, no GPU."""
import pytest

from vllm.poc.mixed_decode import poc_graph_bucket, POC_GRAPH_BUCKETS

B = (8, 16, 32)


@pytest.mark.parametrize("n,expected", [
    (1, 8), (2, 8), (7, 8), (8, 8),      # 1..8 -> 8
    (9, 16), (15, 16), (16, 16),         # 9..16 -> 16
    (17, 32), (31, 32), (32, 32),        # 17..32 -> 32
])
def test_nearest_bucket(n, expected):
    assert poc_graph_bucket(n, B, max_batch=32) == expected


def test_capped_at_max_batch():
    # bucket never exceeds max_batch even if a larger bucket would fit
    assert poc_graph_bucket(5, B, max_batch=4) == 4
    assert poc_graph_bucket(20, B, max_batch=16) == 16


def test_above_largest_bucket_falls_back_to_max_batch():
    # n beyond the largest bucket -> capped at max_batch (scheduler caps n<=max anyway)
    assert poc_graph_bucket(40, B, max_batch=32) == 32


def test_monotonic_nondecreasing():
    prev = 0
    for n in range(1, 33):
        b = poc_graph_bucket(n, B, max_batch=32)
        assert b >= n and b >= prev
        prev = b


def test_default_buckets_constant():
    # the module default is used when buckets not passed
    assert poc_graph_bucket(1) == POC_GRAPH_BUCKETS[0]
    assert poc_graph_bucket(32) == 32

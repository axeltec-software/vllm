"""pinned_to_device must be BYTE-IDENTICAL to the direct torch.tensor(list, device)
it replaces — it only changes the host->device transport (pinned + non_blocking) to
avoid the per-step sync stall in the decode-PoC tail / seeded routing. If the values
ever differ, PoC artifacts drift and consensus breaks, so this is the guard.
"""
import pytest
import torch

from vllm.poc.gpu_random import pinned_to_device


@pytest.mark.parametrize("dtype", [torch.int64, torch.long, torch.int32])
@pytest.mark.parametrize("vals", [
    [0], [1, 2, 3], list(range(32)), [255, 0, 17, 3, 99],
    [i * 7 + 1 for i in range(32)],          # decode-step-like
])
def test_identical_to_direct_tensor(dtype, vals):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    direct = torch.tensor(vals, dtype=dtype, device=dev)
    got = pinned_to_device(vals, dtype, dev)
    assert got.dtype == direct.dtype
    assert got.device.type == direct.device.type
    assert torch.equal(got, direct), f"value mismatch: {got.tolist()} != {direct.tolist()}"


def test_usable_as_index_select_index():
    """The tail uses it as an index into hidden_states — must gather the same rows
    as Python-list advanced indexing (the op it replaces)."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden = torch.randn(64, 8, device=dev)
    idxs = [3, 3, 0, 63, 17, 40]                     # last-token rows, dups allowed
    ref = hidden[idxs]                                # what the old code did
    got = hidden.index_select(0, pinned_to_device(idxs, torch.long, dev))
    assert torch.equal(got, ref)


def test_empty_and_singleton():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert torch.equal(pinned_to_device([], torch.int64, dev),
                       torch.tensor([], dtype=torch.int64, device=dev))
    assert pinned_to_device([42], torch.int64, dev).item() == 42


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))

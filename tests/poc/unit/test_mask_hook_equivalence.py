"""Equivalence + safety gates for the mask-mode Householder hook rewrite.

The mask-mode hook (mixed batching) was rewritten from a clone + ``nonzero`` +
scatter into an IN-PLACE, STATIC-SHAPE masked where-blend so it is CUDA-graph
safe (no new-tensor allocation, no data-dependent shape). ``apply_householder`` is
``x - 2*(x·v)*v`` reduced over the last dim only — fully per-row — so transforming
the whole batch then selecting the PoC rows is BIT-IDENTICAL to transforming only
those rows. These tests pin that:

  1. new output == the OLD clone+scatter reference, BIT-for-bit, across shapes;
  2. PoC rows (mask=True) == binary-mode apply_householder; chat rows unchanged;
  3. the transform is in-place (same tensor object, mutated) — no clone.
"""
import torch
import torch.nn as nn
import pytest

from vllm.poc.layer_hooks import LayerHouseholderHook
from vllm.poc.gpu_random import apply_householder

HID = 64


def _make_hook() -> LayerHouseholderHook:
    class _Layer(nn.Module):
        def __init__(self, h):
            super().__init__()
            self.linear = nn.Linear(h, h)

        def forward(self, x):
            return self.linear(x)

    class _Model(nn.Module):
        def __init__(self, n, h):
            super().__init__()
            self.model = nn.ModuleDict(
                {"layers": nn.ModuleList([_Layer(h) for _ in range(n)])}
            )

    return LayerHouseholderHook(_Model(2, HID), "blk_hash", torch.device("cpu"), HID)


def _ref_old(output, poc_mask, v):
    """Faithful copy of the ORIGINAL clone + nonzero + scatter mask-mode transform
    (the behaviour the rewrite must reproduce exactly)."""
    def st(x):
        original_shape = x.shape
        if x.dim() == 3:
            x = x.view(-1, x.shape[-1])
        mask = poc_mask.to(x.device)
        if mask.shape[0] != x.shape[0]:
            if mask.shape[0] < x.shape[0]:
                mask = torch.cat([
                    mask,
                    torch.zeros(x.shape[0] - mask.shape[0], dtype=torch.bool, device=x.device),
                ])
            else:
                mask = mask[:x.shape[0]]
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return x.view(original_shape) if len(original_shape) == 3 else x
        pt = apply_householder(x[idx], v.to(x[idx].dtype))
        res = x.clone()
        res[idx] = pt
        return res.view(original_shape) if len(original_shape) == 3 else res

    if isinstance(output, tuple):
        if len(output) >= 2:
            rest = output[2:] if len(output) > 2 else ()
            return (st(output[0]), st(output[1])) + rest
        return (st(output[0]),)
    return st(output)


def _tensors(output):
    """Flatten an output (tensor or tuple) into a list of tensors (skip None)."""
    if isinstance(output, tuple):
        return [t for t in output if isinstance(t, torch.Tensor)]
    return [output]


@pytest.fixture(scope="module")
def hook():
    return _make_hook()


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("shape", [(8, HID), (2, 4, HID)], ids=["2d", "3d"])
def test_new_equals_old_reference_bitwise(hook, seed, shape):
    """New in-place where-blend == old clone+scatter, bit-for-bit (>=2 PoC rows)."""
    g = torch.Generator().manual_seed(seed)
    total = shape[0] * (shape[1] if len(shape) == 3 else 1)
    x = torch.randn(*shape, generator=g)
    v = torch.randn(HID, generator=g)
    # mask with >=2 PoC positions (the multi-nonce spirit)
    mask = torch.zeros(total, dtype=torch.bool)
    mask[1] = mask[3] = True
    if total > 5:
        mask[5] = True

    ref = _ref_old(x.clone(), mask, v)
    got = hook._apply_selective_transform(x.clone(), mask, v)

    rt, gt = _tensors(ref), _tensors(got)
    assert len(rt) == len(gt)
    for r, gg in zip(rt, gt):
        assert torch.equal(r, gg), "mask-mode rewrite diverged from old behaviour"


def test_poc_rows_match_binary_chat_rows_untouched(hook):
    """PoC rows (mask=True) equal binary-mode apply_householder; chat rows unchanged."""
    g = torch.Generator().manual_seed(7)
    x = torch.randn(6, HID, generator=g)
    v = torch.randn(HID, generator=g)
    mask = torch.tensor([False, True, False, True, False, False])
    x0 = x.clone()

    out = hook._apply_selective_transform(x.clone(), mask, v)
    out_t = _tensors(out)[0]

    expected = apply_householder(x0, v)  # binary mode on all rows
    for i in range(6):
        if mask[i]:
            assert torch.equal(out_t[i], expected[i]), f"PoC row {i} != binary transform"
        else:
            assert torch.equal(out_t[i], x0[i]), f"chat row {i} was modified"


def test_in_place_same_object_no_clone(hook):
    """Mask-mode transform mutates and returns the SAME tensor object (no clone)."""
    g = torch.Generator().manual_seed(3)
    x = torch.randn(5, HID, generator=g)
    v = torch.randn(HID, generator=g)
    mask = torch.tensor([True, False, True, False, True])

    out = hook._apply_selective_transform(x, mask, v)
    assert out is x, "mask-mode hook must mutate in place (same object), not clone"


def test_empty_mask_is_noop(hook):
    """All-False mask leaves every row unchanged."""
    g = torch.Generator().manual_seed(9)
    x = torch.randn(4, HID, generator=g)
    v = torch.randn(HID, generator=g)
    x0 = x.clone()
    out = hook._apply_selective_transform(x, torch.zeros(4, dtype=torch.bool), v)
    assert torch.equal(_tensors(out)[0], x0)


def test_tuple_with_none_residual(hook):
    """Tuple output with residual=None: hidden transformed, None passes through.

    The OLD clone+scatter path CRASHED here (it called the transform on None);
    the in-place rewrite handles it (strictly more robust), so we assert the
    correct behaviour directly rather than against the old reference."""
    g = torch.Generator().manual_seed(4)
    x = torch.randn(4, HID, generator=g)
    v = torch.randn(HID, generator=g)
    mask = torch.tensor([True, False, True, False])
    x0 = x.clone()
    out = hook._apply_selective_transform((x.clone(), None), mask, v)
    assert isinstance(out, tuple) and out[1] is None
    expected = apply_householder(x0, v)
    for i in range(4):
        ref_row = expected[i] if mask[i] else x0[i]
        assert torch.equal(out[0][i], ref_row)


def test_mask_shorter_than_batch_pads(hook):
    """Mask shorter than the batch is zero-padded (extra rows treated as chat)."""
    g = torch.Generator().manual_seed(5)
    x = torch.randn(6, HID, generator=g)
    v = torch.randn(HID, generator=g)
    short = torch.tensor([True, False, True])  # len 3 < 6
    got = hook._apply_selective_transform(x.clone(), short, v)
    ref = _ref_old(x.clone(), short, v)
    assert torch.equal(_tensors(got)[0], _tensors(ref)[0])

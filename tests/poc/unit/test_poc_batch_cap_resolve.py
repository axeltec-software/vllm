"""poc_max_batch_size AUTO resolution — pure logic, no GPU.

The per-step PoC nonce cap used to be a HARDCODED 32, which throttled PoC below the
machine's real concurrency (max_num_seqs) — on a big box inference scales to hundreds
of sequences while PoC stayed pinned at 32. Now 0 == AUTO -> resolve to max_num_seqs
so PoC fills the batch like inference; any explicit >0 is honored. This guards that
resolution so the throttle can't silently come back."""
from vllm.poc.mixed_decode import resolve_poc_max_batch_size as R


def test_auto_zero_resolves_to_max_num_seqs():
    assert R(0, 256) == 256
    assert R(0, 32) == 32
    assert R(0, 1024) == 1024


def test_explicit_override_is_honored():
    assert R(64, 256) == 64       # explicit below the engine limit
    assert R(512, 256) == 512     # explicit above it — operator's call
    assert R(32, 256) == 32       # can still pin to 32 on purpose


def test_auto_scales_with_the_machine():
    # the whole point: a bigger machine (larger max_num_seqs) -> a bigger PoC cap,
    # with NO code change and NO fixed 32 pin.
    assert R(0, 128) > R(0, 32)
    assert R(0, 256) != 32

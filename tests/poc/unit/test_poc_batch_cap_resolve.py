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


# ---- KV-derived cap (regression: 1xB300 hang, 19.08) -----------------------
# AUTO used to copy max_num_seqs, a scheduler knob with no memory awareness:
# 704 nonces x 512 tokens against a 302672-token pool -> livelock, GPU 0%.
from vllm.poc.mixed_decode import poc_kv_capacity

B300_BLOCKS, B300_BLOCK, B300_SEQ = 18917, 16, 512     # 302672 tokens


def test_kv_capacity_matches_pool_arithmetic():
    assert poc_kv_capacity(B300_BLOCKS, B300_BLOCK, B300_SEQ, 0) == 591


def test_kv_capacity_counts_the_decode_budget_too():
    assert poc_kv_capacity(100, 16, 128, 128) == 6
    assert poc_kv_capacity(100, 16, 128, 0) == 12


def test_kv_capacity_unknown_pool_is_zero():
    assert poc_kv_capacity(0, 16, 512, 0) == 0
    assert poc_kv_capacity(None, 16, 512, 0) == 0
    assert poc_kv_capacity(100, 0, 512, 0) == 0


def test_auto_cap_clamped_by_kv_not_max_num_seqs():
    kv = poc_kv_capacity(B300_BLOCKS, B300_BLOCK, B300_SEQ, 0)
    assert R(0, 704, kv) == 591
    assert R(0, 704, kv) * B300_SEQ <= B300_BLOCKS * B300_BLOCK


def test_auto_cap_uses_concurrency_when_tighter():
    assert R(0, 64, 591) == 64


def test_auto_cap_falls_back_when_pool_unknown():
    assert R(0, 704, 0) == 704


def test_explicit_cap_honored_verbatim():
    assert R(128, 704, 591) == 128


def test_decode_manager_pool_never_empty_under_auto():
    """Regression: manager sized its pool from the RAW config value (0 under
    AUTO after lazy resolution) -> empty pool -> prefill emitted pure-path
    artifact, decode chain never ran."""
    from types import SimpleNamespace
    from vllm.poc.mixed_decode import get_decode_manager

    runner = SimpleNamespace(
        cache_config=SimpleNamespace(
            poc_max_batch_size=0, poc_seq_len=64, poc_max_tokens=8,
            num_gpu_blocks=0, block_size=16),
        vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=256)),
    )
    mgr = get_decode_manager(runner)
    assert mgr.allocate("poc-x", nonce=1, seq_len=64, max_tokens=8) is not None

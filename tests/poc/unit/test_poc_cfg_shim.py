# SPDX-License-Identifier: Apache-2.0
"""PoC knobs must be readable from a CacheConfig that does not declare them.

The knobs (`poc_share`, `poc_route_window`, `poc_seq_len`, `poc_max_tokens`,
`poc_max_batch_size`, `poc_vector_artifacts`) are fields of OUR fork's CacheConfig.
The plugin is meant to drop onto a stock vLLM, where those attributes simply do not
exist — a direct `cache_config.poc_share` there raises AttributeError on the first
PoC step, i.e. the plugin only ever ran because it was deployed on the fork.

Every read now goes through ``poc_cfg``, which falls back to the SAME default the
fork declares. That equality is the part worth pinning: if the fork changed a
default and this table did not, a fork deploy and a stock deploy would disagree on
consensus-relevant values (route window, seq_len, max_tokens) with nothing failing.
"""
from types import SimpleNamespace

import pytest

from vllm.poc.admission import PoCAdmission
from vllm.poc.mixed_decode import POC_CONFIG_DEFAULTS, poc_cfg


def test_present_fields_win_over_defaults():
    cfg = SimpleNamespace(poc_share=0.5, poc_route_window=16, poc_seq_len=64,
                          poc_max_tokens=8, poc_max_batch_size=32,
                          poc_vector_artifacts=True)
    assert poc_cfg(cfg, "poc_share") == 0.5
    assert poc_cfg(cfg, "poc_route_window") == 16
    assert poc_cfg(cfg, "poc_max_batch_size") == 32
    assert poc_cfg(cfg, "poc_vector_artifacts") is True


@pytest.mark.parametrize("name,default", sorted(POC_CONFIG_DEFAULTS.items()))
def test_stock_vllm_config_falls_back(name, default):
    """A stock CacheConfig has none of these attributes."""
    assert poc_cfg(SimpleNamespace(), name) == default


def test_unknown_knob_is_a_typo_not_a_silent_none():
    with pytest.raises(KeyError):
        poc_cfg(SimpleNamespace(), "poc_shrae")


def test_defaults_track_the_forks_cache_config():
    """THE guard. A default that drifts from the fork makes a stock deploy compute
    different artifacts than a fork deploy, silently."""
    from vllm.config.cache import CacheConfig

    for name, default in POC_CONFIG_DEFAULTS.items():
        declared = getattr(CacheConfig, name)
        assert declared == default, (
            f"{name}: fork declares {declared!r}, shim defaults to {default!r}")


def _req(poc=False, computed=0, prompt=64):
    return SimpleNamespace(
        poc_params=SimpleNamespace(seq_len=64, max_tokens=16) if poc else None,
        num_computed_tokens=computed, num_prompt_tokens=prompt)


def test_admission_runs_against_a_stock_cache_config():
    """End to end: the scheduler hook must survive a config with no PoC fields —
    this is the call that used to raise AttributeError on stock vLLM."""
    poc = _req(poc=True, computed=5)
    scheduler = SimpleNamespace(
        running=[poc], waiting=[],
        scheduler_config=SimpleNamespace(max_num_seqs=256),
        cache_config=SimpleNamespace(num_gpu_blocks=1024, block_size=16),
    )
    a = PoCAdmission(scheduler, 4096)

    assert a.active is True
    # poc_share defaults to 1.0 and no chat is present -> the whole budget
    assert a.over_budget(poc, 4096) is False
    # poc_max_batch_size defaults to AUTO (0) -> capped by max_num_seqs / KV
    assert a.skip(poc) is False

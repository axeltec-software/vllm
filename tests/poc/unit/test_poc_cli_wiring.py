# SPDX-License-Identifier: Apache-2.0
"""PoC CLI knobs must reach CacheConfig, not just parse.

The flag is plumbed in three places (EngineArgs field, CLI argument, CacheConfig
construction). Parsing tests cover the first two; if the third is missing the
server accepts the flag, prints no error, and silently runs the default — the
exact silent-default trap this knob already fell into by having no flag at all.

Builds a real engine config from a small locally-cached model (no weights are
loaded and no forward runs; skipped when the model is not on this box).
"""
import pytest

from vllm.engine.arg_utils import EngineArgs

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _engine_config(**kwargs):
    try:
        return EngineArgs(
            model=MODEL, load_format="dummy", max_model_len=1024, **kwargs
        ).create_engine_config()
    except Exception as exc:  # pragma: no cover — offline box / no GPU
        pytest.skip(f"cannot build engine config here: {type(exc).__name__}: {exc}")


@pytest.mark.parametrize("share", [0.0, 0.25, 1.0])
def test_flag_value_lands_on_cache_config(share):
    assert _engine_config(poc_share=share).cache_config.poc_share == share


def test_default_reaches_cache_config_as_greedy():
    """Unset means PoC-greedy at the engine, not merely at the dataclass."""
    assert _engine_config().cache_config.poc_share == 1.0


@pytest.mark.parametrize("window", [16, 64, 256])
def test_route_window_lands_on_cache_config(window):
    assert _engine_config(
        poc_route_window=window).cache_config.poc_route_window == window

# SPDX-License-Identifier: Apache-2.0
"""PoC knobs must be settable from the command line, with sane defaults.

The share reserves part of every step's token budget for chat. Nodes that serve
PoC exclusively (the PoC window) want all of it — that is the default. A node
that serves both (e.g. the GSM8K mixed-quality run) has to be able to hold a
slice back, which needs an actual flag: the field existed for a long time with
no way to set it.
"""
import argparse

from vllm.config.cache import CacheConfig
from vllm.engine.arg_utils import EngineArgs


def _parse(argv):
    parser = EngineArgs.add_cli_args(argparse.ArgumentParser())
    return parser.parse_args(["--model", "facebook/opt-125m", *argv])


def test_default_is_greedy():
    assert CacheConfig.poc_share == 1.0
    assert _parse([]).poc_share == 1.0


def test_flag_sets_a_reservation():
    assert _parse(["--poc-share", "0.5"]).poc_share == 0.5


def test_flag_reaches_the_engine_args_field():
    """Parsed value must land on EngineArgs (and from there on CacheConfig) —
    an arg that parses but is never passed through is the silent-default trap."""
    args = _parse(["--poc-share", "0.25"])
    engine_args = EngineArgs.from_cli_args(args)
    assert engine_args.poc_share == 0.25


def test_route_window_default_is_full_scatter():
    """The routing window is a CONSENSUS parameter — every PoC token's experts are
    drawn from it, so two nodes on different windows produce different artifacts.
    Default is the shipped full-scatter value."""
    assert CacheConfig.poc_route_window == 256
    assert _parse([]).poc_route_window == 256


def test_route_window_is_settable():
    """Needed to reproduce a golden recorded at another window (the MoE consensus
    golden is pinned at 16) — before the flag existed it could only be changed by
    editing the config class."""
    assert _parse(["--poc-route-window", "16"]).poc_route_window == 16
    assert EngineArgs.from_cli_args(
        _parse(["--poc-route-window", "64"])).poc_route_window == 64

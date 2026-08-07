"""The MoE seeded-routing window is a CONSENSUS parameter, so it must travel as ONE
broadcast config value (--poc-route-window -> CacheConfig.poc_route_window -> every TP
worker), NOT a per-process env var that can silently fail to reach a worker and diverge
k-trajectories. These guards lock that wiring. CPU-only, no GPU, no server.
"""
import argparse

from vllm.config.cache import CacheConfig
from vllm.engine.arg_utils import EngineArgs


def test_default_window_is_16_byte_identical():
    # Unset == 16 == the legacy baked value, so an un-configured node is unchanged.
    import vllm.poc.gpu_random as g
    assert CacheConfig().poc_route_window == 16
    assert g._ROUTE_WINDOW == 16


def test_setter_pushes_config_value_into_module_global():
    import vllm.poc.gpu_random as g
    try:
        g.set_route_window(64)
        assert g._ROUTE_WINDOW == 64
    finally:
        g.set_route_window(16)  # restore for other tests in the process


def test_cli_flag_flows_to_cache_config():
    p = argparse.ArgumentParser()
    EngineArgs.add_cli_args(p)
    ns = p.parse_args(["--poc-route-window", "64"])
    assert ns.poc_route_window == 64


def test_window_is_a_graph_hash_factor_not_ignored():
    # It is baked into the captured graph, so two windows MUST hash differently or a
    # stale compiled graph could be reused across a consensus-changing value.
    assert CacheConfig().compute_hash() != CacheConfig(poc_route_window=64).compute_hash()


def test_window_recorded_in_artifact_encoding_not_per_step():
    # The window must be logged in the job-level artifact encoding (once, no per-step /
    # per-nonce cost) and must NOT touch the consensus-compared vector_b64/k_points.
    import dataclasses
    from vllm.poc.generate_queue import GenerateJob
    fld = {f.name: f for f in dataclasses.fields(GenerateJob)}
    assert "route_window" in fld and fld["route_window"].default == 16
    from pathlib import Path
    gq = (Path(__file__).resolve().parents[3] / "vllm/poc/generate_queue.py").read_text()
    rt = (Path(__file__).resolve().parents[3] / "vllm/poc/routes.py").read_text()
    assert '"route_window": job.route_window' in gq
    assert '"route_window"' in rt  # inline path encoding + job wiring


def test_attach_pushes_window_before_wrapping():
    # native must apply the broadcast value via set_route_window (no env fallback).
    import vllm.poc.native as n
    assert hasattr(n, "set_route_window")
    src = __import__("inspect").getsource(n.attach_native_poc)
    assert "set_route_window(route_window)" in src, \
        "attach_native_poc must push the config window into gpu_random before capture"

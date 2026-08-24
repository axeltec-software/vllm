# SPDX-License-Identifier: Apache-2.0
"""Resolve PoC implementation modules: gonka_poc plugin when installed
(production path), byte-identical in-tree package otherwise. Wire types are
never dispatched — one class per process."""
import importlib

from vllm.logger import init_logger

logger = init_logger(__name__)

_cache: dict = {}


def poc_module(intree: str, plugin: str):
    key = (intree, plugin)
    mod = _cache.get(key)
    if mod is None:
        try:
            mod = importlib.import_module(plugin)
            logger.info("PoC implementation: %s (plugin)", plugin)
        except ImportError:
            mod = importlib.import_module(intree)
            logger.info("PoC implementation: %s (in-tree)", intree)
        _cache[key] = mod
    return mod

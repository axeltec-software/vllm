"""Shared sizing helpers for the PoC KV-cache block reservation.

A single source of truth so the scheduler (which carves blocks out of the chat
free pool), the worker (which writes PoC KV into the reserved region), and the
API layer (which rejects oversized requests) all agree on the math.

The reservation carves blocks ``[0, poc_reserved_blocks)`` out of the chat free
pool.  ``execute_poc_forward`` writes PoC KV with sequential physical block IDs
starting at 0; the highest block it touches for a batch is exactly
``poc_blocks_needed(batch, seq_len, max_tokens, block_size) - 1``.  As long as a
request's actual footprint stays within the reservation, PoC and chat never
collide.

Note: the scheduler computes the reservation with ``scheduler_block_size``
(``cache_config.block_size`` * DCP * PCP).  PoC itself uses
``cache_config.block_size`` (no context-parallel multiplier) and does not
support context parallelism, so in every supported PoC deployment DCP == PCP == 1
and the two block sizes are equal.
"""
import logging
import math


def poc_blocks_needed(
    batch_size: int, seq_len: int, max_tokens: int, block_size: int
) -> int:
    """Number of KV blocks a single PoC forward pass occupies.

    Matches the physical block layout in ``poc_model_runner.execute_poc_forward``:
    each of ``batch_size`` sequences gets ``ceil((seq_len + max_tokens)/block) + 1``
    blocks (the ``+1`` is the per-sequence decode slack).
    """
    
    #batch_size=64
    #seq_len=256
    #max_tokens=256
    #block_size=16

    res = batch_size * (
        math.ceil((seq_len + max_tokens) / block_size) + 1
    )
    logger = logging.getLogger(__name__)
    logger.warning(
        f"poc_blocks_needed(batch_size={batch_size}, seq_len={seq_len}, max_tokens={max_tokens}, block_size={block_size}) = {res}"
    )

    return res


def poc_reserved_blocks(cache_config, block_size: int) -> int:
    """Blocks reserved for PoC, sized from the configured worst-case PoC params."""
    return poc_blocks_needed(
        cache_config.poc_max_batch_size,
        cache_config.poc_seq_len,
        cache_config.poc_max_tokens,
        block_size,
    )

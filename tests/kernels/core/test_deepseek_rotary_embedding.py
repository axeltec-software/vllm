# SPDX-License-Identifier: Apache-2.0
"""
Tests for miscellaneous utilities
"""

from typing import Optional

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import DeepseekScalingRotaryEmbedding


def rotary_embedding_opcheck(rot,
                             positions: torch.Tensor,
                             query: torch.Tensor,
                             key: Optional[torch.Tensor] = None,
                             offsets: Optional[torch.Tensor] = None):

    cos_sin_cache = rot.cos_sin_cache.to(query.device, dtype=query.dtype)
    if offsets is not None:
        if rot.is_neox_style:
            opcheck(torch.ops._C.rotary_embedding_deepseek_neox_offsets_fused,
                    (query, key, cos_sin_cache, positions, offsets, rot.rotary_dim), test_utils=["test_schema", "test_autograd_registration", "test_faketensor"])
        else:
            opcheck(torch.ops._C.rotary_embedding_deepseek_offsets_fused,
                    (query, key, cos_sin_cache, positions, offsets, rot.rotary_dim), test_utils=["test_schema", "test_autograd_registration", "test_faketensor"])
    else:
        if rot.is_neox_style:
            opcheck(torch.ops._C.rotary_embedding_deepseek_neox_fused,
                    (query, key, cos_sin_cache, positions, rot.rotary_dim), test_utils=["test_schema", "test_autograd_registration", "test_faketensor"])
        else:
            opcheck(torch.ops._C.rotary_embedding_deepseek_fused,
                    (query, key, cos_sin_cache, positions, rot.rotary_dim), test_utils=["test_schema", "test_autograd_registration", "test_faketensor"])


@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("max_position", [11, 4096, 32768])
@pytest.mark.parametrize("is_neox_style", [True, False])
@pytest.mark.parametrize("rotary_dim", [32])
@pytest.mark.parametrize("head_size", [32, 108])
@pytest.mark.parametrize("seq_len", [11, 1024])
@pytest.mark.parametrize("scaling_factor", [0.5, 1.0, 1.5])
def test_rotary_embedding_opcheck(dist_init, device, max_position,
                                  is_neox_style, rotary_dim, head_size,
                                  seq_len, scaling_factor):
    batch_size = 1
    base = 10000
    num_heads = 7
    rot = DeepseekScalingRotaryEmbedding(head_size, rotary_dim, max_position, base,
                          is_neox_style, scaling_factor, torch.bfloat16)

    positions = torch.randint(0,
                              max_position, (batch_size, seq_len),
                              device=device)
    head_stride = head_size

    query = torch.randn(batch_size,
                        seq_len,
                        num_heads,
                        head_stride,
                        dtype=torch.bfloat16,
                        device=device)
    key = torch.randn_like(query)
    query = query[..., :head_size]
    key = key[..., :head_size]
    rotary_embedding_opcheck(rot, positions, query, key)
    offsets = torch.zeros(batch_size * seq_len,
                          device=device,
                          dtype=torch.long)
    rotary_embedding_opcheck(rot, positions, query, key, offsets)


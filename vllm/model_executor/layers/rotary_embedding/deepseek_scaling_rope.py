# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from typing import Optional

import torch

from vllm.platforms import current_platform

from .base import RotaryEmbedding
from .common import (rotate_gptj, rotate_neox, yarn_find_correction_range,
                     yarn_linear_ramp_mask)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with YaRN method.

    Credits to Peng et al. github.com/jquesnelle/yarn
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        # Get n-d magnitude scaling corrected for interpolation.
        self.mscale = float(
            yarn_get_mscale(self.scaling_factor, float(mscale)) /
            yarn_get_mscale(self.scaling_factor, float(mscale_all_dim)) *
            attn_factor)
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style, dtype)

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
        pos_freqs = self.base**(
            torch.arange(0,
                         self.rotary_dim,
                         2,
                         dtype=torch.float,
                         device=current_platform.device_type) /
            self.rotary_dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = yarn_find_correction_range(self.beta_fast, self.beta_slow,
                                               self.rotary_dim, self.base,
                                               self.max_position_embeddings)
        # Get n-d rotational scaling corrected for extrapolation
        inv_freq_mask = (1 - yarn_linear_ramp_mask(
            low, high, self.rotary_dim // 2,
            dtype=torch.float)) * self.extrapolation_factor
        inv_freq_mask = inv_freq_mask.to(device=current_platform.device_type)
        inv_freq = inv_freq_interpolation * (
            1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        t = torch.arange(self.max_position_embeddings * self.scaling_factor,
                         device=current_platform.device_type,
                         dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = (freqs.cos() * self.mscale)
        sin = (freqs.sin() * self.mscale)
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        from vllm import _custom_ops as ops
        if offsets is not None:
            if self.is_neox_style:
                query, key = ops.rotary_embedding_deepseek_neox_offsets_fused(query.contiguous(), 
                                                                 key.contiguous(), 
                                                                 self.cos_sin_cache.contiguous(), 
                                                                 positions.contiguous(), 
                                                                 offsets.contiguous(), self.rotary_dim)
            else:
                query, key = ops.rotary_embedding_deepseek_offsets_fused(query.contiguous(), 
                                                            key.contiguous(), 
                                                            self.cos_sin_cache.contiguous(), 
                                                            positions.contiguous(), 
                                                            offsets.contiguous(), self.rotary_dim)
        else:
            if self.is_neox_style:
                query, key = ops.rotary_embedding_deepseek_neox_fused(query.contiguous(), 
                                                         key.contiguous(), 
                                                         self.cos_sin_cache.contiguous(), 
                                                         positions.contiguous(), self.rotary_dim)
            else:
                query, key = ops.rotary_embedding_deepseek_fused(query.contiguous(), 
                                                                 key.contiguous(), 
                                                                 self.cos_sin_cache.contiguous(), 
                                                                 positions.contiguous(), self.rotary_dim)
        return query, key

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.forward_native(positions, query, key, offsets)

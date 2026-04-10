# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass


@dataclass
class PoCParams:
    """Parameters for a single PoC nonce request."""
    block_hash: str
    public_key: str
    block_height: int
    nonce: int
    seq_len: int = 256
    k_dim: int = 12
    # Decode-mode parameters (enabled by --poc-decode server flag)
    poc_decode: bool = False   # run decode steps after prefill
    max_tokens: int = 0        # number of decode steps (0 = prefill-only)

    def clone(self) -> "PoCParams":
        return PoCParams(
            block_hash=self.block_hash,
            public_key=self.public_key,
            block_height=self.block_height,
            nonce=self.nonce,
            seq_len=self.seq_len,
            k_dim=self.k_dim,
            poc_decode=self.poc_decode,
            max_tokens=self.max_tokens,
        )

    def __post_init__(self):
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}")
        if self.k_dim <= 0:
            raise ValueError(f"k_dim must be positive, got {self.k_dim}")
        if self.max_tokens < 0:
            raise ValueError(f"max_tokens must be >= 0, got {self.max_tokens}")

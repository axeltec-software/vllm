from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PoCState(Enum):
    IDLE = "IDLE"
    GENERATING = "GENERATING"
    STOPPED = "STOPPED"


@dataclass
class PoCConfig:
    """Configuration for a PoC generation round."""
    block_hash: str
    block_height: int
    public_key: str
    node_id: int = 0
    node_count: int = 1
    # 0 = no client-side chunking; the engine batches (capped per step by
    # poc_max_batch_size, which auto-scales to max_num_seqs). Was 32, which would
    # have pinned in-flight nonces regardless of the machine.
    batch_size: int = 0
    seq_len: int = 256
    k_dim: int = 12
    poc_stronger_rng: bool = False
    callback_url: Optional[str] = None

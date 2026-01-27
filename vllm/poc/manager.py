import time
import torch
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass

from .config import PoCConfig, PoCState

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.executor.executor_base import ExecutorBase


@dataclass
class PoCStats:
    total_checked: int = 0
    total_valid: int = 0
    start_time: float = 0.0

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time if self.start_time > 0 else 0.0

    @property
    def rate(self) -> float:
        return self.total_checked / self.elapsed if self.elapsed > 0 else 0.0


class PoCManager:
    """Legacy PoC manager for collective_rpc-based execution.

    In the integrated architecture, routes.py submits PoC nonces through
    engine_client.generate() instead of collective_rpc. This manager is
    kept for backward compatibility.
    """

    def __init__(
        self,
        model_executor: "ExecutorBase",
        model_config,
        vllm_config: "VllmConfig",
    ):
        self.model_executor = model_executor
        self.model_config = model_config
        self.vllm_config = vllm_config
        self.device = self._get_device()

        self.state = PoCState.IDLE
        self.config: Optional[PoCConfig] = None
        self.stats = PoCStats()
        self._nonce_counter = 0

    def _get_device(self) -> torch.device:
        if hasattr(self.model_executor, 'driver_worker'):
            return self.model_executor.driver_worker.device
        return torch.device("cuda:0")

    def init_round(self, config: PoCConfig) -> None:
        if self.state == PoCState.GENERATING:
            raise RuntimeError("Round already in progress")
        self.config = config
        self.stats = PoCStats(start_time=time.time())
        self._nonce_counter = config.node_id
        self.state = PoCState.IDLE

    def start_generate(self) -> None:
        if self.config is None:
            raise RuntimeError("Round not initialized")
        self.state = PoCState.GENERATING

    def stop_round(self) -> None:
        self.state = PoCState.STOPPED

    def get_status(self) -> dict:
        return {
            "state": self.state.value,
            "total_checked": self.stats.total_checked,
            "total_valid": self.stats.total_valid,
            "elapsed_seconds": self.stats.elapsed,
            "rate_per_second": self.stats.rate,
        }

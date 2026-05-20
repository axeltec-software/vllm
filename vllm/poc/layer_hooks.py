"""Per-round layer hooks for structure breaking.

Applies transformations between transformer layers to break
the model learned output structure.

PATCHED: Replaced ContextVar with simple global bool for torch.compile compatibility.
torch.compile/dynamo cannot trace ContextVar.get() calls.
"""
from contextlib import contextmanager
from typing import List

import torch

from .gpu_random import generate_householder_vector, apply_householder

# Simple global flag instead of ContextVar (dynamo-compatible)
# Safe because PoC forward is synchronous and single-threaded
_poc_forward_active_flag: bool = False


@contextmanager
def poc_forward_context():
    """Context manager for PoC forward passes.

    Hooks only transform hidden states when this context is active.
    This allows inference and PoC to coexist without interference.
    """
    global _poc_forward_active_flag
    _poc_forward_active_flag = True
    try:
        yield
    finally:
        _poc_forward_active_flag = False


def is_poc_forward_active() -> bool:
    """Check if PoC forward context is active."""
    return _poc_forward_active_flag


class LayerHouseholderHook:
    """Per-round Householder reflections applied between transformer layers."""

    def __init__(
        self,
        model: torch.nn.Module,
        block_hash: str,
        device: torch.device,
        hidden_size: int,
    ):
        self.hooks: List = []
        self.block_hash = block_hash
        layers = self._find_layers(model)
        # One pre-allocated buffer per layer. GPU address is fixed for
        # the lifetime of this object; only the contents change.
        self.reflection_vectors: List[torch.Tensor] = [
            torch.empty(hidden_size, dtype=torch.float32, device=device)
            for _ in layers
        ]
        self._fill_reflection_vectors(block_hash, hidden_size, device)

    def _fill_reflection_vectors(
        self,
        block_hash: str,
        hidden_size: int,
        device: torch.device,
    ) -> None:
        """Rewrite reflection vector contents in-place for a new block_hash.

        GPU addresses of self.reflection_vectors do not change, so any
        CUDAGraphWrapper graph that reads these tensors remains valid.
        """
        for i, buf in enumerate(self.reflection_vectors):
            seed_str = f"{block_hash}_layer_{i}_householder"
            v = generate_householder_vector(seed_str, hidden_size, device)
            buf.copy_(v.to(buf.dtype))   # in-place; same data_ptr()
        self.block_hash = block_hash

    def update_block_hash(
        self,
        block_hash: str,
        hidden_size: int,
        device: torch.device,
    ) -> None:
        """Refresh reflection vectors for a new block_hash without re-registering hooks.

        Safe to call while piecewise CUDA graphs are live — no re-capture needed.
        """
        self._fill_reflection_vectors(block_hash, hidden_size, device)

    def _find_layers(self, model: torch.nn.Module) -> List[torch.nn.Module]:
        """Find transformer layers in a model-agnostic way."""
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return list(model.model.layers)
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return list(model.transformer.h)
        elif hasattr(model, "layers"):
            return list(model.layers)
        return []

    def _setup(
        self,
        model: torch.nn.Module,
        block_hash: str,
        device: torch.device,
        hidden_size: int,
    ):
        """Register forward hooks on all transformer layers.

        Reflection vectors must be allocated before calling (done in __init__).
        """
        layers = self._find_layers(model)
        self.num_total_layers = len(layers)

        for i in range(len(layers)):
            hook = layers[i].register_forward_hook(self._create_hook(i))
            self.hooks.append(hook)

    def _create_hook(self, layer_idx: int):
        def hook(module, input, output):
            if not is_poc_forward_active():
                return

            v = self.reflection_vectors[layer_idx]

            if isinstance(output, tuple):
                hidden = output[0]
                hidden.copy_(apply_householder(hidden, v.to(hidden.dtype)))
                if len(output) >= 2:
                    residual = output[1]
                    if residual is not None:
                        residual.copy_(apply_householder(residual, v.to(residual.dtype)))
                return output
            else:
                output.copy_(apply_householder(output, v.to(output.dtype)))
                return output

        return hook

    def detach(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.reflection_vectors = []

    @property
    def num_layers(self) -> int:
        """Number of layers with hooks attached."""
        return len(self.hooks)

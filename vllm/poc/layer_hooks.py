"""Per-round layer hooks for structure breaking.

Applies transformations between transformer layers to break
the model's learned output structure.

Supports two modes:
1. Binary mode (legacy): All positions are PoC or all are chat
2. Mask mode (mixed batching): Position mask specifies which positions are PoC

Plain module-level globals are used instead of ContextVar so that
torch.dynamo can trace through get_poc_position_mask() and
is_poc_forward_active() without raising a graph break. vLLM V1 model
execution is single-threaded, so a global flag is safe.
"""
from contextlib import contextmanager
from typing import List, Optional

import torch

from .gpu_random import generate_householder_vector, apply_householder

_poc_forward_active_flag: bool = False
_poc_position_mask_global: Optional[torch.Tensor] = None


@contextmanager
def poc_forward_context():
    """Context manager for PoC forward passes (legacy binary mode).

    Hooks transform ALL hidden states when this context is active.
    This is for backward compatibility with pure-PoC batches.

    Usage:
        with poc_forward_context():
            hidden_states = model(...)  # All positions transformed
    """
    global _poc_forward_active_flag
    _poc_forward_active_flag = True
    try:
        yield
    finally:
        _poc_forward_active_flag = False


@contextmanager
def poc_forward_context_with_mask(poc_mask: torch.Tensor):
    """Context manager for mixed batching with position mask.

    Hooks selectively transform only PoC positions based on the mask.
    Chat positions pass through unchanged.

    Args:
        poc_mask: Boolean tensor [total_tokens] where True = PoC position

    Usage:
        # For mixed batch: chat (50 tokens) + poc (256 tokens)
        poc_mask = torch.cat([
            torch.zeros(50, dtype=torch.bool),   # chat: no transform
            torch.ones(256, dtype=torch.bool),   # poc: transform
        ])
        with poc_forward_context_with_mask(poc_mask):
            hidden_states = model(...)  # Only PoC positions transformed
    """
    global _poc_position_mask_global
    _poc_position_mask_global = poc_mask
    try:
        yield
    finally:
        _poc_position_mask_global = None


def is_poc_forward_active() -> bool:
    """Check if PoC forward context is active (binary mode)."""
    return _poc_forward_active_flag


def get_poc_position_mask() -> Optional[torch.Tensor]:
    """Get the PoC position mask for selective transformation."""
    return _poc_position_mask_global


class LayerHouseholderHook:
    """Per-round Householder reflections applied between transformer layers.

    These hooks apply the same transform to all nonces in a round (determined
    by block_hash). Combined with per-nonce hidden state transforms, this
    provides strong structure breaking.

    Usage:
        # At round init
        hooks = LayerHouseholderHook(model, block_hash, device, hidden_size)

        # Run forward passes...

        # At round end
        hooks.detach()
    """

    def __init__(
        self,
        model: torch.nn.Module,
        block_hash: str,
        device: torch.device,
        hidden_size: int,
    ):
        self.hooks: List = []
        self.reflection_vectors: List[torch.Tensor] = []
        self.block_hash = block_hash
        self._setup(model, block_hash, device, hidden_size)

    def _find_layers(self, model: torch.nn.Module) -> List[torch.nn.Module]:
        """Find transformer layers in a model-agnostic way."""
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return list(model.model.layers)
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            return list(model.transformer.h)
        elif hasattr(model, 'layers'):
            return list(model.layers)
        return []

    def _setup(
        self,
        model: torch.nn.Module,
        block_hash: str,
        device: torch.device,
        hidden_size: int,
    ):
        """Setup hooks on all transformer layers."""
        layers = self._find_layers(model)
        self.num_total_layers = len(layers)

        for i in range(len(layers)):
            seed_str = f"{block_hash}_layer_{i}_householder"
            v = generate_householder_vector(seed_str, hidden_size, device)
            self.reflection_vectors.append(v)

            hook = layers[i].register_forward_hook(self._create_hook(i))
            self.hooks.append(hook)

    def _create_hook(self, layer_idx: int):
        """Create a forward hook that applies Householder reflection.

        Supports two modes:
        1. Binary mode: Transform ALL positions when poc_forward_context() active
        2. Mask mode: Transform only PoC positions when poc_forward_context_with_mask() active

        vLLM decoder layers typically return (hidden_states, residual).
        We must transform BOTH to prevent residual connections from
        preserving untransformed values.
        """
        def hook(module, input, output):
            poc_mask = get_poc_position_mask()

            if poc_mask is not None:
                v = self.reflection_vectors[layer_idx]
                return self._apply_selective_transform(output, poc_mask, v)

            if not is_poc_forward_active():
                return output

            v = self.reflection_vectors[layer_idx]

            def transform(x):
                return apply_householder(x, v.to(x.dtype))

            if isinstance(output, tuple):
                if len(output) >= 2:
                    hidden = output[0]
                    residual = output[1]
                    rest = output[2:] if len(output) > 2 else ()
                    transformed_hidden = transform(hidden)
                    transformed_residual = transform(residual)
                    return (transformed_hidden, transformed_residual) + rest
                else:
                    hidden = output[0]
                    transformed = transform(hidden)
                    return (transformed,)
            else:
                transformed = transform(output)
                return transformed

        return hook

    def _apply_selective_transform(
        self,
        output,
        poc_mask: torch.Tensor,
        v: torch.Tensor,
    ):
        """Apply transformation only to PoC positions (mask=True).

        Chat positions pass through unchanged.

        Args:
            output: Layer output (tensor or tuple)
            poc_mask: Boolean tensor [total_tokens] where True = PoC position
            v: Householder reflection vector
        """
        def selective_transform(x):
            original_shape = x.shape
            if x.dim() == 3:
                x = x.view(-1, x.shape[-1])

            mask = poc_mask.to(x.device)
            if mask.shape[0] != x.shape[0]:
                if mask.shape[0] < x.shape[0]:
                    pad_size = x.shape[0] - mask.shape[0]
                    mask = torch.cat([
                        mask,
                        torch.zeros(pad_size, dtype=torch.bool, device=x.device)
                    ])
                else:
                    mask = mask[:x.shape[0]]

            poc_indices = mask.nonzero(as_tuple=True)[0]

            if poc_indices.numel() == 0:
                return x.view(original_shape) if len(original_shape) == 3 else x

            poc_hidden = x[poc_indices]
            poc_transformed = apply_householder(poc_hidden, v.to(poc_hidden.dtype))

            result = x.clone()
            result[poc_indices] = poc_transformed

            return result.view(original_shape) if len(original_shape) == 3 else result

        if isinstance(output, tuple):
            if len(output) >= 2:
                hidden = output[0]
                residual = output[1]
                rest = output[2:] if len(output) > 2 else ()
                return (selective_transform(hidden), selective_transform(residual)) + rest
            else:
                return (selective_transform(output[0]),)
        else:
            return selective_transform(output)

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

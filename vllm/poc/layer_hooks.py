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

# Graph mode: a STABLE position-mask buffer (fixed address) the hook reads on
# every forward so the where-blend is recorded into the captured graph. Updated
# in-place: PoC positions True for a mixed batch, all-False for plain chat.
_poc_stable_mask: Optional[torch.Tensor] = None
# True ONLY during the manual skip_compiled mixed-graph capture/replay. Gates the
# stable-mask where-blend so it is NEVER traced by dynamo during chat's compiled
# forward (where it would crash) — only recorded in the manual eager capture.
_poc_manual_capture_active: bool = False


def set_stable_poc_mask(mask: Optional[torch.Tensor]) -> None:
    global _poc_stable_mask
    _poc_stable_mask = mask


def get_stable_poc_mask() -> Optional[torch.Tensor]:
    return _poc_stable_mask


def set_manual_capture_active(active: bool) -> None:
    global _poc_manual_capture_active
    _poc_manual_capture_active = active


def manual_capture_active() -> bool:
    return _poc_manual_capture_active


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
        self.block_hash = block_hash
        layers = self._find_layers(model)

        # Allocate once. GPU addresses are fixed for this object's lifetime.
        # Contents are updated in-place by _fill_reflection_vectors / update_block_hash.
        self.reflection_vectors: List[torch.Tensor] = [
            torch.empty(hidden_size, dtype=torch.float32, device=device)
            for _ in layers
        ]
        self._fill_reflection_vectors(block_hash, hidden_size, device)
        # Register forward hooks on construction (reflection buffers already
        # allocated above). Keeps the constructor's contract: building the hook
        # attaches it. Buffers stay at fixed GPU addresses for cudagraph safety;
        # use update_block_hash() to refresh contents without re-capturing.
        self._setup(model, block_hash, device, hidden_size)

    def _fill_reflection_vectors(
        self,
        block_hash: str,
        hidden_size: int,
        device: torch.device,
    ) -> None:
        """Write new values into the pre-allocated reflection vector buffers.

        GPU addresses of self.reflection_vectors do not change, so any
        CUDA graph that has recorded ops reading these tensors remains valid.
        """
        for i, buf in enumerate(self.reflection_vectors):
            seed_str = f"{block_hash}_layer_{i}_householder"
            v = generate_householder_vector(seed_str, hidden_size, device)
            buf.copy_(v.to(buf.dtype))
        self.block_hash = block_hash

    def update_block_hash(
        self,
        block_hash: str,
        hidden_size: int,
        device: torch.device,
    ) -> None:
        """Refresh for a new block_hash without detaching hooks or re-capturing graphs."""
        self._fill_reflection_vectors(block_hash, hidden_size, device)

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
        """Register forward hooks. Reflection vectors must already be allocated."""
        layers = self._find_layers(model)
        self.num_total_layers = len(layers)

        for i in range(len(layers)):
            hook = layers[i].register_forward_hook(self._create_hook(i))
            self.hooks.append(hook)

    def _create_hook(self, layer_idx: int):
        """Apply Householder reflection in binary or mask mode.

            Binary mode (poc_forward_context active): in-place, CUDA-graph safe.
            Mask mode (poc_forward_context_with_mask active): now ALSO in-place +
            static-shape (masked where-blend), so it is CUDA-graph safe too. PoC
            rows transform identically to binary mode; chat rows pass through.
        """
        def hook(module, input, output):
            poc_mask = get_poc_position_mask()
            # Manual mixed-graph capture only: read the stable mask so the
            # where-blend is recorded into the graph. Gated by the capture flag so
            # chat's COMPILED forward never traces the where-blend (dynamo crash).
            if poc_mask is None and manual_capture_active():
                poc_mask = get_stable_poc_mask()

            if poc_mask is not None:
                # Mask mode: in-place + static-shape (where-blend) → CUDA-graph
                # safe. Transforms PoC positions, passes chat positions through.
                v = self.reflection_vectors[layer_idx]
                return self._apply_selective_transform(output, poc_mask, v)

            if not is_poc_forward_active():
                return output

            v = self.reflection_vectors[layer_idx]

            if isinstance(output, tuple):
                hidden = output[0]
                hidden.copy_(apply_householder(hidden, v.to(hidden.dtype)))
                if len(output) >= 2:
                    residual = output[1]
                    if residual is not None:
                        residual.copy_(apply_householder(residual, v.to(residual.dtype)))
                return output    # same tuple object, tensors updated in-place
            else:
                output.copy_(apply_householder(output, v.to(output.dtype)))
                return output
 
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
        # IN-PLACE + STATIC-SHAPE so the masked hook is CUDA-graph safe:
        #   - transform ALL rows (no data-dependent nonzero → static shape),
        #   - blend by mask back into the SAME tensor (copy_, no clone → stable
        #     address).
        # Householder is per-row independent (x - 2(x·v)v), so transforming the
        # whole batch then selecting the PoC rows yields BYTE-IDENTICAL values for
        # the PoC rows vs transforming only those rows; chat rows are written back
        # unchanged. Net behaviour == the previous clone+scatter, but graph-safe.
        def selective_transform(x):
            if x is None:
                return
            x2 = x.view(-1, x.shape[-1]) if x.dim() == 3 else x
            mask = poc_mask.to(device=x2.device)
            if mask.shape[0] != x2.shape[0]:
                if mask.shape[0] < x2.shape[0]:
                    mask = torch.cat([
                        mask,
                        torch.zeros(x2.shape[0] - mask.shape[0],
                                    dtype=torch.bool, device=x2.device),
                    ])
                else:
                    mask = mask[:x2.shape[0]]
            transformed = apply_householder(x2, v.to(x2.dtype))
            # PoC rows <- transformed, chat rows <- unchanged (in-place on x2,
            # which is a view of x → mutates x).
            x2.copy_(torch.where(mask.unsqueeze(-1), transformed, x2))

        if isinstance(output, tuple):
            selective_transform(output[0])
            if len(output) >= 2:
                selective_transform(output[1])
            return output    # same tuple object, tensors updated in-place
        else:
            selective_transform(output)
            return output

    def detach(self):
        """Remove forward hook handles. Reflection vector buffers are retained."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        # Do NOT clear self.reflection_vectors — the CUDA graph still references them.

    @property
    def num_layers(self) -> int:
        """Number of layers with hooks attached."""
        return len(self.hooks)

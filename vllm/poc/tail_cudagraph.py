"""CUDA graph capture/replay for the decode-PoC post-processing tail.

The model forward (including the in-model reflection/router/embedding transforms
in ``vllm/poc/native.py``) is already captured by vLLM's normal cudagraph
machinery. The per-step sphere-snap math in
``mixed_decode.process_poc_outputs_from_hidden`` that runs AFTER the forward
returns is NOT: its inputs (last-hidden gather, per-request seed/step state) are
freshly allocated at a dynamic batch size every decode step, which CUDA graphs
cannot replay.

This module fixes that with the same technique ``PoCNativeState`` uses for the
in-model path: fixed-size, address-stable buffers — one row per
``PoCMixedDecodeManager`` slot (see ``vllm/poc/mixed_decode.py``) — updated via
``index_copy_`` before each replay. The buffers always span the full
``poc_max_batch_size`` rows; only the rows touched by ``index_copy_``/
``index_select`` for *this* step's active requests carry meaningful data, the
rest is harmless stale/zero data from a previous step (rows are computed
independently, so this never cross-contaminates results).

See benchmarks/poc/CUDAGRAPH_POC_TAIL_STEPS.md for the design rationale and the
"decide once, before padding" invariant this module does NOT touch (it never
reads or writes KV/attention state, only the hidden-state gather + sphere math).
"""
import torch

# Shared with tests/poc/_graph.py's count_tail_graph_launches(), which finds
# cudaGraphLaunch events nested inside a profiler span with this name to
# attribute them to this graph specifically (vs. the main model-forward graph).
POC_TAIL_GRAPH_REPLAY_LABEL = "poc_tail_graph_replay"


class PoCTailGraphManager:
    """Captures and replays the sphere-snap math for decode-PoC's post-forward
    tail. One instance per model runner, sized to ``poc_max_batch_size``."""

    def __init__(
        self,
        max_batch_size: int,
        hidden_size: int,
        sphere_dim: int,
        device: torch.device,
    ):
        self.max_batch_size = max_batch_size
        self.hidden_size = hidden_size
        self.sphere_dim = sphere_dim
        self.device = device
        # Persistent, address-stable buffers — always the full max_batch_size rows.
        self.hidden_buf = torch.zeros(
            max_batch_size, hidden_size, dtype=torch.float32, device=device)
        self.base_seeds_buf = torch.zeros(max_batch_size, dtype=torch.int64, device=device)
        self.prev_k_buf = torch.zeros(max_batch_size, dtype=torch.int64, device=device)
        self.steps_buf = torch.zeros(max_batch_size, dtype=torch.int64, device=device)
        self.k_out_buf = torch.zeros(max_batch_size, dtype=torch.int64, device=device)
        self.bad_out_buf = torch.zeros(max_batch_size, dtype=torch.bool, device=device)
        # margin (top1-top2 cos) and the pre-snap sphere vector q — the margin gate
        # and the vector channel both need these, so the graph must emit them too.
        self.margin_out_buf = torch.zeros(max_batch_size, dtype=torch.float32, device=device)
        self.q_out_buf = torch.zeros(max_batch_size, sphere_dim, dtype=torch.float32, device=device)
        self.graph: "torch.cuda.CUDAGraph | None" = None

    @property
    def ready(self) -> bool:
        return self.graph is not None

    def _body(self, codebook: torch.Tensor) -> None:
        from vllm.poc.gpu_random import random_pick_indices_gpu
        from vllm.poc.sphere import project_to_sphere, snap_with_margin

        # Normalize INSIDE the graph (buffer holds raw hidden values written by
        # run()) so replay always reflects whatever's currently in hidden_buf.
        lh = self.hidden_buf / (self.hidden_buf.norm(dim=-1, keepdim=True) + 1e-8)
        sph = random_pick_indices_gpu(
            self.base_seeds_buf, self.prev_k_buf, self.steps_buf,
            self.hidden_size, self.sphere_dim, self.device,
        )
        q = project_to_sphere(torch.gather(lh, 1, sph))
        k_all, bad_all, margin_all = snap_with_margin(q, codebook)
        self.k_out_buf.copy_(k_all)
        self.bad_out_buf.copy_(bad_all)
        self.margin_out_buf.copy_(margin_all)
        self.q_out_buf.copy_(q)

    def capture(self, codebook: torch.Tensor) -> None:
        """Capture the sphere-snap math once. Buffers hold zeros at this point
        (harmless — see module docstring); what matters is the memory ADDRESSES
        get fixed here, and every future ``run()`` writes into those same
        addresses before replaying."""
        with torch.inference_mode():
            self._body(codebook)  # warmup: NOT captured (allocator/workspace init)
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
            self._body(codebook)
        self.graph = graph

    def run(
        self, hidden_states: torch.Tensor, decode_metas: list[dict],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Write this step's active decode-PoC rows into the persistent buffers
        (keyed by each request's stable PoCMixedDecodeManager slot), replay, and
        return ``(k_all, bad_all, margin_all, q_all)`` in the SAME order as
        ``decode_metas`` — a drop-in replacement for the eager
        ``snap_with_margin(project_to_sphere(...))`` block it replaces; callers
        need no other changes."""
        assert self.ready, "PoCTailGraphManager.run() called before capture()"
        idxs = [m['start_idx'] + m['length'] - 1 for m in decode_metas]
        slots = [m['decode_state'].slot for m in decode_metas]
        slot_idx = torch.tensor(slots, dtype=torch.long, device=self.device)

        lh = hidden_states[idxs].float()  # [B, H], B = len(decode_metas) (dynamic)
        self.hidden_buf.index_copy_(0, slot_idx, lh)
        self.base_seeds_buf.index_copy_(
            0, slot_idx,
            torch.cat([m['decode_state'].base_seeds for m in decode_metas]))
        self.prev_k_buf.index_copy_(
            0, slot_idx,
            torch.cat([m['decode_state'].prev_k_t for m in decode_metas]))
        self.steps_buf.index_copy_(
            0, slot_idx,
            torch.tensor([m['decode_step'] for m in decode_metas],
                         dtype=torch.int64, device=self.device))

        # Named unconditionally (unlike vLLM's own record_function_or_nullcontext
        # scopes, which are opt-in via env var) so graph_inspect.py / test_cuda-
        # graph_engaged.py can attribute cudaGraphLaunch events specifically to
        # THIS graph in a torch-profiler trace, distinguishing it from the main
        # model-forward graph's launches — see tests/poc/_graph.py
        # count_tail_graph_launches(). Overhead when not profiling is negligible
        # (a no-op RAII object).
        from torch.autograd.profiler import record_function
        with record_function(POC_TAIL_GRAPH_REPLAY_LABEL):
            self.graph.replay()

        k_all = self.k_out_buf.index_select(0, slot_idx)
        bad_all = self.bad_out_buf.index_select(0, slot_idx)
        margin_all = self.margin_out_buf.index_select(0, slot_idx)
        q_all = self.q_out_buf.index_select(0, slot_idx)
        return k_all, bad_all, margin_all, q_all

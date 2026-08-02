"""Native PoC transform.

The per-layer Householder reflection applied as INLINE layer code, so vLLM's
native torch.compile + cudagraph capture it like any model op (prefill AND decode,
dynamic KV, any backend) — replacing the un-capturable Python forward-hook and the
hand-rolled CUDA-graph capture.

Each decoder layer is wrapped by ``PoCLayerWrapper``, which runs the original layer
then reflects the output (hidden AND residual) on PoC rows only, selected by a
shared boolean mask buffer. Chat rows pass through unchanged (mask False →
``where`` is identity), so one compiled model serves chat and PoC. The reflection
vectors (one per layer, seeded by block_hash) and the mask live in stable buffers
updated in-place each round/step, so replay reads the live values.
"""
import os

import torch
from torch import nn

from .gpu_random import (expert_logits_from_base, generate_householder_vector,
                         route_base_seed, _seed_from_string, pinned_to_device)

# Debug-only TP guard (VLLM_POC_DEBUG_TP=1): PoC reflection vectors / embeds are
# generated per rank from deterministic seeds and MUST be bit-identical across
# tensor-parallel ranks, else rows reflect/inject differently per rank -> corruption.
_DEBUG_TP = os.environ.get("VLLM_POC_DEBUG_TP") == "1"


def _assert_replicated_across_tp(t: torch.Tensor, name: str) -> None:
    """No-op unless VLLM_POC_DEBUG_TP=1 and TP world size > 1. Fingerprints `t`
    (3 moments) and all-gathers across the TP group, asserting bit-equality so a
    per-rank RNG divergence is caught the moment a TP run hits it."""
    if not _DEBUG_TP:
        return
    try:
        import torch.distributed as dist
        from vllm.distributed import (
            get_tensor_model_parallel_group,
            get_tensor_model_parallel_world_size,
        )
    except ImportError:
        return
    if not dist.is_initialized():
        return
    ws = get_tensor_model_parallel_world_size()
    if ws <= 1:
        return
    x = t.detach().to(torch.float64).reshape(-1)
    pos = torch.arange(1, x.numel() + 1, device=x.device, dtype=torch.float64)
    fp = torch.stack([x.sum(), (x * x).sum(), (x * pos).sum()])
    gathered = [torch.empty_like(fp) for _ in range(ws)]
    dist.all_gather(gathered, fp, group=get_tensor_model_parallel_group().device_group)
    for r in range(1, ws):
        if not torch.equal(gathered[0], gathered[r]):
            raise AssertionError(
                f"PoC '{name}' diverged across TP ranks (rank0 vs rank{r}) — "
                "per-rank RNG non-determinism; PoC is not TP-safe in this setup")


def _reflect(x: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked Householder: rows where mask is True -> x - 2*(x·v)*v; else x.
    Per-row independent, static-shape (no data-dependent control flow) -> the
    compiled graph captures it; cudagraph replays it reading live v/mask."""
    dot = (x * v).sum(-1, keepdim=True)
    transformed = x - 2.0 * dot * v
    return torch.where(mask, transformed, x)


class PoCLayerWrapper(nn.Module):
    """Wraps one decoder layer; reflects its output hidden + residual on PoC rows.
    ``v`` is this layer's reflection vector; ``mask`` is the shared per-row PoC mask
    (both stable buffers, updated in place)."""

    def __init__(self, inner: nn.Module, v: torch.Tensor, mask: torch.Tensor):
        super().__init__()
        self.inner = inner
        self.register_buffer("poc_v", v, persistent=False)
        self.register_buffer("poc_mask", mask, persistent=False)

    def forward(self, *args, **kwargs):
        out = self.inner(*args, **kwargs)
        if isinstance(out, tuple):
            hidden = out[0]
            n = hidden.shape[0]
            m = self.poc_mask[:n].unsqueeze(-1)
            v = self.poc_v[:n]  # per-row reflection vectors [n, hidden]
            hidden = _reflect(hidden, v.to(hidden.dtype), m)
            rest = list(out[1:])
            if rest and rest[0] is not None:  # residual
                rest[0] = _reflect(rest[0], v.to(rest[0].dtype), m)
            return (hidden, *rest)
        n = out.shape[0]
        m = self.poc_mask[:n].unsqueeze(-1)
        return _reflect(out, self.poc_v[:n].to(out.dtype), m)


class PoCEmbeddingWrapper(nn.Module):
    """Wraps the token embedding; for PoC rows, replaces the token embeds with the
    deterministic PoC embeds (from a stable buffer). PoC requests carry dummy token
    IDs so the graphed input_ids path runs; this injects the real PoC embeds INSIDE
    the graph. Chat rows keep their token embeds (mask False)."""

    def __init__(self, inner: nn.Module, embeds: torch.Tensor, mask: torch.Tensor,
                 embed_base: torch.Tensor = None, embed_prev_k: torch.Tensor = None,
                 embed_step: torch.Tensor = None, hidden_size: int = 0):
        super().__init__()
        self.inner = inner
        self.hidden_size = hidden_size
        self.register_buffer("poc_embeds", embeds, persistent=False)
        self.register_buffer("poc_mask", mask, persistent=False)
        # SYNTH = EMBEDDING (in-graph): synth the decode input from the chain buffers.
        self._synth = embed_base is not None
        if self._synth:
            self.register_buffer("embed_base", embed_base, persistent=False)
            self.register_buffer("embed_prev_k", embed_prev_k, persistent=False)
            self.register_buffer("embed_step", embed_step, persistent=False)

    def forward(self, input_ids):
        # PoC rows carry a dummy token id whose embedding is overridden below, but
        # under async scheduling that id can be a stale/sentinel value (e.g. -1 from
        # the previous step's sampled-token plumbing) -> out-of-vocab gather crash.
        # Force masked (PoC) rows to a valid in-vocab id (0); their value is unused.
        n = input_ids.shape[0]
        m_rows = self.poc_mask[:n]
        # On-device clamp (no host sync): masked rows -> 0, chat rows unchanged.
        input_ids = torch.where(m_rows, torch.zeros_like(input_ids), input_ids)
        out = self.inner(input_ids)
        m = m_rows.unsqueeze(-1)
        if not self._synth:
            return torch.where(m, self.poc_embeds[:n].to(out.dtype), out)
        # DECODE rows: synth input[step] IN-GRAPH from (base, prev_k, step) — the SAME
        # derivation as gpu_random.generate_decode_inputs_gpu, so byte-identical, but
        # it rides the captured forward (no eager RNG on the host between steps).
        # prev_k<0 rows (prefill) fall back to the pre-filled embed.
        from vllm.poc.gpu_random import (
            _step_seeds, _batched_normal_t, _SALT_DECODE_EMBED)
        seeds = _step_seeds(self.embed_base[:n], self.embed_step[:n],
                            self.embed_prev_k[:n], _SALT_DECODE_EMBED)
        dec = _batched_normal_t(seeds, self.hidden_size, out.device).to(out.dtype)
        is_dec = (self.embed_prev_k[:n] >= 0).unsqueeze(-1)
        poc_e = torch.where(is_dec, dec, self.poc_embeds[:n].to(out.dtype))
        return torch.where(m, poc_e, out)


class PoCSnapWrapper(nn.Module):
    """Wraps the model's FINAL norm. Runs it, then SNAPS the normed last hidden ->
    sphere_k IN-GRAPH for every row (PoC's 'sampler'), reusing the embed_* seed
    buffers + codebook and writing per-row k/bad/margin/q to the state's snap_*
    buffers. Returns the norm output unchanged so the LM head still runs. The runner
    index_selects the decode rows post-forward — no per-step index_copy_ feed, no
    separate tail graph replay (this in-graph snap replaced the old eager tail)."""

    def __init__(self, inner: nn.Module, state):
        super().__init__()
        self.inner = inner
        self._st = state

    def forward(self, *args, **kwargs):
        out = self.inner(*args, **kwargs)
        h = out[0] if isinstance(out, tuple) else out
        st = self._st
        n = h.shape[0]
        from vllm.poc.gpu_random import random_pick_indices_gpu
        from vllm.poc.sphere import project_to_sphere, snap_with_margin
        lh = h.float()
        lh = lh / (lh.norm(dim=-1, keepdim=True) + 1e-8)
        sph = random_pick_indices_gpu(
            st.embed_base[:n], st.embed_prev_k[:n], st.embed_step[:n],
            st.hidden_size, st.sphere_dim, h.device)
        q = project_to_sphere(torch.gather(lh, 1, sph))
        k_all, bad_all, margin_all = snap_with_margin(q, st.codebook)
        st.snap_k[:n].copy_(k_all)
        st.snap_bad[:n].copy_(bad_all)
        st.snap_margin[:n].copy_(margin_all)
        st.snap_q[:n].copy_(q)
        return out


class PoCRouterWrapper(nn.Module):
    """Wraps an MoE gate (router Linear). For PoC rows (mask True) it REPLACES the
    router logits with deterministic, hidden-INDEPENDENT seeded logits, so MoE
    expert selection (and gate weights) no longer read the noise-prone hidden ->
    removes the routing nondeterminism that drives the decode-PoC honest floor.
    Chat rows (mask False) keep their natural logits untouched.

    The seeded logits are computed HERE, INSIDE the forward — i.e. INSIDE the
    captured cudagraph — from this layer's cached seed base ([max_tokens] int64)
    and the shared per-row decode ``step`` buffer. Both are address-stable and
    updated in place by ``set_routing`` (the graph reads live values), so per step
    the eager path only bumps a tiny [B] step scalar; the Fisher-Yates selection
    (pure integer, no topk/scores/ties -> bit-identical eager==graph) rides in the
    graph instead of stalling the decode pipeline as an eager tail. Static shape ->
    cudagraph-safe."""

    def __init__(self, inner: nn.Module, route_base: torch.Tensor,
                 route_step: torch.Tensor, n_experts: int, top_k: int,
                 mask: torch.Tensor):
        super().__init__()
        self.inner = inner
        self.n_experts = n_experts
        self.top_k = top_k
        self.register_buffer("poc_route_base", route_base, persistent=False)  # [max_tokens] int64
        self.register_buffer("poc_route_step", route_step, persistent=False)  # [max_tokens] int64 (shared)
        self.register_buffer("poc_mask", mask, persistent=False)

    def __getattr__(self, name: str):
        # Delegate unknown attributes (e.g. `.weight`, quant scales) to the wrapped
        # gate, so backends that read gate attributes directly (FlashInfer MoE init)
        # still resolve them — FlashAttention doesn't, which is why it worked there.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("inner"), name)

    def forward(self, *args, **kwargs):
        out = self.inner(*args, **kwargs)
        logits = out[0] if isinstance(out, tuple) else out
        n = logits.shape[0]
        m = self.poc_mask[:n].unsqueeze(-1)
        forced = expert_logits_from_base(                   # in-graph seeded selection
            self.poc_route_base[:n], self.poc_route_step[:n],
            self.n_experts, self.top_k, logits.device).to(logits.dtype)
        logits = torch.where(m, forced, logits)
        return (logits, *out[1:]) if isinstance(out, tuple) else logits


class PoCNativeState:
    """Per-model PoC transform state: PER-ROW reflection vectors per wrapped layer,
    a shared row mask, and a PoC-embeds buffer. Held on the runner; updated each
    round (block_hash) / step (mask, embeds) in place so the captured graph reads
    live values.

    The reflection vectors are per-row ([max_tokens, hidden]) so requests with
    DIFFERENT block_hashes can share one forward batch without stepping on each
    other (each row reflects with its own block's vectors).
    """

    def __init__(self, num_layers: int, hidden_size: int, max_tokens: int,
                 device, dtype):
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_tokens = max_tokens
        self.vectors = [
            torch.zeros(max_tokens, hidden_size, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.mask = torch.zeros(max_tokens, dtype=torch.bool, device=device)
        self.embeds = torch.zeros(max_tokens, hidden_size, device=device, dtype=dtype)
        # (block_hash, nonce-or-None) -> per-layer vectors. nonce=None is the
        # per-block scheme (one draw shared by every nonce of the block); an int
        # nonce is the per-nonce scheme (each nonce gets its own draw).
        self._hash_cache: dict[tuple, list] = {}
        self._last_refl_key: tuple | None = None    # skip redundant per-step rescatter
        # seeded-routing (MANDATORY for MoE; filled by attach_native_poc): per-MoE-layer
        # cached seed base [max_tokens] int64 (block_hash,nonce,layer, hashed once per
        # mapping) + a SHARED per-row decode-step buffer. The forced-logit selection
        # itself runs in-graph inside PoCRouterWrapper.forward; per step set_routing
        # only writes the step buffer (the graph reads base+step live). Static shape.
        self.route_step = torch.zeros(max_tokens, dtype=torch.int64, device=device)
        self.router_meta: list = []                 # [(n_experts, top_k), ...]
        self._route_base: list = []                 # per-layer [max_tokens] int64 sha256 base (cached)
        self._base_key: tuple | None = None         # (hashes,nonces) the base was built for
        self._last_route_key: tuple | None = None   # skip refresh if (hashes,nonces,steps) unchanged
        # PoC-as-a-sampler, part 1: SYNTH = EMBEDDING. The next decode input is the
        # "embedding" of the sampled sphere_k — synthesized IN-GRAPH in the embedding
        # wrapper from these per-row buffers, so it rides vLLM's standard per-step
        # flow with ~zero eager CPU (like chat's token->embed). prev_k<0 = non-decode.
        self.embed_base = torch.zeros(max_tokens, dtype=torch.int64, device=device)
        self.embed_prev_k = torch.full((max_tokens,), -1, dtype=torch.int64, device=device)
        self.embed_step = torch.zeros(max_tokens, dtype=torch.int64, device=device)
        # PoC-as-a-sampler, part 2: SNAP = SAMPLING. A wrapper on the final norm snaps
        # the last hidden -> sphere_k IN-GRAPH (reusing the embed_* seed buffers + the
        # codebook), writing per-row k/bad/margin/q here. The runner index_selects the
        # decode rows post-forward — no separate tail-graph feed (4 index_copy_/step).
        from .sphere import SPHERE_DIM, get_sphere_codebook
        self.sphere_dim = SPHERE_DIM
        self.snap_k = torch.zeros(max_tokens, dtype=torch.int64, device=device)
        self.snap_bad = torch.zeros(max_tokens, dtype=torch.bool, device=device)
        self.snap_margin = torch.zeros(max_tokens, dtype=torch.float32, device=device)
        self.snap_q = torch.zeros(max_tokens, SPHERE_DIM, dtype=torch.float32, device=device)
        self.codebook = get_sphere_codebook().to(device=device).float().contiguous()

    def set_embeds(self, row_embeds: torch.Tensor) -> None:
        """Write the PoC rows' input embeds into the buffer (in place)."""
        n = row_embeds.shape[0]
        self.embeds[:n].copy_(row_embeds)
        _assert_replicated_across_tp(self.embeds[:n], "embeds")

    def set_decode_chain(self, row_base: torch.Tensor, row_prev_k: torch.Tensor,
                         row_step: torch.Tensor) -> None:
        """Publish per-row (base, prev_k, step) so the embedding wrapper synths the
        decode input IN-GRAPH (like set_routing does for the router). Cheap [n]
        uploads; rows with prev_k<0 are non-decode (prefill/chat)."""
        n = row_prev_k.shape[0]
        self.embed_base[:n].copy_(row_base)
        self.embed_prev_k[:n].copy_(row_prev_k)
        self.embed_step[:n].copy_(row_step)

    # Device-side cache bound: per-nonce seeding adds one entry per (block_hash,
    # nonce), so a 128-nonce round is ~128 entries. Entries are stored in the
    # reflection-buffer dtype (num_layers x hidden, fp16/bf16 -> e.g. ~0.75 MB
    # for 94 x 4096), so the cap bounds the cache at ~200 MB worst case — memory
    # allocated AFTER vLLM's startup profiling, so it must stay small. Clear
    # wholesale past the cap (regeneration is cheap, seeded host murmur).
    _HASH_CACHE_MAX = 256

    def _vectors_for(self, block_hash: str, nonce: int | None = None) -> list:
        """Per-layer reflection vectors for (block_hash, nonce) (cached across
        forwards). nonce=None -> the per-block seed (production default); an int
        nonce -> that nonce's own seed, so every nonce reflects with an
        independent draw."""
        key = (block_hash, nonce)
        vs = self._hash_cache.get(key)
        if vs is None:
            if len(self._hash_cache) >= self._HASH_CACHE_MAX:
                self._hash_cache.clear()
            suffix = ("" if nonce is None else f"_nonce{nonce}")
            # Cache in the buffer dtype: halves the footprint vs the generator's
            # fp32 and matches what the scatter writes (copy_ casts identically).
            dt = self.vectors[0].dtype
            vs = [
                generate_householder_vector(
                    f"{block_hash}{suffix}_layer_{i}_householder",
                    self.hidden_size, self.device).to(dt)
                for i in range(self.num_layers)
            ]
            self._hash_cache[key] = vs
        return vs

    def set_row_block_hashes(self, row_hashes: list,
                             row_refl_nonces: list | None = None) -> None:
        """Write each row's reflection vectors from ITS OWN block_hash (in place),
        so requests with different block_hashes coexist in one forward. row_hashes[i]
        = block_hash for row i, or None (left zero; masked out). Generation of the
        vectors is cached per (block_hash, nonce); the scatter is cheap CPU-side setup.

        row_refl_nonces[i] (optional) = the row's nonce when its request runs
        per-nonce reflection seeding, else None (per-block seed, the default —
        omitting the argument reproduces the legacy behavior bit-exactly).

        The reflection vectors do not depend on the decode step, so for a stable
        batch the row->seed mapping is unchanged across all decode steps. Skip
        the zero + per-(row,layer) copy_ (num_layers x B kernels) when the mapping
        matches the last call; the buffers already hold the right values."""
        if row_refl_nonces is None:
            row_refl_nonces = [None] * len(row_hashes)
        refl_key = (tuple(row_hashes), tuple(row_refl_nonces))
        if refl_key == self._last_refl_key:
            return
        for buf in self.vectors:
            buf.zero_()
        for row, (bh, nz) in enumerate(zip(row_hashes, row_refl_nonces)):
            if bh is None:
                continue
            vs = self._vectors_for(bh, nz)
            for i, buf in enumerate(self.vectors):
                buf[row].copy_(vs[i].to(buf.dtype))
        self._last_refl_key = refl_key
        _assert_replicated_across_tp(self.vectors[0], "reflection_vectors[0]")
        # (reflection vectors depend on block_hash [+ nonce when per-nonce seeded],
        # never the step; routing also depends on step so it is refreshed
        # separately, per step, via set_routing.)

    def set_routing(self, row_hashes, row_nonces, row_steps) -> None:
        """Refresh PER-ROW seeded router logits — MANDATORY for MoE. EFFICIENT (the
        K-calc discipline):
          * the sha256 BASE (block_hash,nonce,layer) is hashed ONCE per mapping and
            cached in [max_tokens] int64 buffers — NOT per step,
          * each step only folds `step` ON GPU (expert_logits_from_base): two integer
            murmur kernels per layer, batched [B, n_experts], copied GPU->GPU into the
            static buffer IN PLACE.
        So per step there is NO host string-hashing, NO device->host sync, and the
        captured graph (which only READS the buffer) needs no recapture. Rows with
        block_hash None get base 0 (masked out anyway)."""
        if not self._route_base:
            return
        base_key = (tuple(row_hashes), tuple(row_nonces))
        if base_key != self._base_key:                       # rebuild cached base (host, ONCE/mapping)
            for i, base_buf in enumerate(self._route_base):
                vals = [_seed_from_string(route_base_seed(bh, nz, i)) if bh is not None else 0
                        for bh, nz in zip(row_hashes, row_nonces)]
                base_buf[:len(vals)].copy_(
                    torch.tensor(vals, dtype=torch.int64, device=self.device))
            self._base_key = base_key
        key = (base_key, tuple(row_steps))
        if key == self._last_route_key:                      # nothing changed -> skip
            return
        b = len(row_steps)
        # Per step, ONLY publish the decode step into the shared buffer (tiny [B]
        # upload, sync-free; a direct torch.tensor(list, device=cuda) would block the
        # forward — see pinned_to_device). The Fisher-Yates forced-logit selection
        # runs in-graph in PoCRouterWrapper.forward, reading base+step live.
        self.route_step[:b].copy_(pinned_to_device(row_steps, torch.int64, self.device))
        self._last_route_key = key

    def set_mask(self, row_mask: torch.Tensor | None) -> None:
        """Set which rows are PoC this forward (in place). None -> all chat."""
        self.mask.zero_()
        if row_mask is not None:
            n = row_mask.shape[0]
            self.mask[:n].copy_(row_mask)


def attach_native_poc(model: nn.Module, layers: list, embed_owner, max_tokens: int,
                      hidden_size: int, device, dtype) -> PoCNativeState:
    """Wrap each decoder layer (Householder) AND the token embedding (PoC-embed
    injection) BEFORE compilation, sharing one mask. Returns the state to drive
    them. Idempotent: skipped if already wrapped."""
    if any(isinstance(layer, PoCLayerWrapper) for layer in layers):
        return getattr(model, "_poc_native_state")
    state = PoCNativeState(len(layers), hidden_size, max_tokens, device, dtype)
    for i, layer in enumerate(layers):
        layers[i] = PoCLayerWrapper(layer, state.vectors[i], state.mask)
    if embed_owner is not None and hasattr(embed_owner, "embed_tokens"):
        embed_owner.embed_tokens = PoCEmbeddingWrapper(
            embed_owner.embed_tokens, state.embeds, state.mask,
            state.embed_base, state.embed_prev_k, state.embed_step, hidden_size)
    # SNAP = SAMPLING: wrap the final norm so PoC's snap rides the captured forward.
    if embed_owner is not None and hasattr(embed_owner, "norm"):
        embed_owner.norm = PoCSnapWrapper(embed_owner.norm, state)
    # Seeded-routing is MANDATORY for MoE — part of the PoC algorithm, not a toggle.
    # Natural MoE top-k reads the noise-prone hidden, so cross-HW/backend drift flips
    # the k-th expert and inflates the honest floor; seeding the experts from
    # (block_hash,nonce,step,layer) removes that. There is NO non-seeded path. Wrap
    # every MoE gate, discovered generically (any submodule with .gate + a FusedMoE
    # .experts) -> no per-model code. Chat rows are masked out (natural router kept).
    for wrapper in layers:
        inner_layer = getattr(wrapper, "inner", wrapper)
        moe = next(
            (m for m in inner_layer.modules()
             if hasattr(m, "gate") and hasattr(m, "experts")
             and hasattr(getattr(m, "experts"), "top_k")
             and not isinstance(m.gate, PoCRouterWrapper)),
            None)
        if moe is None:
            continue
        n_exp = int(moe.experts.global_num_experts)
        top_k = int(moe.experts.top_k)
        # Address-stable per-layer seed base [max_tokens] -> the wrapper folds the
        # shared route_step buffer into it and runs the Fisher-Yates selection
        # in-graph (see PoCRouterWrapper). cudagraph-safe (static shape).
        route_base = torch.zeros(state.max_tokens, dtype=torch.int64, device=device)
        state._route_base.append(route_base)
        state.router_meta.append((n_exp, top_k))
        moe.gate = PoCRouterWrapper(moe.gate, route_base, state.route_step,
                                    n_exp, top_k, state.mask)

    model._poc_native_state = state
    return state

"""Phase 2: step-driven mixed decode-PoC support.

A decode-PoC request runs ONE decode token per scheduler step, mixed with chat
in the same forward (instead of running its whole decode loop inside one
pure-batch ``execute_poc_forward`` call). Its KV lives in a stable, contiguous
slice ("slot") of the reserved block range ``[0, poc_reserved_blocks)`` so its
prefill KV persists and each decode step reads it. ``sphere_k`` is chained across
steps via the per-request state held here.

Each PoC request carries a single nonce (routes fan out per-nonce via
``generate(poc_params)``), so the decode state is per-(request, nonce).

The slot/layout helpers at the top are pure (no torch) and unit-tested. The
model-runner helpers at the bottom (moved out of gpu_model_runner.py to keep the
core vLLM footprint minimal) take the GPUModelRunner as ``runner``. Decode-PoC is
always step-driven and mixed with chat (one PoC decode token per scheduler step,
fused into the chat forward — chat is never frozen); validation runs pure.
"""
import math
from dataclasses import dataclass, field

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# Bound on consecutive chat-prefill defers before a decoding PoC is forced an
# exclusive step (fairness valve — keeps PoC from starving under chat churn).
POC_DEFER_LIMIT = 4

# Pure-PoC cudagraph capture buckets: a PoC batch is padded up to the nearest of
# these (capped at poc_max_batch_size), one cudagraph per bucket. Few buckets keep
# captured-graph memory bounded. Tune from observed batch sizes (POC_BATCH logs).
POC_GRAPH_BUCKETS = (8, 16, 32)


def poc_graph_bucket(n: int, buckets=POC_GRAPH_BUCKETS, max_batch: int = 32) -> int:
    """Smallest capture bucket >= n (capped at max_batch). Pure — unit-tested."""
    return min(next((b for b in buckets if b >= n), max_batch), max_batch)


def decode_only_mixing_gate(
    *,
    mixed_cudagraph: bool,
    poc_decode_pending: bool,
    poc_will_prefill: bool,
    chat_will_prefill: bool,
    consecutive_defers: int,
    defer_limit: int = POC_DEFER_LIMIT,
) -> tuple[bool, bool, int]:
    """Decide (defer_chat, defer_poc, consecutive_defers) so chat and PoC share a
    forward only when both decode; prefills run isolated. Mutually exclusive defers.
    Pure (unit-testable). With mixed_cudagraph=False reduces to the original
    behaviour (defer_chat=poc_decode_pending, defer_poc=False). The valve bounds
    consecutive chat-prefill defers so chat churn can't starve a decoding PoC.
    """
    defer_chat = poc_decode_pending or (mixed_cudagraph and poc_will_prefill)
    defer_poc = mixed_cudagraph and (not defer_chat) and chat_will_prefill
    if defer_poc:
        consecutive_defers += 1
        if consecutive_defers > defer_limit:
            # Give the decoding PoC one exclusive (pure-decode, graphable) step.
            defer_poc, defer_chat, consecutive_defers = False, True, 0
    else:
        consecutive_defers = 0
    return defer_chat, defer_poc, consecutive_defers


def poc_per_slot_blocks(poc_seq_len: int, poc_max_tokens: int,
                        block_size: int) -> int:
    """Contiguous reserved blocks per decode-PoC slot.

    Matches ``reservation.poc_blocks_needed`` per-sequence sizing:
    ``ceil((seq_len + max_tokens)/block) + 1`` (the ``+1`` is decode slack).
    With ``poc_max_batch_size`` slots this exactly tiles the reserved range
    ``[0, poc_reserved_blocks)``.
    """
    return math.ceil((poc_seq_len + poc_max_tokens) / block_size) + 1


def poc_slot_block_ids(slot: int, poc_seq_len: int, poc_max_tokens: int,
                       block_size: int) -> list[int]:
    """Physical block IDs for ``slot`` — a contiguous slice of the reserved
    range. The sequence is laid out contiguously, so the token at position ``p``
    uses physical block ``base + p // block_size`` (natural layout; no separate
    decode region needed since each request owns its own contiguous slot).
    """
    per_slot = poc_per_slot_blocks(poc_seq_len, poc_max_tokens, block_size)
    base = slot * per_slot
    return list(range(base, base + per_slot))


def poc_paged_layout(block_ids, seq_len, block_size):
    """Paged slot_mapping + block_table for PoC KV from per-request block ids.

    block_ids: list (one per request) of physical block ids; each list must hold
    >= ceil(seq_len/block_size) ids. Blocks may be NON-contiguous (manager-paged)
    or contiguous (static slot) — token p of a request maps to
    ``block_ids[req][p // block_size] * block_size + p % block_size``.

    Returns (slot_mapping, block_table): slot_mapping is a flat list of length
    len(block_ids)*seq_len; block_table is the per-request block-id lists. Pure
    (no torch / no GPU) so the layout is unit-testable in isolation.
    """
    slot_mapping = []
    for seq_blocks in block_ids:
        for p in range(seq_len):
            slot_mapping.append(seq_blocks[p // block_size] * block_size + p % block_size)
    return slot_mapping, block_ids


@dataclass
class PoCDecodeState:
    """Per-request decode state, carried across scheduler steps."""
    nonce: int
    slot: int
    seq_len: int
    max_tokens: int
    # number of decode steps completed so far (0 == only prefill done).
    step: int = 0
    # previous step's sphere_k; seeds the next step's input embedding + the
    # per-step random dimension selection. -1 before the prefill sphere_k is set.
    prev_k: int = -1
    # full sphere_k trajectory: prefill k, then one per decode step.
    k_points_steps: list[int] = field(default_factory=list)
    # the prefill artifact vector (base64), set at the prefill step.
    vector_b64: str = ""
    n_sphere_mismatches: int = 0


class PoCMixedDecodeManager:
    """Reserved-slot pool + decode state for step-driven mixed decode-PoC.

    One instance per model runner (lazily created). Slots are a finite pool of
    ``poc_max_batch_size`` contiguous reserved-block slices; the scheduler caps
    concurrent decode-PoC requests to that many, so ``allocate`` never starves in
    a correct configuration (returns ``None`` defensively if it would).
    """

    def __init__(self, poc_max_batch_size: int):
        self._free_slots: list[int] = list(range(poc_max_batch_size))
        self._state: dict[str, PoCDecodeState] = {}

    def get(self, req_id: str) -> PoCDecodeState | None:
        return self._state.get(req_id)

    def allocate(self, req_id: str, nonce: int, seq_len: int,
                 max_tokens: int) -> PoCDecodeState | None:
        existing = self._state.get(req_id)
        if existing is not None:
            return existing
        if not self._free_slots:
            return None
        slot = self._free_slots.pop(0)
        st = PoCDecodeState(
            nonce=nonce, slot=slot, seq_len=seq_len, max_tokens=max_tokens
        )
        self._state[req_id] = st
        return st

    def free(self, req_id: str) -> None:
        st = self._state.pop(req_id, None)
        if st is not None:
            self._free_slots.append(st.slot)

    def active_req_ids(self) -> list[str]:
        return list(self._state.keys())


def get_decode_manager(runner) -> "PoCMixedDecodeManager":
    """Lazily get/create the per-runner mixed-decode manager."""
    mgr = getattr(runner, "_poc_mixed_decode_mgr", None)
    if mgr is None:
        mgr = PoCMixedDecodeManager(runner.cache_config.poc_max_batch_size)
        runner._poc_mixed_decode_mgr = mgr
    return mgr


def setup_decode_poc(runner, poc_requests) -> bool:
    """Entry hook (called from gpu_model_runner before _prepare_inputs).

    For each decode-PoC request (max_tokens>0): grab a slot + refresh its
    per-request decode step counter. Under static KV it also points the InputBatch
    block-table row at the reserved blocks; under dynamic KV the manager-allocated
    row already drives decode (see below).

    Returns True if any decode-PoC is active this step, signalling the caller to
    route the batch through the unified step-driven path (NOT the monolithic
    execute_poc_forward). Returns False when there are no decode-PoC requests.
    """
    # Exclude VALIDATION recompute requests: the step-driven mixed path does not
    # consume inference_k_points_steps, so they must fall through to the pure
    # execute_poc_forward (which computes the real aligned n_sphere_mismatches).
    # The scheduler already routes them as a pure batch; this guards the
    # model-runner side so they are never slotted for mixed decode.
    decode_reqs = [r for r in poc_requests
                   if r.poc_params.max_tokens > 0 and not r.poc_params.is_validation]
    if not decode_reqs:
        return False
    mgr = get_decode_manager(runner)
    block_size = runner.cache_config.block_size
    num_groups = len(runner.input_batch.block_table.block_tables)
    # Dynamic KV: the scheduler allocated real (paged) blocks via the manager, so
    # _prepare_inputs already built the correct block-table row + slot_mapping — we
    # must NOT override it with a reserved-slot row. Static: point the row at the
    # reserved contiguous slot blocks (out-of-band KV).
    dynamic = getattr(runner.cache_config, "poc_dynamic_kv", False)
    for r in decode_reqs:
        pp = r.poc_params
        st = mgr.allocate(r.req_id, pp.nonce, pp.seq_len, pp.max_tokens)
        if st is None:
            # Pool exhausted (scheduler caps to poc_max_batch_size; defensive).
            logger.warning("PoC mixed-decode slot pool exhausted for %s",
                           r.req_id)
            continue
        # decode step = tokens computed beyond prefill (0 during the prefill step)
        st.step = max(0, r.num_computed_tokens - pp.seq_len)
        if dynamic:
            continue
        block_ids = poc_slot_block_ids(
            st.slot, pp.seq_len, pp.max_tokens, block_size)
        req_index = runner.input_batch.req_id_to_index.get(r.req_id)
        if req_index is not None:
            runner.input_batch.block_table.add_row(
                tuple([block_ids] * num_groups), req_index)
    return True


def apply_poc_kv_skip(attn_metadata, poc_metadata, num_total_tokens, device):
    """Set slot_mapping = PAD (-1) at PREFILL-ONLY PoC positions to skip KV
    writes. Decode-PoC positions keep their reserved-block slots — they must
    write/read real KV — so they are left untouched."""
    if not attn_metadata or not poc_metadata:
        return
    pad_mask = torch.zeros(num_total_tokens, dtype=torch.bool, device=device)
    has_pad = False
    for meta in poc_metadata:
        if meta.get('decode_state') is None:
            s = meta['start_idx']
            pad_mask[s:s + meta['length']] = True
            has_pad = True
    if not has_pad:
        return
    for layer_metadata in attn_metadata.values():
        if hasattr(layer_metadata, 'slot_mapping'):
            layer_metadata.slot_mapping[pad_mask] = -1


# ---------------------------------------------------------------------------
# Mixed-batch model-runner helpers (moved out of gpu_model_runner.py to keep the
# core vLLM footprint minimal). Each takes the GPUModelRunner as `runner`.
# ---------------------------------------------------------------------------

def build_unified_mixed_batch_inputs(
    runner,
    scheduler_output: "SchedulerOutput",
    chat_input_ids: torch.Tensor | None,
    chat_inputs_embeds: torch.Tensor | None,
    chat_positions: torch.Tensor,
    poc_req_ids: set,
    num_total_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
    """Build unified inputs for mixed batch (chat + PoC in same forward).

    CRITICAL: Preserves scheduler's token order to match slot_mapping.
    Tokens are built in the exact order of runner.input_batch.req_ids.

    Args:
        scheduler_output: The scheduler output with token counts
        chat_input_ids: Chat token IDs [num_total_tokens] or None
        chat_inputs_embeds: Chat embeddings [num_total_tokens, hidden] or None
        chat_positions: Chat positions [num_total_tokens]
        poc_req_ids: Set of PoC request IDs
        num_total_tokens: Total scheduled tokens (chat + PoC)

    Returns:
        Tuple of:
        - unified_embeds: [num_total_tokens, hidden_size]
        - unified_positions: [num_total_tokens]
        - poc_position_mask: [num_total_tokens] bool tensor (True = PoC)
        - poc_metadata: List of dicts with PoC request info
    """
    from vllm.poc.gpu_random import generate_inputs

    hidden_size = runner.model_config.get_hidden_size()
    num_reqs = runner.input_batch.num_reqs
    req_ids = runner.input_batch.req_ids

    tokens_per_req = [scheduler_output.num_scheduled_tokens[req_id]
                      for req_id in req_ids]

    unified_embeds = torch.empty(
        (num_total_tokens, hidden_size),
        dtype=runner.dtype,
        device=runner.device,
    )
    unified_positions = torch.empty(
        num_total_tokens,
        dtype=chat_positions.dtype,
        device=runner.device,
    )
    poc_position_mask = torch.zeros(
        num_total_tokens,
        dtype=torch.bool,
        device=runner.device,
    )
    poc_metadata = []

    offset = 0

    for req_idx in range(num_reqs):
        req_id = req_ids[req_idx]
        num_tokens = tokens_per_req[req_idx]

        if num_tokens <= 0:
            continue

        if req_id in poc_req_ids:
            req_state = runner.requests[req_id]
            poc_params = req_state.poc_params
            seq_len = poc_params.seq_len
            mgr = getattr(runner, "_poc_mixed_decode_mgr", None)
            st = mgr.get(req_id) if mgr is not None else None

            if st is not None and req_state.num_computed_tokens >= seq_len:
                # Phase 2 decode step: one token, embed chained from prev sphere_k.
                from vllm.poc.gpu_random import generate_decode_inputs
                decode_step = req_state.num_computed_tokens - seq_len + 1
                poc_embeds = generate_decode_inputs(
                    poc_params.block_hash, poc_params.public_key,
                    [poc_params.nonce], [st.prev_k], step=decode_step,
                    dim=hidden_size, device=runner.device, dtype=runner.dtype,
                ).view(1, hidden_size)
                unified_embeds[offset:offset + 1] = poc_embeds
                unified_positions[offset] = req_state.num_computed_tokens
                poc_position_mask[offset] = True
                poc_metadata.append({
                    'type': 'poc', 'req_id': req_id, 'start_idx': offset,
                    'length': 1, 'poc_params': poc_params,
                    'decode_state': st, 'decode_step': decode_step,
                })
                offset += 1
            else:
                # Prefill (prefill-only PoC, or the prefill step of a decode-PoC).
                poc_len = num_tokens
                poc_embeds = generate_inputs(
                    poc_params.block_hash,
                    poc_params.public_key,
                    [poc_params.nonce],
                    dim=hidden_size,
                    seq_len=poc_len,
                    device=runner.device,
                    dtype=runner.dtype,
                ).squeeze(0)  # [poc_len, hidden]
                unified_embeds[offset:offset + poc_len] = poc_embeds
                unified_positions[offset:offset + poc_len] = torch.arange(
                    poc_len, device=runner.device, dtype=chat_positions.dtype
                )
                poc_position_mask[offset:offset + poc_len] = True
                poc_metadata.append({
                    'type': 'poc', 'req_id': req_id, 'start_idx': offset,
                    'length': poc_len, 'poc_params': poc_params,
                    'decode_state': st,
                })
                offset += poc_len

        else:
            if chat_inputs_embeds is not None:
                unified_embeds[offset:offset + num_tokens] = (
                    chat_inputs_embeds[offset:offset + num_tokens]
                )
            elif chat_input_ids is not None:
                token_ids = chat_input_ids[offset:offset + num_tokens]
                # v15 renamed get_input_embeddings -> embed_input_ids (and
                # the model is the cudagraph wrapper, which forwards it).
                chat_embeds = runner.model.embed_input_ids(input_ids=token_ids)
                unified_embeds[offset:offset + num_tokens] = chat_embeds

            unified_positions[offset:offset + num_tokens] = (
                chat_positions[offset:offset + num_tokens]
            )
            offset += num_tokens

    return unified_embeds, unified_positions, poc_position_mask, poc_metadata


def process_poc_outputs_from_hidden(
    runner,
    hidden_states: torch.Tensor,
    poc_metadata: list[dict],
) -> dict[str, "PoCOutput"]:
    from vllm.v1.outputs import PoCOutput
    from vllm.poc.gpu_random import random_pick_indices, apply_haar_rotation
    from vllm.poc.data import encode_vector
    from vllm.poc.poc_model_runner import (
        SPHERE_DIM, _SPHERE_CODEBOOK, project_to_sphere, nearest_sphere_index,
    )

    poc_outputs = {}

    for meta in poc_metadata:
        end = meta['start_idx'] + meta['length']
        poc_params = meta['poc_params']
        nonce = poc_params.nonce
        st = meta.get('decode_state')

        last_hidden = hidden_states[end - 1].float()
        last_hidden = last_hidden / (last_hidden.norm() + 1e-8)
        hidden_size = last_hidden.shape[-1]

        def _vector_b64():
            idx = random_pick_indices(
                poc_params.block_hash, poc_params.public_key, [nonce],
                hidden_size, poc_params.k_dim, runner.device)
            xk = last_hidden[idx[0]]
            yk = apply_haar_rotation(
                poc_params.block_hash, poc_params.public_key, [nonce],
                xk.unsqueeze(0), runner.device)[0]
            yk = yk / (yk.norm() + 1e-8)
            return encode_vector(yk.half().cpu().numpy())

        def _sphere_k(prev_point_ids, step):
            codebook = _SPHERE_CODEBOOK.to(
                device=runner.device, dtype=last_hidden.dtype)
            sph = random_pick_indices(
                poc_params.block_hash, poc_params.public_key, [nonce],
                hidden_size, SPHERE_DIM, runner.device,
                prev_point_ids=prev_point_ids, step=step)
            xk_sphere = project_to_sphere(
                torch.gather(last_hidden.unsqueeze(0), 1, sph))
            return int(nearest_sphere_index(xk_sphere, codebook)[0])

        if st is None:
            # Prefill-only PoC: just the vector_b64 artifact.
            poc_outputs[meta['req_id']] = PoCOutput(
                nonce=nonce, vector_b64=_vector_b64())
            continue

        if 'decode_step' not in meta:
            # Prefill step of a decode-PoC: vector_b64 + seed prefill sphere_k.
            st.vector_b64 = _vector_b64()
            k0 = _sphere_k(None, 0)
            st.prev_k = k0
            st.k_points_steps = [k0]
        else:
            # One decode step: chain sphere_k from prev_k.
            step = meta['decode_step']
            k = _sphere_k([st.prev_k], step)
            st.k_points_steps.append(k)
            st.prev_k = k
            if step >= st.max_tokens:
                poc_outputs[meta['req_id']] = PoCOutput(
                    nonce=nonce,
                    vector_b64=st.vector_b64,
                    k_points_steps=st.k_points_steps,
                )
                get_decode_manager(runner).free(meta['req_id'])

    return poc_outputs


def filter_sampling_metadata_for_chat(
        sampling_metadata,
        chat_mask: torch.Tensor,
        chat_indices: list[int],
):
    """Filter sampling metadata to only include chat (non-PoC) requests.

    PoC requests don't need sampling - they produce distance vectors, not tokens.
    Filtering them avoids CUDA asserts from invalid sampling parameters.
    """
    from vllm.v1.sample.metadata import SamplingMetadata
    from vllm.v1.sample.logits_processor.state import LogitsProcessors

    num_chat = len(chat_indices)

    def filter_tensor(t, dim=0):
        if t is None:
            return None
        if t.shape[0] != chat_mask.shape[0]:
            if t.shape[0] >= max(chat_indices) + 1 if chat_indices else 0:
                return t[chat_indices] if dim == 0 else t[chat_indices, :]
            else:
                logger.warning(
                    f"filter_tensor: mask shape {chat_mask.shape[0]} != "
                    f"tensor shape {t.shape}, and cannot index by chat_indices. Skipping filter."
                )
                return t
        return t[chat_mask] if dim == 0 else t[chat_mask, :]

    new_generators = {}
    for new_idx, old_idx in enumerate(chat_indices):
        if old_idx in sampling_metadata.generators:
            new_generators[new_idx] = sampling_metadata.generators[old_idx]

    new_output_token_ids = (
        [sampling_metadata.output_token_ids[i] for i in chat_indices]
        if sampling_metadata.output_token_ids
        else []
    )

    new_bad_words = {}
    for new_idx, old_idx in enumerate(chat_indices):
        if old_idx in sampling_metadata.bad_words_token_ids:
            new_bad_words[new_idx] = sampling_metadata.bad_words_token_ids[old_idx]

    # TODO: Properly filter logits processors if needed
    new_logitsprocs = LogitsProcessors()

    # NOTE: v15 SamplingMetadata fields only. The proposal targeted an older
    # vLLM that had all_enforced/mixed_enforced/enforced_token_ids/
    # enforced_tokens/enforced_req_ids; v15 replaced those with a single
    # enforced_next_token_ids tensor (+ batch_logprobs_mode/
    # logprobs_is_processed/spec_token_ids). Build with the v15 schema.
    return SamplingMetadata(
        temperature=filter_tensor(sampling_metadata.temperature),
        all_greedy=sampling_metadata.all_greedy,
        all_random=sampling_metadata.all_random,
        top_p=filter_tensor(sampling_metadata.top_p),
        top_k=filter_tensor(sampling_metadata.top_k),
        generators=new_generators,
        max_num_logprobs=sampling_metadata.max_num_logprobs,
        no_penalties=sampling_metadata.no_penalties,
        prompt_token_ids=filter_tensor(sampling_metadata.prompt_token_ids, dim=0),
        frequency_penalties=filter_tensor(sampling_metadata.frequency_penalties),
        presence_penalties=filter_tensor(sampling_metadata.presence_penalties),
        repetition_penalties=filter_tensor(sampling_metadata.repetition_penalties),
        output_token_ids=new_output_token_ids,
        allowed_token_ids_mask=filter_tensor(sampling_metadata.allowed_token_ids_mask, dim=0)
            if sampling_metadata.allowed_token_ids_mask is not None else None,
        bad_words_token_ids=new_bad_words,
        logitsprocs=new_logitsprocs,
        batch_logprobs_mode=sampling_metadata.batch_logprobs_mode,
        logprobs_is_processed=(
            filter_tensor(sampling_metadata.logprobs_is_processed)
            if sampling_metadata.logprobs_is_processed is not None else None
        ),
        spec_token_ids=(
            [sampling_metadata.spec_token_ids[i] for i in chat_indices]
            if sampling_metadata.spec_token_ids else None
        ),
        enforced_next_token_ids=(
            filter_tensor(sampling_metadata.enforced_next_token_ids)
            if sampling_metadata.enforced_next_token_ids is not None else None
        ),
    )


def expand_sampler_output_for_poc(
        sampler_output: "SamplerOutput",
        chat_mask: torch.Tensor,
        chat_indices: list[int],
        num_total_reqs: int,
) -> "SamplerOutput":
    """Expand sampler output to include dummy tokens for PoC requests.

    PoC requests were filtered before sampling. This reconstructs the full
    output with placeholder values for PoC positions.
    """
    from vllm.v1.outputs import SamplerOutput, LogprobsTensors

    # PoC positions get token 0 (will be ignored in post-processing)
    full_sampled = torch.zeros(
        (num_total_reqs, sampler_output.sampled_token_ids.shape[-1]),
        dtype=sampler_output.sampled_token_ids.dtype,
        device=sampler_output.sampled_token_ids.device,
    )
    full_sampled[chat_mask] = sampler_output.sampled_token_ids

    full_logprobs = None
    if sampler_output.logprobs_tensors is not None:
        lp = sampler_output.logprobs_tensors
        if lp.top_token_ids is not None:
            full_top_ids = torch.zeros(
                (num_total_reqs,) + lp.top_token_ids.shape[1:],
                dtype=lp.top_token_ids.dtype,
                device=lp.top_token_ids.device,
            )
            full_top_ids[chat_mask] = lp.top_token_ids
        else:
            full_top_ids = None

        if lp.top_logprobs is not None:
            full_top_lp = torch.zeros(
                (num_total_reqs,) + lp.top_logprobs.shape[1:],
                dtype=lp.top_logprobs.dtype,
                device=lp.top_logprobs.device,
            )
            full_top_lp[chat_mask] = lp.top_logprobs
        else:
            full_top_lp = None

        if lp.sampled_logprobs is not None:
            full_sampled_lp = torch.zeros(
                (num_total_reqs,) + lp.sampled_logprobs.shape[1:],
                dtype=lp.sampled_logprobs.dtype,
                device=lp.sampled_logprobs.device,
            )
            full_sampled_lp[chat_mask] = lp.sampled_logprobs
        else:
            full_sampled_lp = None

        full_logprobs = LogprobsTensors(
            top_token_ids=full_top_ids,
            top_logprobs=full_top_lp,
            sampled_logprobs=full_sampled_lp,
        )

    return SamplerOutput(
        sampled_token_ids=full_sampled,
        logprobs_tensors=full_logprobs,
    )

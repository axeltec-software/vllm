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


def poc_is_pure_path(poc_params) -> bool:
    """A PoC request runs a pure (exclusive) batch when prefill-only or a
    validation recompute — only the pure path consumes inference_k_points_steps
    (the mixed generation path returns -1 for n_sphere_mismatches). Decode
    GENERATION runs step-driven mixed with chat. Pure (unit-testable)."""
    return not (poc_params.max_tokens > 0 and not poc_params.is_validation)


def poc_step_num_tokens(poc_params, num_computed_tokens: int) -> int:
    """Tokens to schedule for a PoC request this step: mixed decode generation
    prefills seq_len once then 1 token/step; the pure / prefill-only path is a
    single seq_len step. Pure (unit-testable)."""
    if not poc_is_pure_path(poc_params):
        return poc_params.seq_len if num_computed_tokens == 0 else 1
    return poc_params.seq_len


def poc_alloc_footprint(poc_params, num_new_tokens: int) -> int:
    """Dynamic-KV blocks to allocate: the pure path runs the whole decode loop in
    one step so it reserves seq_len+max_tokens upfront; the mixed path reserves
    one step's tokens. Pure (unit-testable)."""
    if poc_is_pure_path(poc_params):
        return poc_params.seq_len + poc_params.max_tokens
    return num_new_tokens


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


# ---------------------------------------------------------------------------
# execute_model dispatch helpers (moved out of gpu_model_runner.py). Each takes
# the GPUModelRunner as `runner`; these are the thin core hooks' bodies.
# ---------------------------------------------------------------------------

def dispatch_pure_poc(runner, poc_requests, poc_req_ids):
    """Pure-PoC batch: full prefill + KV-bound decode forward via
    execute_poc_forward. Returns the value execute_model should return (the raw
    poc_result on non-last PP ranks, else a ModelRunnerOutput carrying the
    per-nonce PoCOutputs). PoC writes KV into its allocated blocks, so this
    co-exists with live chat without collective_rpc/abort.
    """
    from vllm.poc.poc_model_runner import execute_poc_forward
    from vllm.poc.data import encode_vector
    from vllm.v1.outputs import PoCOutput, ModelRunnerOutput
    from vllm.distributed.parallel_state import get_pp_group

    first_params = poc_requests[0].poc_params
    nonces = [req.poc_params.nonce for req in poc_requests]
    inference_steps = [
        req.poc_params.inference_k_points_steps for req in poc_requests
    ]
    # Pad the PoC batch to the nearest capture bucket (<= reserved
    # poc_max_batch_size), one cudagraph per bucket. Few buckets keeps the
    # captured-graph memory bounded; bit-exactness across buckets isn't required
    # (validation tolerates boundary flips). Padding nonces are negative (never
    # collide) and dropped by the nonce->req_id map.
    poc_max_batch = runner.cache_config.poc_max_batch_size
    poc_dynamic_kv = getattr(runner.cache_config, "poc_dynamic_kv", False)
    block_ids_arg = None
    if poc_dynamic_kv:
        # Dynamic KV: eager (no per-N cudagraph) -> no bucket padding. Pass each
        # request's manager-allocated paged blocks (the scheduler allocated the
        # full seq_len+max_tokens footprint).
        block_ids_arg = [
            list(runner.requests[req.req_id].block_ids[0])
            for req in poc_requests
        ]
    else:
        n_real = len(nonces)
        bucket = poc_graph_bucket(n_real, POC_GRAPH_BUCKETS, poc_max_batch)
        logger.info("POC_BATCH real=%d bucket=%d", n_real, bucket)  # bucket-tuning stats
        pad = bucket - n_real
        if pad > 0:
            nonces = nonces + [-(i + 1) for i in range(pad)]
            inference_steps = inference_steps + [None] * pad
    _poc_kwargs = dict(
        block_hash=first_params.block_hash,
        public_key=first_params.public_key,
        nonces=nonces,
        seq_len=first_params.seq_len,
        hidden_size=runner.model_config.get_hidden_size(),
        k_dim=first_params.k_dim,
        poc_decode=first_params.poc_decode,
        max_tokens=first_params.max_tokens,
        inference_k_points_steps_per_nonce=(
            inference_steps
            if any(s is not None for s in inference_steps)
            else None
        ),
        debug=any(req.poc_params.debug for req in poc_requests),
        block_ids=block_ids_arg,
    )
    poc_result = execute_poc_forward(runner, **_poc_kwargs)
    # If chat aliased the cudagraph and every nonce came back NaN,
    # execute_poc_forward reset the graph state; retry once so the client gets a
    # valid result instead of an empty batch.
    if nonces and not poc_result.get("nonces"):
        logger.warning("PoC all-NaN; retrying after graph reset")
        poc_result = execute_poc_forward(runner, **_poc_kwargs)

    if not get_pp_group().is_last_rank:
        return poc_result

    # Map results back to request ids by nonce (robust to NaN-dropped nonces).
    # k_points_steps is the decode-PoC artifact and must be carried through;
    # debug fields are passed when present.
    nonce_to_req_id = {
        req.poc_params.nonce: req.req_id for req in poc_requests
    }
    k_points_steps_list = poc_result.get("k_points_steps_list", [])
    mismatch_count = poc_result.get("mismatch_count", [])
    sph_indices_steps = poc_result.get("sph_indices_steps", [])
    sph_values_steps = poc_result.get("sph_values_steps", [])
    vectors = poc_result["vectors"]
    result_nonces = poc_result["nonces"]

    poc_outputs_dict = {}
    for j, nonce in enumerate(result_nonces):
        req_id = nonce_to_req_id.get(nonce)
        if req_id is None:
            continue
        poc_outputs_dict[req_id] = PoCOutput(
            nonce=nonce,
            vector_b64=encode_vector(vectors[j]),
            k_points_steps=(
                k_points_steps_list[j] if k_points_steps_list else []
            ),
            n_sphere_mismatches=(mismatch_count[j] if mismatch_count else -1),
            sph_indices_steps=(
                sph_indices_steps[j] if sph_indices_steps else []
            ),
            sph_values_steps=(
                sph_values_steps[j] if sph_values_steps else []
            ),
        )

    poc_output = ModelRunnerOutput(
        req_ids=list(poc_req_ids),
        req_id_to_index={req_id: i for i, req_id in enumerate(poc_req_ids)},
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        poc_outputs=poc_outputs_dict,
    )
    # Store so sample_tokens() can return it when called by the async batch queue
    # (execute_model_state is not set for pure-PoC batches, so sample_tokens()
    # would otherwise return None and trigger RuntimeError in step_with_batch_queue).
    runner._poc_direct_output = poc_output
    return poc_output


def is_mixed_decode_graphable(runner, *, is_mixed_batch, max_num_scheduled_tokens,
                              num_tokens_unpadded, num_reqs,
                              poc_requests, chat_requests):
    """A uniform 1-token-decode mixed batch is graphable: dropping force_eager
    lets vLLM build+re-plan its stable cudagraph decode buffers each step, so the
    manual mixed graph reads fresh metadata on replay. Logs a contract violation
    when a real chat+PoC batch is NOT uniform-decode (a prefill mix leaked past
    the scheduler decode-only gate -> eager fallback).
    """
    graphable = (
        (not runner.model_config.enforce_eager)
        and is_mixed_batch
        and max_num_scheduled_tokens == runner.uniform_decode_query_len
        and num_tokens_unpadded == max_num_scheduled_tokens * num_reqs
    )
    chat_and_poc = bool(poc_requests) and bool(chat_requests)
    if (not runner.model_config.enforce_eager) and chat_and_poc and not graphable:
        logger.warning(
            "POC_CONTRACT_VIOLATION mixed-eager batch "
            "(num_reqs=%d max_query=%d unpadded=%d) — a prefill-mixed batch "
            "leaked past the scheduler decode-only gate",
            num_reqs, max_num_scheduled_tokens, num_tokens_unpadded)
    return graphable


def setup_poc_forward_hooks(runner, poc_position_mask, poc_metadata):
    """(Re)configure the Householder layer hooks for this forward and return
    ``(poc_context, compile_bypass)`` context managers for the model call.

    Graph mode: the persistent hook stays attached (baked into the captured
    graph); drive the transform via the stable mask buffer and refresh the
    reflection vectors for this round's block_hash (no context / no compile
    bypass — we WANT the graph). Eager mode: tear down on a plain-chat forward,
    (re)build per block_hash for a PoC forward, drive via the context mask and
    bypass torch.compile.
    """
    from contextlib import nullcontext
    from vllm.poc.layer_hooks import (
        poc_forward_context_with_mask, LayerHouseholderHook,
    )
    from vllm.poc.poc_model_runner import bypass_torch_compile

    if not runner.model_config.enforce_eager:
        if getattr(runner, '_poc_stable_mask_buf', None) is None:
            # Lazy: attach the (flag-gated, chat-safe) persistent hook + stable
            # mask buffer on the first mixed-cudagraph forward.
            _ensure_mixed_graph_hooks(runner)
        buf = runner._poc_stable_mask_buf
        if poc_position_mask is not None and poc_metadata:
            bh = poc_metadata[0]['poc_params'].block_hash
            hooks = getattr(runner, '_poc_layer_hooks', None)
            if hooks is not None and hooks.block_hash != bh:
                hooks.update_block_hash(
                    bh, runner.model_config.get_hidden_size(), runner.device)
                hooks.block_hash = bh
            n = poc_position_mask.shape[0]
            buf[:n].copy_(poc_position_mask)
            buf[n:].zero_()
        else:
            buf.zero_()
        return nullcontext(), nullcontext()

    # Eager mixed path (default).
    if poc_position_mask is None and getattr(runner, '_poc_layer_hooks', None) is not None:
        runner._poc_layer_hooks.detach()
        runner._poc_layer_hooks = None

    if poc_position_mask is not None and poc_metadata:
        block_hash = poc_metadata[0]['poc_params'].block_hash
        hidden_size = runner.model_config.get_hidden_size()
        cached_hooks = getattr(runner, '_poc_layer_hooks', None)
        if cached_hooks is None or cached_hooks.block_hash != block_hash:
            if cached_hooks is not None:
                cached_hooks.detach()
            cached_hooks = LayerHouseholderHook(
                runner.model, block_hash, runner.device, hidden_size)
            runner._poc_layer_hooks = cached_hooks

    poc_context = (
        poc_forward_context_with_mask(poc_position_mask)
        if poc_position_mask is not None else nullcontext()
    )
    compile_bypass = (
        bypass_torch_compile() if poc_position_mask is not None else nullcontext()
    )
    return poc_context, compile_bypass


def poc_mixed_graph_dispatch(runner, inputs_embeds, positions, attn_metadata,
                             slot_mappings, num_tokens_padded,
                             num_tokens_across_dp, cudagraph_mode,
                             poc_position_mask):
    """Fused mixed cudagraph forward. Manual skip_compiled capture so the
    flag-gated Householder where-blend bakes into the graph (no dynamo);
    attention reuses vLLM's stable decode buffers (reset+retry on NaN). Decode
    mixed batches graph here; a pure-PoC prefill step (scheduler makes prefill
    exclusive) is graphed per-N by poc_graphed_prefill_batch. Returns the hidden
    states (model_output).
    """
    from vllm.config import CUDAGraphMode
    graphable = cudagraph_mode != CUDAGraphMode.NONE
    poc_meta = (runner._mixed_batch_info or {}).get("poc_metadata") or []
    prefill_metas = [m for m in poc_meta if "decode_step" not in m]
    # Dynamic KV: prefill uses manager-allocated (paged) blocks via the standard
    # metadata, so run it eager (the reserved-slot per-N prefill graph does not
    # apply). Decode still rides the mixed decode graph.
    if (not graphable) and poc_meta \
            and not getattr(runner.cache_config, "poc_dynamic_kv", False) \
            and len(prefill_metas) == len(poc_meta) \
            and all(m.get("decode_state") is not None for m in prefill_metas) \
            and bool(poc_position_mask.all()):
        from vllm.poc.poc_model_runner import poc_graphed_prefill_batch
        prefill_metas.sort(key=lambda m: m["start_idx"])
        pp0 = prefill_metas[0]["poc_params"]
        slots = [m["decode_state"].slot for m in prefill_metas]
        return poc_graphed_prefill_batch(
            runner, pp0.seq_len, pp0.max_tokens,
            inputs_embeds[:len(slots) * pp0.seq_len], slots)
    return _mixed_graph_forward(
        runner, inputs_embeds, positions, attn_metadata, slot_mappings,
        num_tokens_padded, num_tokens_across_dp, graphable=graphable)


def _ensure_mixed_graph_hooks(runner) -> None:
    # Attach a persistent Householder hook + a stable all-False position-mask
    # buffer BEFORE capture, so warmup forwards record the where-blend into the
    # graph (replay can't run Python hooks).
    from vllm.poc.layer_hooks import set_stable_poc_mask, LayerHouseholderHook
    buf = torch.zeros(runner.max_num_tokens, dtype=torch.bool, device=runner.device)
    runner._poc_stable_mask_buf = buf
    set_stable_poc_mask(buf)
    if getattr(runner, '_poc_layer_hooks', None) is None:
        runner._poc_layer_hooks = LayerHouseholderHook(
            runner.model, "poc_graph_capture_default",
            runner.device, runner.model_config.get_hidden_size())


def _mixed_graph_forward(runner, inputs_embeds, positions, attn_metadata,
                         slot_mappings, num_tokens, num_tokens_across_dp,
                         graphable=True):
    """Manual capture/replay of the mixed forward (skip_compiled bakes the mask
    hook's where-blend). graphable=False (PoC prefill) runs eager, never captured.
    See wiki: mixed-prefill-cudagraph-illegal-access-only-decode-graphable."""
    import os
    from collections import defaultdict
    from vllm.forward_context import set_forward_context
    from vllm.poc.layer_hooks import set_manual_capture_active

    cache = getattr(runner, "_poc_mixed_cuda_graphs", {})
    key = int(num_tokens)
    model = runner.model

    def _ctx():
        return set_forward_context(
            attn_metadata, runner.vllm_config,
            num_tokens=num_tokens, num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings, skip_compiled=True,
        )

    def _fwd():
        return model(input_ids=None, positions=positions,
                     intermediate_tensors=None, inputs_embeds=inputs_embeds)

    if not graphable:
        # Non-graphable (PoC prefill): eager, NEVER captured — see docstring.
        set_manual_capture_active(True)
        try:
            with _ctx():
                out = _fwd()
        finally:
            set_manual_capture_active(False)
        return out[0] if isinstance(out, tuple) else out

    if key not in cache:
        # Opt-in graph audit (VLLM_POC_GRAPH_DUMP=<dir>).
        _dump_dir = os.environ.get("VLLM_POC_GRAPH_DUMP")
        set_manual_capture_active(True)
        try:
            if _dump_dir:
                # Profile the warmup forward (the same _fwd() that gets captured,
                # so its kernel count == the graph content). Profiling an extra
                # replay would double-write KV.
                from torch.profiler import profile as _prof, ProfilerActivity as _PA
                with _prof(activities=[_PA.CUDA]) as _pr:
                    with _ctx():
                        warm = _fwd()
                    torch.cuda.synchronize()
                _hist = defaultdict(int)
                for _e in _pr.events():
                    if str(getattr(_e, "device_type", "")).endswith("CUDA"):
                        _hist[_e.name] += 1
                _k = sum(_hist.values())
                # Kernel histogram (count<TAB>name) for benchmarks/poc/graph_report.py.
                try:
                    os.makedirs(_dump_dir, exist_ok=True)
                    with open(os.path.join(_dump_dir,
                              f"mixed_{int(num_tokens)}.kernels.txt"), "w") as _f:
                        for _name, _c in sorted(_hist.items(), key=lambda kv: -kv[1]):
                            _f.write(f"{_c}\t{_name}\n")
                except Exception as _ex:
                    logger.warning("POC_GRAPH_AUDIT write failed: %s", _ex)
                logger.info("POC_GRAPH_AUDIT mixed_%d: %d kernels, %d distinct "
                            "(== captured graph content)", int(num_tokens), _k, len(_hist))
            else:
                with _ctx():
                    warm = _fwd()
                torch.cuda.synchronize()
            warm_hidden = (warm[0] if isinstance(warm, tuple) else warm).clone()
            if not hasattr(runner, "_poc_mixed_graph_pool"):
                runner._poc_mixed_graph_pool = torch.cuda.graph_pool_handle()
            g = torch.cuda.CUDAGraph()
            if _dump_dir:
                try:
                    g.enable_debug_mode()  # required for debug_dump (driver permitting)
                except Exception:
                    pass
            with _ctx():
                with torch.cuda.graph(g, pool=runner._poc_mixed_graph_pool):
                    out = _fwd()
        finally:
            set_manual_capture_active(False)
        captured = out[0] if isinstance(out, tuple) else out
        cache[key] = (g, captured)
        runner._poc_mixed_cuda_graphs = cache
        logger.info("POC_GRAPH mixed captured: num_tokens=%d", int(num_tokens))
        if _dump_dir:
            # Graphviz DAG (cudaGraphDebugDotPrint); no-op on some drivers.
            try:
                os.makedirs(_dump_dir, exist_ok=True)
                path = os.path.join(_dump_dir, f"mixed_{int(num_tokens)}.dot")
                g.debug_dump(path)
                if os.path.exists(path):
                    logger.info("POC_GRAPH_DUMP wrote %s", path)
            except Exception as e:
                logger.warning("POC_GRAPH_DUMP failed: %s", e)
        return warm_hidden  # first call: eager warmup result
    g, captured = cache[key]
    with _ctx():
        g.replay()
    runner._poc_mixed_replays = getattr(runner, "_poc_mixed_replays", 0) + 1
    if runner._poc_mixed_replays % 64 == 0:
        logger.info("POC_GRAPH mixed replays=%d", runner._poc_mixed_replays)
    if torch.isnan(captured).any():
        # Chat aliased the graph → drop it so the next step recaptures fresh.
        for _a in ("_poc_mixed_cuda_graphs", "_poc_mixed_graph_pool"):
            if hasattr(runner, _a):
                delattr(runner, _a)
    return captured

"""PoC model runner for vLLM 0.15.x V1 architecture.

Full model forward pass with proper V1 attention metadata.
Uses actual KV cache blocks for attention to work correctly.
Batched forward pass — processes all nonces in a single forward call.
"""
import math
import torch
import torch.distributed as dist
import numpy as np
from typing import List, Optional, Dict, Any

from vllm.distributed import get_pp_group, get_tp_group
from vllm.distributed.communication_op import broadcast_tensor_dict
from vllm.forward_context import set_forward_context
from vllm.sequence import IntermediateTensors
from vllm.logger import init_logger

from .gpu_random import (
    generate_inputs,
    generate_inputs_concat_murmur,
    generate_decode_inputs,
    random_pick_indices,
    apply_haar_rotation,
)
from .data import encode_vector
from .layer_hooks import LayerHouseholderHook, poc_forward_context

logger = init_logger(__name__)

DEFAULT_K_DIM = 12

# SPHERE_DIM: dimension of the hidden-state slice projected onto the sphere.
# SPHERE_POINTS: number of equidistant codebook points on that sphere.
SPHERE_DIM = 256
SPHERE_POINTS = 16


def project_to_sphere(v: torch.Tensor) -> torch.Tensor:
    """Normalize [..., dim] vectors to the unit sphere (L2 norm = 1)."""
    return v / (v.norm(dim=-1, keepdim=True) + 1e-8)


def _halton_on_sphere(n_points: int, dim: int) -> torch.Tensor:
    """Return n_points deterministic, low-discrepancy unit vectors on S^(dim-1).

    Uses the Halton sequence (base-prime per dimension) mapped to the sphere
    via the logit transform. Identical output for any call with the same args.
    """
    _PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    coords: list[list[float]] = []
    for d in range(dim):
        base = _PRIMES[d % len(_PRIMES)]
        col: list[float] = []
        for i in range(1, n_points + 1):
            f, r = 1.0, 0.0
            j = i
            while j > 0:
                f /= base
                r += f * (j % base)
                j //= base
            col.append(r)
        coords.append(col)
    raw = torch.tensor(coords, dtype=torch.float32).T.clamp(0.01, 0.99)
    pts = torch.log(raw / (1.0 - raw))
    return project_to_sphere(pts)


def build_equidistant_codebook(
    n_points: int,
    dim: int,
    n_steps: int = 500,
    lr: float = 0.05,
) -> torch.Tensor:
    """Build approximately equidistant points on S^(dim-1) via Thomson problem.

    Minimizes electrostatic repulsion energy so points spread uniformly.
    Deterministic initialisation (Halton sequence). Result is cached in
    _SPHERE_CODEBOOK at module load time.
    """
    with torch.inference_mode(mode=False):
        pts = _halton_on_sphere(n_points, dim).clone().requires_grad_(True)
        opt = torch.optim.Adam([pts], lr=lr)
        eye = torch.eye(n_points)
        for _ in range(n_steps):
            opt.zero_grad()
            p = project_to_sphere(pts)
            diff = p.unsqueeze(0) - p.unsqueeze(1)
            d2 = (diff * diff).sum(-1)
            energy = ((1.0 - eye) / (d2 + 1e-8)).sum()
            energy.backward()
            opt.step()
        result = project_to_sphere(pts).detach()
    return result


_SPHERE_CODEBOOK: torch.Tensor = build_equidistant_codebook(SPHERE_POINTS, SPHERE_DIM)


def nearest_sphere_index(query: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Return the index of the nearest codebook point for each query vector.

    Args:
        query: unit vectors [batch, dim]
        codebook: unit vectors [SPHERE_POINTS, dim]
    Returns:
        index tensor [batch] with values in [0, SPHERE_POINTS)
    """
    sims = query.float() @ codebook.float().T
    return sims.argmax(dim=-1)

# NOTE: attention metadata must NOT be cached across PoC calls.
# The metadata builder's internal state (workspace buffers, page-table
# references) is mutated by every inference engine step.  Reusing a
# stale metadata object causes the attention backend to write only a
# fraction of the expected KV entries, producing all-NaN hidden states.
# The cost of rebuilding is <1 ms per call (vs ~15 ms for the model
# forward), so the overhead is negligible.


def _ensure_layer_hooks(worker, block_hash, hidden_size):
    """Ensure layer hooks are installed for the given block_hash."""
    model = worker.model_runner.model
    device = worker.device
    existing_hook = getattr(worker, "_poc_layer_hooks", None)
    if existing_hook is not None:
        if existing_hook.block_hash == block_hash:
            return
        existing_hook.detach()
    hook = LayerHouseholderHook(model, block_hash, device, hidden_size)
    hook._setup(model, block_hash, device, hidden_size)
    worker._poc_layer_hooks = hook


def _get_block_size(worker):
    """Get the KV cache block size from the worker config."""
    return worker.cache_config.block_size


def _create_v1_attn_metadata(batch_size, seq_len, block_size, device, worker):
    """Create attention metadata for batch_size sequences.

    Uses the worker's metadata builders to create the correct metadata
    for whatever attention backend is configured (FlashAttention,
    FlashInfer, etc.).
    """
    from vllm.v1.attention.backend import CommonAttentionMetadata

    blocks_per_seq = math.ceil(seq_len / block_size)
    total_tokens = batch_size * seq_len

    # slot_mapping: each sequence gets its own block range
    all_slots = []
    for seq_idx in range(batch_size):
        base_block = seq_idx * blocks_per_seq
        for t in range(seq_len):
            block_idx = base_block + t // block_size
            all_slots.append(block_idx * block_size + t % block_size)
    slot_mapping = torch.tensor(all_slots, dtype=torch.long, device=device)

    # block_table: [batch_size, blocks_per_seq]
    block_table = torch.arange(
        batch_size * blocks_per_seq, dtype=torch.int32, device=device
    ).view(batch_size, blocks_per_seq)

    # query_start_loc: [0, seq_len, 2*seq_len, ..., batch_size*seq_len]
    query_start_loc_gpu = (
        torch.arange(batch_size + 1, dtype=torch.int32, device=device) * seq_len
    )
    query_start_loc_cpu = (
        torch.arange(batch_size + 1, dtype=torch.int32, device="cpu") * seq_len
    )

    seq_lens_gpu = torch.full(
        (batch_size,), seq_len, dtype=torch.int32, device=device
    )
    seq_lens_cpu = torch.full(
        (batch_size,), seq_len, dtype=torch.int32, device="cpu"
    )

    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc_gpu,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens_gpu,
        num_reqs=batch_size,
        num_actual_tokens=total_tokens,
        max_query_len=seq_len,
        max_seq_len=seq_len,
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
        causal=True,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=torch.zeros(
            batch_size, dtype=torch.int32, device="cpu"
        ),
    )

    model_runner = worker.model_runner
    attn_metadata_dict = {}
    slot_mapping_dict = {}

    for kv_cache_group_attn_groups in model_runner.attn_groups:
        for attn_group in kv_cache_group_attn_groups:
            builder = attn_group.get_metadata_builder(0)
            metadata = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
            for layer_name in attn_group.layer_names:
                attn_metadata_dict[layer_name] = metadata
                slot_mapping_dict[layer_name] = slot_mapping

    return attn_metadata_dict, slot_mapping_dict


def _create_decode_attn_metadata_with_history(
    batch_size,
    prefill_seq_len,
    step,
    block_size,
    device,
    worker,
    prefill_blocks_per_seq,
    max_decode_blocks_per_seq,
    decode_block_start,
):
    """Create attention metadata for a single decode step with full context history.

    One new token per sequence (the query) attends to all prefill_seq_len + step
    tokens in the KV cache.  Physical block layout must be consistent with what
    _create_v1_attn_metadata wrote during the prefill phase:
      - seq i prefill blocks: i*prefill_blocks_per_seq .. (i+1)*prefill_blocks_per_seq - 1
      - seq i decode blocks:  decode_block_start + i*max_decode_blocks_per_seq + j
    """
    from vllm.v1.attention.backend import CommonAttentionMetadata

    new_pos = prefill_seq_len + step - 1
    block_in_seq = new_pos // block_size
    slot_in_block = new_pos % block_size
    context_len = prefill_seq_len + step
    total_blocks_for_context = math.ceil(context_len / block_size)

    slot_mapping_list = []
    block_table_rows = []

    for seq_idx in range(batch_size):
        if block_in_seq < prefill_blocks_per_seq:
            phys_block = seq_idx * prefill_blocks_per_seq + block_in_seq
        else:
            decode_blk_idx = block_in_seq - prefill_blocks_per_seq
            phys_block = (
                decode_block_start
                + seq_idx * max_decode_blocks_per_seq
                + decode_blk_idx
            )
        slot_mapping_list.append(phys_block * block_size + slot_in_block)

        row = []
        for blk_in_seq in range(total_blocks_for_context):
            if blk_in_seq < prefill_blocks_per_seq:
                row.append(seq_idx * prefill_blocks_per_seq + blk_in_seq)
            else:
                decode_blk_idx = blk_in_seq - prefill_blocks_per_seq
                row.append(
                    decode_block_start
                    + seq_idx * max_decode_blocks_per_seq
                    + decode_blk_idx
                )
        block_table_rows.append(row)

    slot_mapping = torch.tensor(slot_mapping_list, dtype=torch.long, device=device)
    block_table = torch.tensor(block_table_rows, dtype=torch.int32, device=device)

    query_start_loc_gpu = torch.arange(
        batch_size + 1, dtype=torch.int32, device=device
    )
    query_start_loc_cpu = torch.arange(
        batch_size + 1, dtype=torch.int32, device="cpu"
    )
    seq_lens_gpu = torch.full(
        (batch_size,), context_len, dtype=torch.int32, device=device
    )
    seq_lens_cpu = torch.full(
        (batch_size,), context_len, dtype=torch.int32, device="cpu"
    )

    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc_gpu,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens_gpu,
        num_reqs=batch_size,
        num_actual_tokens=batch_size,
        max_query_len=1,
        max_seq_len=context_len,
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
        causal=True,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=torch.full(
            (batch_size,), prefill_seq_len + step - 1,
            dtype=torch.int32, device="cpu",
        ),
    )

    model_runner = worker.model_runner
    attn_metadata_dict = {}
    slot_mapping_dict = {}

    for kv_cache_group_attn_groups in model_runner.attn_groups:
        for attn_group in kv_cache_group_attn_groups:
            builder = attn_group.get_metadata_builder(0)
            metadata = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
            for layer_name in attn_group.layer_names:
                attn_metadata_dict[layer_name] = metadata
                slot_mapping_dict[layer_name] = slot_mapping

    return attn_metadata_dict, slot_mapping_dict


def _get_or_create_attn_metadata(batch_size, seq_len, block_size, device, worker):
    """Create fresh attention metadata for the given parameters."""
    return _create_v1_attn_metadata(batch_size, seq_len, block_size, device, worker)


@torch.inference_mode()
def execute_poc_forward(
    worker,
    block_hash: str,
    public_key: str,
    nonces: List[int],
    seq_len: int,
    hidden_size: int,
    k_dim: int = DEFAULT_K_DIM,
    poc_stronger_rng: bool = False,
    poc_decode: bool = False,
    max_tokens: int = 0,
    inference_k_points_steps_per_nonce: Optional[List[Optional[List[int]]]] = None,
    debug: bool = False,
) -> Optional[Dict[str, Any]]:
    """Execute batched PoC forward pass on a V1 worker.

    Processes all nonces in a single forward call for maximum throughput.
    When poc_decode=True and max_tokens > 0, runs additional decode steps
    after prefill, chaining each step's sphere_k into the next step's seed.
    """
    device = worker.device
    dtype = worker.model_config.dtype
    model = worker.model_runner.model
    vllm_config = worker.vllm_config
    batch_size = len(nonces)

    tp_group = get_tp_group()
    is_tp_driver = tp_group.rank_in_group == 0

    # TP SYNC
    if tp_group.world_size > 1:
        dist.barrier(group=tp_group.cpu_group)
        if is_tp_driver:
            broadcast_tensor_dict({
                "poc_go": True,
                "seq_len": seq_len,
                "hidden_size": hidden_size,
                "nonces": nonces,
                "k_dim": k_dim,
                "poc_stronger_rng": poc_stronger_rng,
                "poc_decode": 1 if poc_decode else 0,
                "max_tokens": max_tokens,
            }, src=0)
        else:
            broadcast_data = broadcast_tensor_dict(src=0)
            seq_len = int(broadcast_data["seq_len"])
            hidden_size = int(broadcast_data["hidden_size"])
            nonces = list(broadcast_data["nonces"])
            k_dim = int(broadcast_data["k_dim"])
            batch_size = len(nonces)
            poc_stronger_rng = bool(broadcast_data["poc_stronger_rng"])
            poc_decode = bool(broadcast_data["poc_decode"])
            max_tokens = int(broadcast_data["max_tokens"])

    pp_group = get_pp_group()

    # Pre-forward sync
    if tp_group.world_size > 1:
        dist.barrier(group=tp_group.cpu_group)
    torch.cuda.synchronize()

    _ensure_layer_hooks(worker, block_hash, hidden_size)

    # Get block_size and prepare attention metadata (cached, reused)
    block_size = _get_block_size(worker)
    attn_metadata, slot_mapping_dict = _get_or_create_attn_metadata(
        batch_size, seq_len, block_size, device, worker
    )

    # Positions for the batch
    positions = torch.arange(seq_len, device=device).repeat(batch_size)

    # Generate inputs for all nonces at once
    intermediate_tensors = None
    inputs_embeds = None

    if pp_group.is_first_rank:
        kv_caches = getattr(worker.model_runner, "kv_caches", [])
        kv_scratch = None
        needed_elems = batch_size * seq_len * hidden_size
        for kv in kv_caches:
            if kv.numel() >= needed_elems:
                kv_scratch = kv.flatten()[:needed_elems].view(
                    batch_size, seq_len, hidden_size)
                break
        if kv_scratch is not None:
            from .gpu_random import _seed_from_string, _normal
            for i, nonce in enumerate(nonces):
                seed = _seed_from_string(
                    f"{block_hash}_{public_key}_nonce{nonce}")
                vals = _normal(seed, seq_len * hidden_size, device)
                kv_scratch[i].copy_(vals.view(seq_len, hidden_size).to(dtype))
                del vals
            inputs_embeds = kv_scratch
        else:
            _gen_fn = generate_inputs_concat_murmur if poc_stronger_rng else generate_inputs
            inputs_embeds = _gen_fn(
                block_hash, public_key, nonces,
                dim=hidden_size, seq_len=seq_len,
                device=device, dtype=dtype,
            )
    else:
        intermediate_tensors = IntermediateTensors(
            pp_group.recv_tensor_dict(all_gather_group=get_tp_group())
        )

    with set_forward_context(
        attn_metadata, vllm_config,
        num_tokens=batch_size * seq_len,
        slot_mapping=slot_mapping_dict,
        skip_compiled=True,
    ):
        with poc_forward_context():
            hidden_states = model(
                input_ids=None,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds.view(-1, hidden_size) if inputs_embeds is not None else None,
            )

    # PP: send to next rank if not last
    if not pp_group.is_last_rank:
        if isinstance(hidden_states, IntermediateTensors):
            pp_group.send_tensor_dict(
                hidden_states.tensors, all_gather_group=get_tp_group()
            )
        return None

    # Handle tuple return
    if isinstance(hidden_states, tuple):
        hidden_states = hidden_states[0]

    # Extract last hidden per sequence
    hidden_states = hidden_states.view(batch_size, seq_len, -1)
    last_hidden = hidden_states[:, -1, :].float()  # [batch_size, hidden_size]

    # NaN detection
    nan_mask = torch.isnan(last_hidden).any(dim=-1)  # [batch_size]
    if nan_mask.any():
        clean_idx = (~nan_mask).nonzero(as_tuple=True)[0]
        nan_count = nan_mask.sum().item()
        logger.warning("NaN in %d/%d hidden states (GPU fault?)", nan_count, batch_size)

        if clean_idx.numel() == 0:
            logger.error("All %d nonces produced NaN — batch rejected", batch_size)
            return {"nonces": [], "vectors": np.empty((0, k_dim), dtype=np.float16)}

        last_hidden = last_hidden[clean_idx]
        nonces = [nonces[i] for i in clean_idx.tolist()]
        batch_size = len(nonces)

    # Normalize to unit sphere
    last_hidden = last_hidden / (last_hidden.norm(dim=-1, keepdim=True) + 1e-8)

    # Batched k-dim pick + Haar rotation
    indices = random_pick_indices(block_hash, public_key, nonces, hidden_size, k_dim, device)
    xk = torch.gather(last_hidden, 1, indices)
    yk = apply_haar_rotation(block_hash, public_key, nonces, xk, device)

    # Normalize output vectors
    yk = yk / (yk.norm(dim=-1, keepdim=True) + 1e-8)

    # Convert to FP16
    vectors_f16 = yk.half().cpu().numpy()  # [batch_size, k_dim]

    # Late NaN check after FP16 conversion
    nan_out = np.isnan(vectors_f16).any(axis=1)
    if nan_out.any():
        clean = ~nan_out
        vectors_f16 = vectors_f16[clean]
        nonces = [n for n, c in zip(nonces, clean) if c]
        if inference_k_points_steps_per_nonce is not None:
            inference_k_points_steps_per_nonce = [
                s for s, c in zip(inference_k_points_steps_per_nonce, clean) if c
            ]
        batch_size = len(nonces)
        logger.warning("NaN in FP16 output — %d nonces filtered", nan_out.sum())

    # -------------------------------------------------------------------------
    # Sphere projection (prefill)
    # -------------------------------------------------------------------------
    codebook = _SPHERE_CODEBOOK.to(device=device, dtype=last_hidden.dtype)
    sphere_indices = random_pick_indices(
        block_hash, public_key, nonces, hidden_size, SPHERE_DIM, device
    )
    xk_sphere = project_to_sphere(torch.gather(last_hidden, 1, sphere_indices))
    nearest_k_points = nearest_sphere_index(xk_sphere, codebook)
    nearest_k_points_list: List[int] = nearest_k_points.cpu().tolist()

    # Per-nonce: sphere_k history (index 0 = prefill)
    k_points_steps_per_nonce: List[List[int]] = [[k] for k in nearest_k_points_list]

    # Mismatch counters: -1 means inference (non-validation) nonce
    if inference_k_points_steps_per_nonce is None:
        mismatch_count: List[int] = [-1] * batch_size
    else:
        mismatch_count = [
            0 if steps is not None else -1
            for steps in inference_k_points_steps_per_nonce
        ]
        for i, steps in enumerate(inference_k_points_steps_per_nonce):
            if steps is not None and len(steps) > 0:
                if nearest_k_points_list[i] != steps[0]:
                    mismatch_count[i] += 1

    # prev_k for decode: validation nonces use inference prefill k to keep trajectories aligned
    prev_k: List[int] = []
    for i in range(batch_size):
        steps = (inference_k_points_steps_per_nonce or [None] * batch_size)[i]
        if steps is not None and len(steps) > 0:
            prev_k.append(steps[0])
        else:
            prev_k.append(nearest_k_points_list[i])

    # Debug buffers
    sph_indices_per_nonce: List[List[List[int]]] = [[] for _ in range(batch_size)]
    sph_values_per_nonce: List[List[str]] = [[] for _ in range(batch_size)]
    if debug:
        sphere_indices_cpu = sphere_indices.cpu().numpy()
        xk_sphere_cpu = xk_sphere.float().cpu().numpy()
        for i in range(batch_size):
            sph_indices_per_nonce[i].append(sphere_indices_cpu[i].tolist())
            sph_values_per_nonce[i].append(encode_vector(xk_sphere_cpu[i]))

    # -------------------------------------------------------------------------
    # Decode loop
    # -------------------------------------------------------------------------
    if poc_decode and max_tokens > 0:
        if pp_group.world_size > 1:
            logger.warning(
                "PoC decode loop not supported with pipeline parallelism "
                "(pp_group.world_size=%d); skipping decode steps.",
                pp_group.world_size,
            )
        else:
            prefill_blocks_per_seq = math.ceil(seq_len / block_size)
            decode_block_start = batch_size * prefill_blocks_per_seq
            max_decode_blocks_per_seq = (
                math.ceil((seq_len + max_tokens) / block_size)
                - prefill_blocks_per_seq
                + 1
            )

            for step in range(1, max_tokens + 1):
                if tp_group.world_size > 1:
                    dist.barrier(group=tp_group.cpu_group)
                    if is_tp_driver:
                        broadcast_tensor_dict({"prev_k": prev_k}, src=0)
                    else:
                        prev_k = list(broadcast_tensor_dict(src=0)["prev_k"])

                decode_embeds = generate_decode_inputs(
                    block_hash, public_key, nonces, prev_k,
                    step=step, dim=hidden_size, device=device, dtype=dtype,
                )

                decode_pos = torch.full(
                    (batch_size,), seq_len + step - 1,
                    device=device, dtype=torch.long,
                )
                dec_attn, dec_slot = _create_decode_attn_metadata_with_history(
                    batch_size, seq_len, step, block_size, device, worker,
                    prefill_blocks_per_seq, max_decode_blocks_per_seq,
                    decode_block_start,
                )

                with set_forward_context(
                    dec_attn, vllm_config,
                    num_tokens=batch_size,
                    slot_mapping=dec_slot,
                    skip_compiled=True,
                ):
                    with poc_forward_context():
                        hs_dec = model(
                            input_ids=None,
                            positions=decode_pos,
                            intermediate_tensors=None,
                            inputs_embeds=decode_embeds.view(-1, hidden_size),
                        )

                if isinstance(hs_dec, tuple):
                    hs_dec = hs_dec[0]
                last_hidden_dec = hs_dec.view(batch_size, 1, -1)[:, 0, :].float()
                last_hidden_dec = last_hidden_dec / (
                    last_hidden_dec.norm(dim=-1, keepdim=True) + 1e-8
                )

                sph_idx_dec = random_pick_indices(
                    block_hash, public_key, nonces, hidden_size, SPHERE_DIM, device,
                    prev_point_ids=prev_k,
                    step=step,
                )
                xk_sph_dec = project_to_sphere(torch.gather(last_hidden_dec, 1, sph_idx_dec))
                step_k_points = nearest_sphere_index(xk_sph_dec, codebook).cpu().tolist()

                if debug:
                    sph_idx_dec_cpu = sph_idx_dec.cpu().numpy()
                    xk_sph_dec_cpu = xk_sph_dec.float().cpu().numpy()
                    for i in range(batch_size):
                        sph_indices_per_nonce[i].append(sph_idx_dec_cpu[i].tolist())
                        sph_values_per_nonce[i].append(encode_vector(xk_sph_dec_cpu[i]))

                for i, computed_k in enumerate(step_k_points):
                    k_points_steps_per_nonce[i].append(computed_k)

                new_prev_k: List[int] = []
                for i, computed_k in enumerate(step_k_points):
                    steps = (inference_k_points_steps_per_nonce or [None] * batch_size)[i]
                    if steps is not None and step < len(steps):
                        inf_k = steps[step]
                        if computed_k != inf_k:
                            mismatch_count[i] += 1
                        new_prev_k.append(inf_k)
                    else:
                        new_prev_k.append(computed_k)
                prev_k = new_prev_k

    return {
        "nonces": nonces,
        "vectors": vectors_f16,
        "sphere_k_list": nearest_k_points_list,
        "k_points_steps_list": k_points_steps_per_nonce,
        "mismatch_count": mismatch_count,
        "sph_indices_steps": sph_indices_per_nonce,
        "sph_values_steps": sph_values_per_nonce,
    }

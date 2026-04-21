"""PoC model runner for V1 architecture.

Executes PoC forward passes for requests identified by the scheduler.
Adapted from v0.9.1 integration to work with V1's unified scheduler.
"""
import statistics
import time
from contextlib import contextmanager
from typing import List, Optional, Dict, Any

import torch
import torch.compiler
import torch.distributed as dist

from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.config import CUDAGraphMode
from vllm.distributed import get_pp_group, get_tp_group
from vllm.distributed.communication_op import broadcast_tensor_dict
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import PoCOutput

from .gpu_random import (
    generate_inputs,
    generate_decode_inputs,
    random_pick_indices,
    apply_haar_rotation,
)
from .data import encode_vector

logger = init_logger(__name__)

DEFAULT_K_DIM = 12

# ── Sphere experiment knobs ───────────────────────────────────────────────
# SPHERE_DIM : dimension of the truncated hidden-state slice.
#              3 = ordinary sphere S², 2 = circle S¹, 16 = hypersphere, …
# K_POINTS   : number of reference codebook points on that sphere.
#              Points are placed to be as equidistant as possible (Thomson problem).
SPHERE_DIM: int = 256
K_POINTS:   int = 16


@contextmanager
def bypass_torch_compile():
    """Temporarily set the compiling flag to bypass torch.compile wrappers.

    The @support_torch_compile decorator checks torch.compiler.is_compiling()
    and bypasses compilation if True. By setting this flag, we force all
    compiled models to use their raw forward() method.

    This is needed for PoC because:
    1. PoC uses input_ids=None with inputs_embeds
    2. torch.dynamo traced the model expecting input_ids to be a tensor
    3. Calling with input_ids=None causes 'NoneType has no attribute size'
    """
    old_flag = torch.compiler._is_compiling_flag
    torch.compiler._is_compiling_flag = True
    try:
        yield
    finally:
        torch.compiler._is_compiling_flag = old_flag


def _create_v1_attn_metadata(
    model_runner,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Create V1 attention metadata for PoC prefill.

    Uses the model runner's metadata builders to create proper backend-specific
    metadata for each attention layer. Uses PAD_SLOT_ID to skip KV cache writes.

    Returns:
        Dict mapping layer names to their backend-specific AttentionMetadata
    """
    from vllm.v1.attention.backends.utils import CommonAttentionMetadata

    num_tokens = batch_size * seq_len

    query_start_loc = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    query_start_loc[1:] = torch.arange(1, batch_size + 1, dtype=torch.int32, device=device) * seq_len
    query_start_loc_cpu = query_start_loc.cpu()

    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    seq_lens_cpu = seq_lens.cpu()

    num_computed_tokens_cpu = torch.zeros(batch_size, dtype=torch.int32, device="cpu")

    block_table_tensor = torch.empty((batch_size, 0), dtype=torch.int32, device=device)

    slot_mapping = torch.full((num_tokens,), PAD_SLOT_ID, dtype=torch.int64, device=device)

    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu,
        num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_reqs=batch_size,
        num_actual_tokens=num_tokens,
        max_query_len=seq_len,
        max_seq_len=seq_len,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        causal=True,
        is_poc=True,
    )

    attn_metadata_dict: Dict[str, Any] = {}

    for attn_groups in model_runner.attn_groups:
        for attn_group in attn_groups:
            builder = attn_group.get_metadata_builder()
            layer_metadata = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
                fast_build=True,  # Skip AOT scheduling for PoC
            )
            for layer_name in attn_group.layer_names:
                attn_metadata_dict[layer_name] = layer_metadata

    return attn_metadata_dict


# ---------------------------------------------------------------------------
# Sphere projection utilities  (experimental)
# ---------------------------------------------------------------------------

def project_to_sphere(v: torch.Tensor) -> torch.Tensor:
    """Normalize [..., dim] vectors to the unit sphere (L2 norm = 1)."""
    return v / (v.norm(dim=-1, keepdim=True) + 1e-8)


def _halton_on_sphere(n_points: int, dim: int) -> torch.Tensor:
    """Return n_points deterministic, low-discrepancy unit vectors on S^(dim-1).

    Uses the Halton sequence (base-prime per dimension) mapped to the sphere
    via the logit transform.  No randomness — identical output for any call
    with the same (n_points, dim).
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

    # [n_points, dim] in (0, 1)^dim  →  logit  →  R^dim  →  sphere
    raw = torch.tensor(coords, dtype=torch.float32).T.clamp(0.01, 0.99)
    pts = torch.log(raw / (1.0 - raw))          # logit: roughly normal spread
    return project_to_sphere(pts)


def build_equidistant_codebook(
    n_points: int,
    dim: int,
    n_steps: int = 500,
    lr: float = 0.05,
) -> torch.Tensor:
    """Build a codebook of approximately equidistant points on S^(dim-1).

    Solves the Thomson problem: minimize the electrostatic repulsion energy
    (sum of 1/distance for all pairs) so points spread as uniformly as
    possible over the sphere.

    Initialisation is deterministic (Halton sequence) — no randomness.
    The result is cached in _SPHERE_CODEBOOK at module load time.

    Args:
        n_points : number of points  → K_POINTS
        dim      : sphere dimension  → SPHERE_DIM
        n_steps  : gradient-descent steps (more = closer to optimum)
        lr       : Adam learning rate

    Returns:
        [n_points, dim]  float32 unit vectors
    """
    # inference_mode(False) is required: the worker process runs inside
    # torch.inference_mode which is stricter than no_grad — tensors cannot
    # track gradients at all, so .backward() would fail without this.
    with torch.inference_mode(mode=False):
        pts = _halton_on_sphere(n_points, dim).clone().requires_grad_(True)
        opt = torch.optim.Adam([pts], lr=lr)

        eye = torch.eye(n_points)                   # mask for diagonal
        for _ in range(n_steps):
            opt.zero_grad()
            p = project_to_sphere(pts)              # keep on sphere each step
            diff = p.unsqueeze(0) - p.unsqueeze(1)  # [n, n, dim]
            d2   = (diff * diff).sum(-1)            # [n, n]  pairwise sq-dist
            # Thomson energy: repulsion ∝ 1/d; mask self-pairs
            energy = ((1.0 - eye) / (d2 + 1e-8)).sum()
            energy.backward()
            opt.step()

        result = project_to_sphere(pts).detach()
    return result


# Built once at module load — reused for every batch.
# Change SPHERE_DIM / K_POINTS at the top and restart to get a new codebook.
_SPHERE_CODEBOOK: torch.Tensor = build_equidistant_codebook(K_POINTS, SPHERE_DIM)


def nearest_sphere_index(
    query: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    """Return the index of the nearest codebook point for each query vector.

    On the unit sphere cosine similarity = dot product, so argmax of the
    dot product gives the nearest neighbour.

    Args:
        query    : [batch, dim]    — unit vectors (one per nonce)
        codebook : [K_POINTS, dim] — unit vectors from _SPHERE_CODEBOOK

    Returns:
        [batch]  int64 — index k in [0, K_POINTS)
    """
    sims = query.float() @ codebook.float().T   # [batch, K_POINTS]
    return sims.argmax(dim=-1)                  # [batch]


@torch.inference_mode()
def execute_poc_batch(
    model_runner,
    poc_requests: List[Any],  # List of Request objects with poc_params
    intermediate_tensors: Optional["IntermediateTensors"] = None,
) -> "List[PoCOutput] | IntermediateTensors":
    """Execute PoC forward pass for a batch of PoC requests.

    Args:
        model_runner: The GPUModelRunner instance
        poc_requests: List of Request objects with poc_params set
        intermediate_tensors: Pre-received tensors from the previous PP stage
            (provided by gpu_worker.py for non-first PP ranks).

    Returns:
        List[PoCOutput] on the last PP rank, or IntermediateTensors on
        non-last PP ranks (for gpu_worker.py to send to the next stage).
    """
    if not poc_requests:
        return []
    
    device = model_runner.device
    dtype = model_runner.model_config.dtype
    model = model_runner.model
    vllm_config = model_runner.vllm_config
    
    tp_group = get_tp_group()
    is_tp_driver = tp_group.rank_in_group == 0
    
    # Extract PoC parameters from requests
    first_poc = poc_requests[0].poc_params
    block_hash = first_poc.block_hash
    public_key = first_poc.public_key
    seq_len = first_poc.seq_len
    k_dim = first_poc.k_dim
    
    nonces = [req.poc_params.nonce for req in poc_requests]
    batch_size = len(nonces)
    hidden_size = model_runner.model_config.get_hidden_size()

    from vllm.poc.layer_hooks import LayerHouseholderHook

    cached_hooks = getattr(model_runner, '_poc_layer_hooks', None)
    if cached_hooks is None or cached_hooks.block_hash != block_hash:
        if cached_hooks is not None:
            cached_hooks.detach()  # Clean up old hooks
        cached_hooks = LayerHouseholderHook(model, block_hash, device, hidden_size)
        model_runner._poc_layer_hooks = cached_hooks
        logger.debug(f"PoC: Registered {cached_hooks.num_layers} layer hooks for block_hash={block_hash[:16]}...")

    poc_decode = first_poc.poc_decode
    max_tokens = first_poc.max_tokens

    if tp_group.world_size > 1:
        # Rendezvous: ensure all TP ranks have entered before broadcast
        dist.barrier(group=tp_group.cpu_group)
        
        if is_tp_driver:
            # Driver: broadcast PoC metadata (Python values only - uses CPU group)
            broadcast_tensor_dict({
                "poc_go": True,
                "seq_len": seq_len,
                "hidden_size": hidden_size,
                "nonces": nonces,
                "k_dim": k_dim,
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
            poc_decode = bool(broadcast_data["poc_decode"])
            max_tokens = int(broadcast_data["max_tokens"])
    
    pp_group = get_pp_group()

    inputs_embeds = None
    
    # =========================================================================
    # TIMING: Phase 1 - Input Generation
    # =========================================================================
    torch.cuda.synchronize()
    t_input_start = time.perf_counter()
    
    if pp_group.is_first_rank:
        # Generate deterministic inputs on GPU
        inputs_embeds = generate_inputs(
            block_hash, public_key, nonces,
            dim=hidden_size, seq_len=seq_len,
            device=device, dtype=dtype,
        )
        if inputs_embeds is None:
            raise RuntimeError(f"generate_inputs returned None for batch_size={batch_size}, "
                             f"seq_len={seq_len}, hidden_size={hidden_size}")
        logger.debug(f"PoC inputs_embeds shape: {inputs_embeds.shape}")
    
    # Create attention metadata and positions
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    attn_metadata = _create_v1_attn_metadata(model_runner, batch_size, seq_len, device)

    torch.cuda.synchronize()
    t_input_end = time.perf_counter()
    
    # =========================================================================
    # TP SYNC: Pre-forward rendezvous
    # =========================================================================
    if tp_group.world_size > 1:
        dist.barrier(group=tp_group.cpu_group)
    
    torch.cuda.synchronize()
    
    # =========================================================================
    # TIMING: Phase 2 - Model Forward
    # =========================================================================
    t_fwd_start = time.perf_counter()
    
    # Forward pass with PoC context (activates layer hooks)
    from vllm.poc.layer_hooks import poc_forward_context

    with set_forward_context(attn_metadata, vllm_config,
                             cudagraph_runtime_mode=CUDAGraphMode.NONE):
        with poc_forward_context():
            with bypass_torch_compile():
                hidden_states = model(
                    input_ids=None,
                    positions=positions.flatten(),
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds.view(-1, hidden_size) if inputs_embeds is not None else None,
                )
    
    torch.cuda.synchronize()
    t_fwd_end = time.perf_counter()

    if not pp_group.is_last_rank:
        assert isinstance(hidden_states, IntermediateTensors), (
            f"Expected IntermediateTensors from non-last PP rank model forward, "
            f"got {type(hidden_states)}"
        )
        hidden_states.kv_connector_output = None
        return hidden_states
    
    # =========================================================================
    # TIMING: Phase 3 - Post-processing (prefill)
    # =========================================================================
    t_post_start = time.perf_counter()
    
    # Extract last token hidden state
    hidden_states = hidden_states.view(batch_size, seq_len, -1)
    last_hidden = hidden_states[:, -1, :].float()
    
    # Normalize to unit sphere
    last_hidden = last_hidden / (last_hidden.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Per-nonce k-dim pick + Haar rotation
    indices = random_pick_indices(block_hash, public_key, nonces, hidden_size, k_dim, device)
    xk = torch.gather(last_hidden, 1, indices)

    yk = apply_haar_rotation(block_hash, public_key, nonces, xk, device)

    # Normalize
    yk = yk / (yk.norm(dim=-1, keepdim=True) + 1e-8)

    # Convert to FP16 for artifact encoding
    vectors_f16 = yk.half().cpu().numpy()

    # ── Sphere projection experiment ──────────────────────────────────────
    # Pick SPHERE_DIM dimensions from the hidden state and project to sphere.
    # The K_POINTS equidistant codebook is pre-built at module load (_SPHERE_CODEBOOK).
    t_sphere_start = time.perf_counter()
    sphere_indices = random_pick_indices(
        block_hash, public_key, nonces, hidden_size, SPHERE_DIM, device
    )
    xk_sphere = project_to_sphere(
        torch.gather(last_hidden, 1, sphere_indices)
    )   # [batch, SPHERE_DIM]  — unit vectors on S^(SPHERE_DIM-1)

    codebook = _SPHERE_CODEBOOK.to(device=device, dtype=last_hidden.dtype)
    # k ∈ [0, K_POINTS): index of nearest equidistant codebook point per nonce
    sphere_k      = nearest_sphere_index(xk_sphere, codebook)  # [batch]
    sphere_k_list = sphere_k.cpu().tolist()
    t_sphere_end = time.perf_counter()
    t_sphere = t_sphere_end - t_sphere_start

    last_hidden_cpu = last_hidden.float().cpu().numpy()
    xk_sphere_cpu   = xk_sphere.float().cpu().numpy()

    torch.cuda.synchronize()
    t_post_end = time.perf_counter()

    # =========================================================================
    # TIMING: Phase 4 - Decode loop (poc_decode mode)
    # =========================================================================
    # sphere_k_steps_per_nonce[i]: computed k at each step (index 0 = prefill).
    sphere_k_steps_per_nonce: List[List[int]] = [[k] for k in sphere_k_list]

    # Per-nonce mismatch counters for validation requests.
    # -1 means the nonce is an inference (non-validation) request.
    inf_steps_per_nonce: List[Optional[List[int]]] = [
        req.poc_params.inference_sphere_k_steps for req in poc_requests
    ]
    mismatch_count: List[int] = [
        0 if s is not None else -1 for s in inf_steps_per_nonce
    ]

    # Check prefill k against inference reference (step 0).
    for i, inf_steps in enumerate(inf_steps_per_nonce):
        if inf_steps is not None and len(inf_steps) > 0:
            if sphere_k_list[i] != inf_steps[0]:
                mismatch_count[i] += 1

    # Initial prev_k for the decode loop: validation requests use the
    # inference prefill k so both servers share the same decode trajectory.
    prev_k: List[int] = [
        (inf_steps_per_nonce[i][0]
         if inf_steps_per_nonce[i] is not None and len(inf_steps_per_nonce[i]) > 0
         else sphere_k_list[i])
        for i in range(batch_size)
    ]

    t_decode_total = 0.0
    t_sphere_decode_total = 0.0
    t_decode_step_list: List[float] = []
    t_sphere_decode_step_list: List[float] = []

    if poc_decode and max_tokens > 0:
        if pp_group.world_size > 1:
            logger.warning(
                "PoC decode loop is not supported with pipeline parallelism "
                "(pp_group.world_size=%d); skipping decode steps.",
                pp_group.world_size,
            )
        else:
            for step in range(1, max_tokens + 1):
                torch.cuda.synchronize()
                t_dec_step_start = time.perf_counter()

                # ── TP sync ──────────────────────────────────────────────
                if tp_group.world_size > 1:
                    dist.barrier(group=tp_group.cpu_group)

                # Generate decode embedding seeded by prev_k.
                # For validation nonces prev_k already holds the inference k,
                # so both servers run an identical forward pass.
                decode_embeds = generate_decode_inputs(
                    block_hash, public_key, nonces, prev_k,
                    step=step, dim=hidden_size,
                    device=device, dtype=dtype,
                )  # [batch_size, 1, hidden_size]

                decode_pos = torch.full(
                    (batch_size,), seq_len + step - 1,
                    device=device, dtype=torch.long,
                )
                decode_attn_meta = _create_v1_attn_metadata(
                    model_runner, batch_size, 1, device
                )

                with set_forward_context(decode_attn_meta, vllm_config,
                                         cudagraph_runtime_mode=CUDAGraphMode.NONE):
                    with poc_forward_context():
                        with bypass_torch_compile():
                            hs_dec = model(
                                input_ids=None,
                                positions=decode_pos,
                                intermediate_tensors=None,
                                inputs_embeds=decode_embeds.view(-1, hidden_size),
                            )

                torch.cuda.synchronize()

                hs_dec = hs_dec.view(batch_size, 1, -1)
                last_hidden_dec = hs_dec[:, 0, :].float()
                last_hidden_dec = last_hidden_dec / (
                    last_hidden_dec.norm(dim=-1, keepdim=True) + 1e-8
                )

                t_sphere_dec_step_start = time.perf_counter()
                sph_idx_dec = random_pick_indices(
                    block_hash, public_key, nonces, hidden_size, SPHERE_DIM, device
                )
                xk_sph_dec = project_to_sphere(
                    torch.gather(last_hidden_dec, 1, sph_idx_dec)
                )
                sphere_k_dec = nearest_sphere_index(xk_sph_dec, codebook)
                step_k_list: List[int] = sphere_k_dec.cpu().tolist()
                t_sphere_dec_step_stop = time.perf_counter()


                for i, computed_k in enumerate(step_k_list):
                    sphere_k_steps_per_nonce[i].append(computed_k)

                # Build prev_k for the next step.  Validation nonces use the
                # inference reference k so the trajectories stay aligned.
                new_prev_k: List[int] = []
                for i, computed_k in enumerate(step_k_list):
                    inf_steps = inf_steps_per_nonce[i]
                    if inf_steps is not None and step < len(inf_steps):
                        inf_k = inf_steps[step]
                        if computed_k != inf_k:
                            mismatch_count[i] += 1
                        new_prev_k.append(inf_k)
                    else:
                        new_prev_k.append(computed_k)
                prev_k = new_prev_k

                t_dec_step_stop = time.perf_counter()
                t_decode_total += t_dec_step_stop - t_dec_step_start
                t_decode_step_list.append(t_dec_step_stop - t_dec_step_start)

                t_sphere_decode_step = t_sphere_dec_step_stop - t_sphere_dec_step_start
                t_sphere_decode_total += t_sphere_decode_step
                t_sphere_decode_step_list.append(t_sphere_decode_step)   

            logger.debug(
                "PoC decode: %d steps completed in %.4fs total",
                max_tokens, t_decode_total,
            )

    t_input = t_input_end - t_input_start
    t_fwd = t_fwd_end - t_fwd_start
    t_post = t_post_end - t_post_start
    t_total = t_input + t_fwd + t_post + t_decode_total
    t_decode_step_median = statistics.median(t_decode_step_list) if t_decode_step_list else 0.0
    t_sphere_decode_step_median = statistics.median(t_sphere_decode_step_list) if t_sphere_decode_step_list else 0.0

    logger.info(
        f"POC Timing: batch={batch_size}, seq_len={seq_len}, "
        f"decode_steps={max_tokens if poc_decode else 0} | "
        f"input_gen={t_input:.4f}s, model_fwd={t_fwd:.4f}s, "
        f"postproc={t_post:.4f}s, decode={t_decode_total:.4f}s, "
        f"sphere_projection_prefill={t_sphere:.4f}s, "
        f"sphere_projection_decode={t_sphere_decode_total:.4f}s, "
        f"decode_step_median={t_decode_step_median:.4f}s, "
        f"sphere_decode_step_median={t_sphere_decode_step_median:.4f}s, "
        f"total={t_total:.4f}s"
    )

    results = []
    for i, nonce in enumerate(nonces):
        vector_b64 = encode_vector(vectors_f16[i])
        results.append(PoCOutput(
            nonce=nonce,
            vector_b64=vector_b64,
            hidden_state_b64=encode_vector(last_hidden_cpu[i]),
            reduced_hidden_state_b64=encode_vector(xk_sphere_cpu[i]),
            sphere_k=sphere_k_list[i],
            sphere_k_steps=sphere_k_steps_per_nonce[i],
            n_sphere_mismatches=mismatch_count[i],
        ))

    return results

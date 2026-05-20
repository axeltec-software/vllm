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
    random_pick_indices,
    apply_haar_rotation,
)
from .layer_hooks import LayerHouseholderHook, poc_forward_context

logger = init_logger(__name__)

DEFAULT_K_DIM = 12

# NOTE: attention metadata must NOT be cached across PoC calls.
# The metadata builder's internal state (workspace buffers, page-table
# references) is mutated by every inference engine step.  Reusing a
# stale metadata object causes the attention backend to write only a
# fraction of the expected KV entries, producing all-NaN hidden states.
# The cost of rebuilding is <1 ms per call (vs ~15 ms for the model
# forward), so the overhead is negligible.


def _ensure_layer_hooks(worker, block_hash, hidden_size):
    """Ensure layer hooks are installed and current for the given block_hash."""
    model = worker.model_runner.model
    device = worker.device
    existing_hook = getattr(worker, "_poc_layer_hooks", None)
    if existing_hook is None:
        hook = LayerHouseholderHook(model, block_hash, device, hidden_size)
        hook._setup(model, block_hash, device, hidden_size)
        worker._poc_layer_hooks = hook
    elif existing_hook.block_hash != block_hash:
        # Update contents in-place — no hook re-registration, no re-capture.
        existing_hook.update_block_hash(block_hash, hidden_size, device)


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


def _get_poc_input_buffers(worker, batch_size, seq_len, hidden_size, device, dtype):
    """Return (embeds_buf, positions_buf) with stable GPU addresses.

    Allocated once per (batch_size, seq_len); the same tensor objects
    are reused on every subsequent call so CUDAGraphWrapper sees the
    same data_ptr() during capture and replay.
    Positions are fixed (arange(seq_len).repeat(batch_size)) and never
    need updating between calls.
    """
    cache = getattr(worker, "_poc_input_bufs", {})
    key = (batch_size, seq_len)
    if key not in cache:
        embeds_buf = torch.zeros(
            batch_size * seq_len, hidden_size, device=device, dtype=dtype
        )
        positions_buf = (
            torch.arange(seq_len, device=device, dtype=torch.long)
            .repeat(batch_size)
        )
        cache[key] = (embeds_buf, positions_buf)
        worker._poc_input_bufs = cache
    return cache[key]

def _get_static_attn_metadata(worker, batch_size, seq_len, block_size, device):
    """Return cached attention metadata with stable tensor addresses.

    The existing NOTE (lines 31–37) warns against caching because the
    metadata builder's workspace is mutated by inference engine steps.
    For piecewise CUDA graphs this trade-off is accepted: the POC forward
    runs between engine scheduling ticks (via collective_rpc), so the
    builder's workspace is idle when the POC runs. Metadata is cached
    per (batch_size, seq_len); a different shape triggers a new build.
    """
    cache = getattr(worker, "_poc_attn_meta_cache", {})
    key = (batch_size, seq_len)
    if key not in cache:
        meta, slot_map = _create_v1_attn_metadata(
            batch_size, seq_len, block_size, device, worker
        )
        cache[key] = (meta, slot_map)
        worker._poc_attn_meta_cache = cache
    return cache[key]


def _get_or_capture_poc_graph(
    worker,
    batch_size, seq_len, hidden_size,
    device, dtype, vllm_config,
    attn_metadata, slot_mapping_dict,
    embeds_buf, positions_buf,
):
    """Return (graph, captured_hidden) for the given (batch_size, seq_len) shape.

    First call: runs one eager warmup forward (JIT compile + FlashInfer plan()),
    then captures the full model forward into a torch.cuda.CUDAGraph.
    Subsequent calls: returns the cached (graph, captured_hidden) immediately.

    After each graph.replay() captured_hidden contains the updated output.
    Updating embeds_buf in-place before replay feeds new embeddings to the graph.
    Updating reflection_vectors in-place (via update_block_hash) feeds new
    Householder vectors to the captured hook ops.
    """
    cache = getattr(worker, "_poc_cuda_graphs", {})
    key = (batch_size, seq_len)
    if key in cache:
        return cache[key]

    model = worker.model_runner.model
    num_tokens = batch_size * seq_len

    # Warmup: compiles JIT kernels and runs FlashInfer plan() which sets up
    # GPU workspace tensors at stable addresses used by all future replays.
    with set_forward_context(
        attn_metadata, vllm_config,
        num_tokens=num_tokens,
        slot_mapping=slot_mapping_dict,
        skip_compiled=True,
    ):
        with poc_forward_context():
            _ = model(
                input_ids=None,
                positions=positions_buf,
                inputs_embeds=embeds_buf,
            )
    torch.cuda.synchronize()

    # Isolated pool so POC graphs don't compete with the inference engine pool.
    if not hasattr(worker, "_poc_graph_pool"):
        worker._poc_graph_pool = torch.cuda.graph_pool_handle()

    graph = torch.cuda.CUDAGraph()
    with set_forward_context(
        attn_metadata, vllm_config,
        num_tokens=num_tokens,
        slot_mapping=slot_mapping_dict,
        skip_compiled=True,
    ):
        with poc_forward_context():
            with torch.cuda.graph(graph, pool=worker._poc_graph_pool):
                captured_out = model(
                    input_ids=None,
                    positions=positions_buf,
                    inputs_embeds=embeds_buf,
                )

    if isinstance(captured_out, tuple):
        captured_hidden = captured_out[0]
    else:
        captured_hidden = captured_out

    cache[key] = (graph, captured_hidden)
    worker._poc_cuda_graphs = cache
    logger.info(
        "POC CUDA graph captured for batch_size=%d seq_len=%d (%d tokens)",
        batch_size, seq_len, num_tokens,
    )
    return graph, captured_hidden


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
) -> Optional[Dict[str, Any]]:
    """Execute batched PoC forward pass on a V1 worker.

    Processes all nonces in a single forward call for maximum throughput.
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
            }, src=0)
        else:
            broadcast_data = broadcast_tensor_dict(src=0)
            seq_len = int(broadcast_data["seq_len"])
            hidden_size = int(broadcast_data["hidden_size"])
            nonces = list(broadcast_data["nonces"])
            k_dim = int(broadcast_data["k_dim"])
            batch_size = len(nonces)
            poc_stronger_rng = bool(broadcast_data["poc_stronger_rng"])

    pp_group = get_pp_group()

    # Pre-forward sync
    if tp_group.world_size > 1:
        dist.barrier(group=tp_group.cpu_group)
    torch.cuda.synchronize()

    _ensure_layer_hooks(worker, block_hash, hidden_size)

    # Get block_size and prepare attention metadata (cached, reused)
    block_size = _get_block_size(worker)
    attn_metadata, slot_mapping_dict = _get_static_attn_metadata(
        worker, batch_size, seq_len, block_size, device
    )

    embeds_buf, positions_buf = _get_poc_input_buffers(
        worker, batch_size, seq_len, hidden_size, device, dtype
    )

    # Generate embeddings and copy into the static buffer so the
    # CUDAGraphWrapper always sees the same data_ptr() for inputs_embeds
    intermediate_tensors = None

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
            embeds_buf.copy_(kv_scratch.view(-1, hidden_size))
        else:
            _gen_fn = generate_inputs_concat_murmur if poc_stronger_rng else generate_inputs
            raw = _gen_fn(
                block_hash, public_key, nonces,
                dim=hidden_size, seq_len=seq_len,
                device=device, dtype=dtype,
            )
            embeds_buf.copy_(raw.view(-1, hidden_size))
        graph, captured_hidden = _get_or_capture_poc_graph(
            worker,
            batch_size, seq_len, hidden_size,
            device, dtype, vllm_config,
            attn_metadata, slot_mapping_dict,
            embeds_buf, positions_buf,
        )
        with set_forward_context(
            attn_metadata, vllm_config,
            num_tokens=batch_size * seq_len,
            slot_mapping=slot_mapping_dict,
            skip_compiled=True,
        ):
            with poc_forward_context():
                graph.replay()
        hidden_states = captured_hidden
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
                    positions=positions_buf,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=None,
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
        logger.warning("NaN in FP16 output — %d nonces filtered", nan_out.sum())

    return {
        "nonces": nonces,
        "vectors": vectors_f16,
    }

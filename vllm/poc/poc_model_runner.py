"""PoC model runner for V1 architecture.

Executes PoC forward passes for requests identified by the scheduler.
Adapted from v0.9.1 integration to work with V1's unified scheduler.
"""
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
    random_pick_indices,
    apply_haar_rotation,
)
from .data import encode_vector

logger = init_logger(__name__)

DEFAULT_K_DIM = 12


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


@torch.inference_mode()
def execute_poc_batch(
    model_runner,
    poc_requests: List[Any],  # List of Request objects with poc_params
) -> List[PoCOutput]:
    """Execute PoC forward pass for a batch of PoC requests.
    
    Args:
        model_runner: The GPUModelRunner instance
        poc_requests: List of Request objects with poc_params set
        
    Returns:
        List of PoCOutput objects, one per request
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
            }, src=0)
        else:
            broadcast_data = broadcast_tensor_dict(src=0)
            seq_len = int(broadcast_data["seq_len"])
            hidden_size = int(broadcast_data["hidden_size"])
            nonces = list(broadcast_data["nonces"])
            k_dim = int(broadcast_data["k_dim"])
            batch_size = len(nonces)
    
    # Generate embeddings on first PP rank, receive intermediate tensors on others
    intermediate_tensors = None
    inputs_embeds = None
    
    pp_group = get_pp_group()
    
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
    else:
        # Receive from previous PP rank
        intermediate_tensors = IntermediateTensors(
            pp_group.recv_tensor_dict(all_gather_group=get_tp_group())
        )
    
    # Create attention metadata and positions
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    attn_metadata = None

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
    
    # PP: send to next rank if not last
    if not pp_group.is_last_rank:
        if isinstance(hidden_states, IntermediateTensors):
            pp_group.send_tensor_dict(
                hidden_states.tensors, all_gather_group=get_tp_group()
            )
        return []
    
    # =========================================================================
    # TIMING: Phase 3 - Post-processing
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

    torch.cuda.synchronize()
    t_post_end = time.perf_counter()

    t_input = t_input_end - t_input_start
    t_fwd = t_fwd_end - t_fwd_start
    t_post = t_post_end - t_post_start
    t_total = t_input + t_fwd + t_post
    logger.info(
        f"POC Timing: batch={batch_size}, seq_len={seq_len} | "
        f"input_gen={t_input:.4f}s, model_fwd={t_fwd:.4f}s, postproc={t_post:.4f}s, "
        f"total={t_total:.4f}s"
    )

    results = []
    for i, nonce in enumerate(nonces):
        vector_b64 = encode_vector(vectors_f16[i])
        results.append(PoCOutput(
            nonce=nonce,
            vector_b64=vector_b64,
        ))

    return results

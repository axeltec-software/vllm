# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
import torch

_SAMPLING_EPS = 1e-5


def is_spec_decode_unsupported(sampling_params: SamplingParams) -> bool:
    """True if request is incompatible with speculative decoding"""
    return (sampling_params.frequency_penalty != 0.0
            or sampling_params.presence_penalty != 0.0
            or sampling_params.repetition_penalty != 1.0
            or sampling_params.min_p > _SAMPLING_EPS
            or sampling_params.logprobs is not None)


@triton.jit
def get_valid_tokens_kernel(
    sampled_token_ids_ptr,
    cu_num_draft_tokens_ptr,
    valid_tokens_ptr,
    num_rejected_tokens_ptr,
    num_tokens: tl.constexpr,
):
    req_id = tl.program_id(axis=0)
    sampled_token_ids_ptr += req_id * num_tokens
    
    if req_id == 0:
        start_draft_token = 0
    else:
        start_draft_token = tl.load(cu_num_draft_tokens_ptr + req_id - 1)
    end_draft_token = tl.load(cu_num_draft_tokens_ptr + req_id)
    num_draft_token = end_draft_token - start_draft_token

    last_valid_token = -1
    rejected_count = 0

    for token_id in tl.static_range(num_tokens):
        token = tl.load(sampled_token_ids_ptr + token_id)
        last_valid_token = tl.where(token != -1, token, last_valid_token)
        rejected_count += (token == -1)

    tl.store(valid_tokens_ptr + req_id, last_valid_token)
    
    final_rejected_count = tl.where(num_draft_token > 0, rejected_count, 0)
    tl.store(num_rejected_tokens_ptr + req_id, final_rejected_count)


def get_valid_tokens(
    sampled_token_ids: torch.Tensor,
    cu_num_draft_tokens: torch.Tensor,
):
    num_req, num_tokens = sampled_token_ids.size()
    
    next_token_ids = torch.empty(num_req, dtype=torch.int32, device="cuda")
    num_rejected_tokens = torch.empty(num_req, dtype=torch.int32, device="cuda")
    
    get_valid_tokens_kernel[(num_req,)](
        sampled_token_ids,
        cu_num_draft_tokens,
        next_token_ids,
        num_rejected_tokens,
        num_tokens=num_tokens,
    )
    
    return next_token_ids, num_rejected_tokens

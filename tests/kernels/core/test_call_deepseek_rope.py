import torch
import pytest
from torch.profiler import profile, ProfilerActivity
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import DeepseekScalingRotaryEmbedding


@pytest.mark.parametrize("is_neox_style, use_offsets, expected_func", [
    (False, False, "rotary_embedding_deepseek_fused"),
    (True,  False, "rotary_embedding_deepseek_neox_fused"),
    (False, True,  "rotary_embedding_deepseek_offsets_fused"),
    (True,  True,  "rotary_embedding_deepseek_neox_offsets_fused"),
])
def test_forward_calls_correct_custom_op(is_neox_style, use_offsets, expected_func):
    batch_size = 1
    base = 10000
    num_heads = 7
    max_position = 11
    head_size=64
    rotary_dim = 32
    device = "cuda"
    scaling_factor = 1.0
    seq_len = 11
    head_stride = head_size
    rot = DeepseekScalingRotaryEmbedding(head_size, rotary_dim, max_position, base,
                          is_neox_style, scaling_factor, torch.bfloat16)


    query = torch.randn(batch_size,
                        seq_len,
                        num_heads,
                        head_stride,
                        dtype=torch.bfloat16,
                        device=device)
    key = torch.randn_like(query)
    query = query[..., :head_size]
    key = key[..., :head_size]
    positions = torch.randint(0,
                              max_position, (batch_size, seq_len),
                              device=device)
    offsets = torch.zeros(batch_size * seq_len,
                          device=device,
                          dtype=torch.long) if use_offsets else None

    expected_op_name = f"_C::{expected_func}"

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        out_query, out_key = rot.forward(positions, query, key, offsets)

    ops = [e.key for e in prof.key_averages()]

    assert any(expected_op_name in op for op in ops), (
        f"Expected op '{expected_op_name}' not found in: {ops}"
    )

    assert out_query.shape == query.shape
    assert out_key.shape == key.shape


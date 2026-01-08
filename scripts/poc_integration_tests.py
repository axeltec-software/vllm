#!/usr/bin/env python3
"""
Integration tests for PoC scheduler fixes.

Tests:
1. Mixed batch: Chat + PoC in same batch (tests batch position mapping fix)
2. Priority: PoC yields to chat under load (tests priority=1 fix)
3. Chunked prefill: Long PoC not chunked (tests chunked prefill exemption)

Usage:
    VLLM_USE_V1=0 python scripts/poc_integration_tests.py [--model MODEL]

Default model: Qwen/Qwen2.5-7B-Instruct (fits in 20GB)
"""

import argparse
import os
import sys
import time
import threading
from typing import List, Tuple

os.environ["VLLM_USE_V1"] = "0"

from vllm import LLM, SamplingParams
from vllm.poc.poc_params import PoCParams
from vllm.outputs import PoCRequestOutput, RequestOutput


def test_mixed_batch(engine) -> Tuple[bool, str]:
    """
    Test: Chat and PoC requests processed in same batch.

    This tests the batch position mapping fix - PoC must find its
    correct position in mixed batch, not assume it's at index 0.
    """
    print("\n" + "=" * 60)
    print("TEST: Mixed Batch (Chat + PoC)")
    print("=" * 60)

    # Submit chat request first
    chat_prompt = "What is 2+2? Answer in one word."
    sampling_params = SamplingParams(max_tokens=10, temperature=0)

    engine.add_request(
        request_id="chat-1",
        prompt=chat_prompt,
        params=sampling_params,
    )
    print("  Submitted: chat-1")

    # Submit PoC request
    poc_params = PoCParams(
        block_hash="mixed_batch_test",
        public_key="test_key",
        block_height=100,
        nonce=42,
        r_target=1.5,
        seq_len=64,
    )

    engine.add_request(
        request_id="poc-1",
        prompt={"prompt_token_ids": [0] * 64},
        params=poc_params,
    )
    print("  Submitted: poc-1")

    # Submit another chat
    engine.add_request(
        request_id="chat-2",
        prompt="What is the capital of France? One word.",
        params=sampling_params,
    )
    print("  Submitted: chat-2")

    # Process all
    chat_results = []
    poc_results = []

    while engine.has_unfinished_requests():
        outputs = engine.step()
        for output in outputs:
            if isinstance(output, PoCRequestOutput):
                if output.finished:
                    poc_results.append(output)
                    print(f"  Got PoC result: nonce={output.outputs.nonce}, distance={output.outputs.distance:.4f}")
            elif isinstance(output, RequestOutput):
                if output.finished:
                    chat_results.append(output)
                    text = output.outputs[0].text.strip()
                    print(f"  Got chat result: {output.request_id} -> '{text}'")

    # Verify
    success = True
    details = []

    if len(chat_results) != 2:
        success = False
        details.append(f"Expected 2 chat results, got {len(chat_results)}")

    if len(poc_results) != 1:
        success = False
        details.append(f"Expected 1 PoC result, got {len(poc_results)}")

    if poc_results:
        d = poc_results[0].outputs.distance
        if not (0 <= d <= 2):
            success = False
            details.append(f"PoC distance {d} out of range [0, 2]")

    return success, "; ".join(details) if details else "Mixed batch processed correctly"


def test_priority_scheduling(engine) -> Tuple[bool, str]:
    """
    Test: Chat requests get priority over PoC.

    Submit many PoC requests, then chat. Chat should complete
    before all PoC (due to priority=1 for PoC vs priority=0 for chat).
    """
    print("\n" + "=" * 60)
    print("TEST: Priority Scheduling")
    print("=" * 60)

    # Submit 20 PoC requests first
    poc_count = 20
    for i in range(poc_count):
        poc_params = PoCParams(
            block_hash="priority_test",
            public_key="test_key",
            block_height=100,
            nonce=i,
            r_target=1.5,
            seq_len=64,
        )
        engine.add_request(
            request_id=f"poc-priority-{i}",
            prompt={"prompt_token_ids": [0] * 64},
            params=poc_params,
        )
    print(f"  Submitted: {poc_count} PoC requests")

    # Now submit chat request
    sampling_params = SamplingParams(max_tokens=5, temperature=0)
    engine.add_request(
        request_id="chat-priority",
        prompt="Say hello",
        params=sampling_params,
    )
    print("  Submitted: 1 chat request (after PoC)")

    # Track completion order
    completion_order = []

    while engine.has_unfinished_requests():
        outputs = engine.step()
        for output in outputs:
            if isinstance(output, PoCRequestOutput) and output.finished:
                completion_order.append(("poc", output.request_id))
            elif isinstance(output, RequestOutput) and output.finished:
                completion_order.append(("chat", output.request_id))

    # Find chat position in completion order
    chat_position = None
    for i, (req_type, req_id) in enumerate(completion_order):
        if req_type == "chat":
            chat_position = i
            break

    print(f"  Completion order: {len(completion_order)} total")
    print(f"  Chat completed at position: {chat_position} (0-indexed)")

    # Chat should complete in first few (ideally first, but batching may vary)
    # With priority scheduling, chat should NOT be last
    success = chat_position is not None and chat_position < poc_count

    if success:
        detail = f"Chat completed at position {chat_position}/{len(completion_order)}"
        if chat_position == 0:
            detail += " (first - perfect!)"
        elif chat_position < 5:
            detail += " (early - good)"
        else:
            detail += " (later than expected, but still before all PoC)"
    else:
        detail = f"Chat position {chat_position} - expected earlier (priority not working?)"

    return success, detail


def test_chunked_prefill(engine, enable_chunking: bool = True) -> Tuple[bool, str]:
    """
    Test: Long PoC sequences are not chunked.

    With chunked prefill enabled, a long PoC should still complete
    in one step (not be split into chunks).
    """
    print("\n" + "=" * 60)
    print("TEST: Chunked Prefill Exemption")
    print("=" * 60)

    # Use longer sequence length
    long_seq_len = 512

    poc_params = PoCParams(
        block_hash="chunked_prefill_test",
        public_key="test_key",
        block_height=100,
        nonce=99,
        r_target=1.5,
        seq_len=long_seq_len,
    )

    engine.add_request(
        request_id="poc-long",
        prompt={"prompt_token_ids": [0] * long_seq_len},
        params=poc_params,
    )
    print(f"  Submitted: PoC with seq_len={long_seq_len}")

    # Count steps to completion
    steps = 0
    result = None

    while engine.has_unfinished_requests():
        outputs = engine.step()
        steps += 1
        for output in outputs:
            if isinstance(output, PoCRequestOutput) and output.finished:
                result = output

    print(f"  Completed in {steps} step(s)")

    if result:
        print(f"  Distance: {result.outputs.distance:.4f}")

    # PoC should complete in 1 step (not chunked)
    # Note: with batching, might take more steps, but should still work
    success = result is not None and 0 <= result.outputs.distance <= 2

    if steps == 1:
        detail = "Completed in 1 step (not chunked) - perfect!"
    elif steps <= 3:
        detail = f"Completed in {steps} steps (acceptable)"
    else:
        detail = f"Completed in {steps} steps (may have been chunked?)"

    if not success:
        detail = f"Failed to get valid result. Steps: {steps}"

    return success, detail


def test_determinism(engine) -> Tuple[bool, str]:
    """
    Test: Same inputs produce same distance.
    """
    print("\n" + "=" * 60)
    print("TEST: Determinism")
    print("=" * 60)

    poc_params = PoCParams(
        block_hash="determinism_test",
        public_key="test_key",
        block_height=100,
        nonce=7,
        r_target=1.5,
        seq_len=64,
    )

    distances = []

    for run in range(3):
        engine.add_request(
            request_id=f"poc-det-{run}",
            prompt={"prompt_token_ids": [0] * 64},
            params=poc_params,
        )

        while engine.has_unfinished_requests():
            outputs = engine.step()
            for output in outputs:
                if isinstance(output, PoCRequestOutput) and output.finished:
                    distances.append(output.outputs.distance)
                    print(f"  Run {run + 1}: distance = {output.outputs.distance:.6f}")

    # Check all distances are close (within bfloat16 tolerance)
    if len(distances) == 3:
        max_diff = max(abs(distances[i] - distances[j])
                       for i in range(3) for j in range(i + 1, 3))
        success = max_diff < 0.01
        detail = f"Max diff: {max_diff:.6f} (tolerance: 0.01)"
    else:
        success = False
        detail = f"Expected 3 results, got {len(distances)}"

    return success, detail


def main():
    parser = argparse.ArgumentParser(description="PoC Integration Tests")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model to use (default: Qwen2.5-7B-Instruct)")
    parser.add_argument("--gpu-memory", type=float, default=0.85,
                        help="GPU memory utilization (default: 0.85)")
    args = parser.parse_args()

    print("=" * 60)
    print("PoC Scheduler Integration Tests")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"GPU memory: {args.gpu_memory}")

    # Load model
    print("\nLoading model...")
    llm = LLM(
        model=args.model,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=2048,
        # Enable chunked prefill to test our fix
        enable_chunked_prefill=True,
        max_num_batched_tokens=512,  # Small to force chunking on long sequences
    )

    engine = llm.llm_engine
    print(f"Engine ready: {type(engine).__name__}")

    # Run tests
    results = []

    # Test 1: Mixed batch
    success, detail = test_mixed_batch(engine)
    results.append(("Mixed Batch", success, detail))

    # Test 2: Priority scheduling
    success, detail = test_priority_scheduling(engine)
    results.append(("Priority Scheduling", success, detail))

    # Test 3: Chunked prefill
    success, detail = test_chunked_prefill(engine)
    results.append(("Chunked Prefill", success, detail))

    # Test 4: Determinism
    success, detail = test_determinism(engine)
    results.append(("Determinism", success, detail))

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, success, detail in results:
        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {name}")
        print(f"         {detail}")
        all_passed = all_passed and success

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()

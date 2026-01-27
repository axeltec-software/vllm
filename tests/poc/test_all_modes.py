#!/usr/bin/env python3
"""Test all PoC modes: pure chat, pure PoC, and mixed batch.

Usage:
    # Start vLLM server first:
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen3-8B-FP8 \
        --port 5002 \
        --gpu-memory-utilization 0.5 \
        --max-model-len 4096

    # Run tests:
    python tests/poc/test_all_modes.py

    # Or with custom server URL:
    python tests/poc/test_all_modes.py --server http://localhost:5002
"""

import argparse
import concurrent.futures
import json
import sys
import time
from typing import Optional

import requests


def test_pure_chat(server_url: str) -> dict:
    """Test pure chat request (no PoC)."""
    print("\n" + "=" * 60)
    print("TEST 1: PURE CHAT")
    print("=" * 60)

    start = time.time()
    response = requests.post(
        f"{server_url}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": "Qwen/Qwen3-8B-FP8",
            "messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}],
            "max_tokens": 20,
        },
        timeout=30,
    )
    elapsed = time.time() - start

    result = {
        "test": "pure_chat",
        "status_code": response.status_code,
        "elapsed": elapsed,
        "success": False,
    }

    if response.status_code == 200:
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        result["content"] = content
        result["success"] = len(content) > 0
        print(f"  Status: {response.status_code}")
        print(f"  Response: {content[:100]}")
        print(f"  Time: {elapsed:.3f}s")
    else:
        print(f"  ERROR: {response.status_code}")
        print(f"  Response: {response.text[:200]}")

    return result


def test_pure_poc(server_url: str) -> dict:
    """Test pure PoC request (no chat)."""
    print("\n" + "=" * 60)
    print("TEST 2: PURE POC")
    print("=" * 60)

    start = time.time()
    response = requests.post(
        f"{server_url}/api/v1/pow/generate",
        headers={"Content-Type": "application/json"},
        json={
            "block_hash": "0xtest_pure_poc_hash",
            "public_key": "0xtest_public_key",
            "nonces": [1, 2, 3, 4, 5],
            "block_height": 1000,
            "r_target": 2.0,  # High threshold to get valid results
            "wait": True,
        },
        timeout=60,
    )
    elapsed = time.time() - start

    result = {
        "test": "pure_poc",
        "status_code": response.status_code,
        "elapsed": elapsed,
        "success": False,
    }

    if response.status_code == 200:
        data = response.json()
        result["total_checked"] = data.get("total_checked", 0)
        result["total_valid"] = data.get("total_valid", 0)
        result["distances"] = data.get("valid_distances", [])
        result["success"] = data.get("status") == "completed" and result["total_checked"] > 0

        print(f"  Status: {data.get('status')}")
        print(f"  Checked: {result['total_checked']} nonces")
        print(f"  Valid: {result['total_valid']} nonces")
        if result["distances"]:
            print(f"  Distances: {[f'{d:.4f}' for d in result['distances'][:3]]}...")
        print(f"  Time: {elapsed:.3f}s")
    else:
        print(f"  ERROR: {response.status_code}")
        print(f"  Response: {response.text[:200]}")

    return result


def test_mixed_batch(server_url: str) -> dict:
    """Test mixed batch (concurrent chat + PoC)."""
    print("\n" + "=" * 60)
    print("TEST 3: MIXED BATCH (concurrent chat + PoC)")
    print("=" * 60)

    def send_chat():
        return requests.post(
            f"{server_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": "Qwen/Qwen3-8B-FP8",
                "messages": [{"role": "user", "content": "Count from 1 to 10."}],
                "max_tokens": 50,
            },
            timeout=60,
        )

    def send_poc():
        return requests.post(
            f"{server_url}/api/v1/pow/generate",
            headers={"Content-Type": "application/json"},
            json={
                "block_hash": "0xtest_mixed_batch_hash",
                "public_key": "0xtest_mixed_key",
                "nonces": list(range(10, 20)),  # 10 nonces
                "block_height": 2000,
                "r_target": 2.0,
                "wait": True,
            },
            timeout=60,
        )

    start = time.time()

    # Run both requests concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        chat_future = executor.submit(send_chat)
        poc_future = executor.submit(send_poc)

        chat_response = chat_future.result()
        poc_response = poc_future.result()

    elapsed = time.time() - start

    result = {
        "test": "mixed_batch",
        "elapsed": elapsed,
        "chat_success": False,
        "poc_success": False,
        "success": False,
    }

    # Check chat result
    if chat_response.status_code == 200:
        chat_data = chat_response.json()
        content = chat_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        result["chat_content"] = content
        result["chat_success"] = len(content) > 0
        print(f"  Chat Status: {chat_response.status_code}")
        print(f"  Chat Response: {content[:80]}...")
    else:
        print(f"  Chat ERROR: {chat_response.status_code}")

    # Check PoC result
    if poc_response.status_code == 200:
        poc_data = poc_response.json()
        result["poc_checked"] = poc_data.get("total_checked", 0)
        result["poc_valid"] = poc_data.get("total_valid", 0)
        result["poc_success"] = poc_data.get("status") == "completed"
        print(f"  PoC Status: {poc_data.get('status')}")
        print(f"  PoC Checked: {result['poc_checked']} nonces")
    else:
        print(f"  PoC ERROR: {poc_response.status_code}")

    result["success"] = result["chat_success"] and result["poc_success"]
    print(f"  Total Time: {elapsed:.3f}s")

    return result


def test_hook_caching(server_url: str) -> dict:
    """Test that hooks are cached for same block_hash."""
    print("\n" + "=" * 60)
    print("TEST 4: HOOK CACHING (same block_hash, multiple requests)")
    print("=" * 60)

    block_hash = "0xtest_hook_caching_hash"
    times = []

    for i in range(3):
        start = time.time()
        response = requests.post(
            f"{server_url}/api/v1/pow/generate",
            headers={"Content-Type": "application/json"},
            json={
                "block_hash": block_hash,
                "public_key": "0xtest_key",
                "nonces": [i * 10 + j for j in range(5)],
                "block_height": 3000 + i,
                "r_target": 2.0,
                "wait": True,
            },
            timeout=60,
        )
        elapsed = time.time() - start
        times.append(elapsed)

        if response.status_code == 200:
            print(f"  Request {i + 1}: {elapsed:.3f}s")
        else:
            print(f"  Request {i + 1}: ERROR {response.status_code}")

    result = {
        "test": "hook_caching",
        "times": times,
        "success": len(times) == 3 and all(t < 5.0 for t in times),
    }

    # Second and third requests should be faster (hooks already registered)
    if len(times) >= 2:
        print(f"  First request: {times[0]:.3f}s (includes hook registration)")
        print(f"  Subsequent avg: {sum(times[1:]) / len(times[1:]):.3f}s (hooks cached)")

    return result


def test_different_block_hash(server_url: str) -> dict:
    """Test that hooks are recreated for different block_hash."""
    print("\n" + "=" * 60)
    print("TEST 5: DIFFERENT BLOCK_HASH (hooks recreated)")
    print("=" * 60)

    results = []
    for i, block_hash in enumerate(["0xhash_a", "0xhash_b", "0xhash_c"]):
        start = time.time()
        response = requests.post(
            f"{server_url}/api/v1/pow/generate",
            headers={"Content-Type": "application/json"},
            json={
                "block_hash": block_hash,
                "public_key": "0xtest_key",
                "nonces": [1, 2, 3],
                "block_height": 4000 + i,
                "r_target": 2.0,
                "wait": True,
            },
            timeout=60,
        )
        elapsed = time.time() - start

        if response.status_code == 200:
            data = response.json()
            results.append({
                "block_hash": block_hash,
                "elapsed": elapsed,
                "distances": data.get("valid_distances", []),
            })
            print(f"  {block_hash}: {elapsed:.3f}s, distances={[f'{d:.3f}' for d in data.get('valid_distances', [])[:2]]}")
        else:
            print(f"  {block_hash}: ERROR {response.status_code}")

    # Different block_hash should produce different distances
    result = {
        "test": "different_block_hash",
        "results": results,
        "success": len(results) == 3,
    }

    if len(results) >= 2:
        d1 = results[0].get("distances", [0])[0] if results[0].get("distances") else 0
        d2 = results[1].get("distances", [0])[0] if results[1].get("distances") else 0
        result["distances_differ"] = abs(d1 - d2) > 0.01
        print(f"  Distances differ: {result['distances_differ']} (expected: True)")

    return result


def test_validate(server_url: str) -> dict:
    """Test PoC validate endpoint."""
    print("\n" + "=" * 60)
    print("TEST 6: VALIDATE (correct distances)")
    print("=" * 60)

    # First generate to get valid distances
    gen_response = requests.post(
        f"{server_url}/api/v1/pow/generate",
        headers={"Content-Type": "application/json"},
        json={
            "block_hash": "0xvalidate_test",
            "public_key": "0xvalidate_key",
            "nonces": [100, 101],
            "block_height": 6000,
            "r_target": 2.0,
            "wait": True,
        },
        timeout=60,
    )

    result = {
        "test": "validate",
        "success": False,
    }

    if gen_response.status_code != 200:
        print(f"  Generate ERROR: {gen_response.status_code}")
        return result

    gen_data = gen_response.json()
    distances = gen_data.get("valid_distances", [])
    print(f"  Generated distances: {[f'{d:.4f}' for d in distances]}")

    # Validate with correct distances
    val_response = requests.post(
        f"{server_url}/api/v1/pow/validate",
        headers={"Content-Type": "application/json"},
        json={
            "block_hash": "0xvalidate_test",
            "public_key": "0xvalidate_key",
            "nonces": [100, 101],
            "block_height": 6000,
            "r_target": 2.0,
            "dist": distances,
            "node_id": 1,
        },
        timeout=60,
    )

    if val_response.status_code == 200:
        val_data = val_response.json()
        fraud = val_data.get("fraud_detected", True)
        computed = val_data.get("computed_distances", [])
        print(f"  Computed distances: {[f'{d:.4f}' for d in computed]}")
        print(f"  Fraud detected: {fraud} (expected: False)")
        result["success"] = not fraud
    else:
        print(f"  Validate ERROR: {val_response.status_code}")

    return result


def test_high_concurrency(server_url: str, num_chat: int = 10, num_poc: int = 10) -> dict:
    """Test high concurrency with many concurrent chat + PoC requests."""
    print("\n" + "=" * 60)
    print(f"TEST 7: HIGH CONCURRENCY ({num_chat} chat + {num_poc} PoC requests)")
    print("=" * 60)

    def send_chat_numbered(idx: int):
        """Send a chat request with a unique question."""
        try:
            response = requests.post(
                f"{server_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": "Qwen/Qwen3-8B-FP8",
                    "messages": [{"role": "user", "content": f"What is {idx} + {idx}? Answer briefly."}],
                    "max_tokens": 150,
                },
                timeout=60,
            )
            return ("chat", idx, response)
        except Exception as e:
            return ("chat", idx, None, str(e))

    def send_poc_numbered(idx: int):
        """Send a PoC request with unique nonces."""
        try:
            response = requests.post(
                f"{server_url}/api/v1/pow/generate",
                headers={"Content-Type": "application/json"},
                json={
                    "block_hash": f"0xhigh_concurrency_{idx}",
                    "public_key": f"0xpubkey_{idx}",
                    "nonces": [idx * 100 + j for j in range(5)],
                    "block_height": 5000 + idx,
                    "r_target": 2.0,
                    "wait": True,
                },
                timeout=60,
            )
            return ("poc", idx, response)
        except Exception as e:
            return ("poc", idx, None, str(e))

    start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_chat + num_poc) as executor:
        futures = []

        for i in range(num_chat):
            futures.append(executor.submit(send_chat_numbered, i))

        for i in range(num_poc):
            futures.append(executor.submit(send_poc_numbered, i))

        results_list = [future.result() for future in concurrent.futures.as_completed(futures)]

    elapsed = time.time() - start

    chat_results = []
    poc_results = []
    
    for result in results_list:
        if result[0] == "chat":
            _, idx, response, *error = result
            if response and response.status_code == 200:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                chat_results.append({"idx": idx, "success": True, "content": content})
                print(f"  Chat #{idx}: {content[:80]}")
            else:
                error_msg = error[0] if error else (response.text[:100] if response else "Unknown error")
                chat_results.append({"idx": idx, "success": False, "error": error_msg})
                print(f"  Chat #{idx}: ERROR - {error_msg}")
        
        elif result[0] == "poc":
            _, idx, response, *error = result
            if response and response.status_code == 200:
                data = response.json()
                checked = data.get("total_checked", 0)
                valid = data.get("total_valid", 0)
                distances = data.get("valid_distances", [])
                poc_results.append({
                    "idx": idx,
                    "success": True,
                    "checked": checked,
                    "valid": valid,
                    "distances": distances
                })
                dist_str = ", ".join([f"{d:.4f}" for d in distances[:3]])
                print(f"  PoC #{idx}: checked={checked}, valid={valid}, distances=[{dist_str}]")
            else:
                error_msg = error[0] if error else (response.text[:100] if response else "Unknown error")
                poc_results.append({"idx": idx, "success": False, "error": error_msg})
                print(f"  PoC #{idx}: ERROR - {error_msg}")

    chat_success = sum(1 for r in chat_results if r.get("success"))
    poc_success = sum(1 for r in poc_results if r.get("success"))
    
    result = {
        "test": "high_concurrency",
        "elapsed": elapsed,
        "chat_sent": num_chat,
        "chat_success": chat_success,
        "poc_sent": num_poc,
        "poc_success": poc_success,
        "success": chat_success == num_chat and poc_success == num_poc,
    }

    print(f"\n  Total Time: {elapsed:.3f}s")
    print(f"  Chat: {chat_success}/{num_chat} successful")
    print(f"  PoC: {poc_success}/{num_poc} successful")
    return result


def test_validate_fraud(server_url: str) -> dict:
    """Test PoC validate detects fraud with wrong distances."""
    print("\n" + "=" * 60)
    print("TEST 8: VALIDATE FRAUD DETECTION (wrong distances)")
    print("=" * 60)

    # Validate with obviously wrong distances
    response = requests.post(
        f"{server_url}/api/v1/pow/validate",
        headers={"Content-Type": "application/json"},
        json={
            "block_hash": "0xvalidate_test",
            "public_key": "0xvalidate_key",
            "nonces": [100],
            "block_height": 6000,
            "r_target": 2.0,
            "dist": [0.001],  # Wrong distance
            "node_id": 1,
        },
        timeout=60,
    )

    result = {
        "test": "validate_fraud",
        "success": False,
    }

    if response.status_code == 200:
        data = response.json()
        fraud = data.get("fraud_detected", False)
        computed = data.get("computed_distances", [])
        print(f"  Provided distance: 0.001")
        print(f"  Computed distance: {computed[0]:.4f}" if computed else "  No computed distance")
        print(f"  Fraud detected: {fraud} (expected: True)")
        result["success"] = fraud
    else:
        print(f"  Validate ERROR: {response.status_code}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Test all PoC modes")
    parser.add_argument(
        "--server",
        default="http://localhost:5002",
        help="vLLM server URL (default: http://localhost:5002)",
    )
    args = parser.parse_args()

    server_url = args.server.rstrip("/")

    print("=" * 60)
    print("POC MODE TESTS")
    print(f"Server: {server_url}")
    print("=" * 60)

    # Check server health
    try:
        health = requests.get(f"{server_url}/health", timeout=5)
        if health.status_code != 200:
            print(f"ERROR: Server not healthy (status {health.status_code})")
            sys.exit(1)
        print("Server: OK")
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to server at {server_url}")
        print("Start the server first:")
        print("  python -m vllm.entrypoints.openai.api_server \\")
        print("    --model Qwen/Qwen3-8B-FP8 --port 5002")
        sys.exit(1)

    # Run all tests
    results = []
    # results.append(test_pure_chat(server_url))
    # results.append(test_pure_poc(server_url))
    # results.append(test_mixed_batch(server_url))
    # results.append(test_hook_caching(server_url))
    # results.append(test_different_block_hash(server_url))
    # results.append(test_validate(server_url))
    results.append(test_high_concurrency(server_url, num_chat=10, num_poc=10))
    # results.append(test_validate_fraud(server_url))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        if not r["success"]:
            all_passed = False
        print(f"  {r['test']}: {status}")

    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()

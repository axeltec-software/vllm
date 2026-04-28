#!/usr/bin/env python3
"""Test all PoC modes: pure chat, pure PoC, and mixed batch.

Usage:
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2-1.5B-Instruct \
        --port 8102 \
        --gpu-memory-utilization 0.5 \
        --max-model-len 4096

    python tests/poc/test_all_modes.py
    python tests/poc/test_all_modes.py --server http://0.0.0.0:8102 --model Qwen/Qwen2-1.5B-Instruct
"""

import argparse
import base64
import concurrent.futures
import random
import struct
import sys
import time

import requests

CHAT_PROMPTS = [
    "Explain how a {thing} works in 2-3 sentences.",
    "Write a short poem about {thing}.",
    "What are 3 interesting facts about {thing}?",
    "Compare and contrast {thing} with {thing2}.",
    "Describe {thing} to a 5-year-old.",
    "What would happen if {thing} didn't exist?",
    "Write a haiku about {thing}.",
    "Tell me a joke involving {thing}.",
]

THINGS = [
    "quantum computing", "black holes", "sourdough bread", "the Roman Empire",
    "electric cars", "photosynthesis", "jazz music", "neural networks",
    "volcanoes", "origami", "the stock market", "espresso",
    "coral reefs", "satellites", "penguins", "compilers",
]


def _random_chat_prompt() -> str:
    template = random.choice(CHAT_PROMPTS)
    return template.format(thing=random.choice(THINGS), thing2=random.choice(THINGS))


def _poc_request_body(block_hash, nonces, model, public_key="0xtest_key",
                      block_height=100, seq_len=256, k_dim=12):
    """Build a PoC /generate request matching the reference API format."""
    return {
        "block_hash": block_hash,
        "block_height": block_height,
        "public_key": public_key,
        "node_id": 0,
        "node_count": 1,
        "nonces": nonces,
        "params": {"model": model, "seq_len": seq_len, "k_dim": k_dim},
        "batch_size": 32,
        "wait": True,
    }


def _decode_b64_vector(b64: str, k_dim: int = 12):
    """Decode base64 FP16 LE vector to list of floats."""
    raw = base64.b64decode(b64)
    count = len(raw) // 2
    values = struct.unpack(f'<{count}e', raw)
    return list(values)


def _check_artifact(artifact: dict, k_dim: int = 12) -> bool:
    """Validate artifact has correct format."""
    if "nonce" not in artifact or "vector_b64" not in artifact:
        return False
    try:
        vec = _decode_b64_vector(artifact["vector_b64"], k_dim)
        return len(vec) == k_dim
    except Exception:
        return False


def test_pure_chat(server_url: str, model: str) -> dict:
    print("\n" + "=" * 60)
    print("TEST 1: PURE CHAT")
    print("=" * 60)

    start = time.time()
    response = requests.post(
        f"{server_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}],
            "max_tokens": 20,
        },
        timeout=30,
    )
    elapsed = time.time() - start

    result = {"test": "pure_chat", "status_code": response.status_code, "elapsed": elapsed, "success": False}

    if response.status_code == 200:
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        result["content"] = content
        result["success"] = len(content) > 0
        print(f"  Response: {content[:100]}")
        print(f"  Time: {elapsed:.3f}s")
    else:
        print(f"  ERROR: {response.status_code} {response.text[:200]}")

    return result


def test_pure_poc(server_url: str, model: str) -> dict:
    print("\n" + "=" * 60)
    print("TEST 2: PURE POC")
    print("=" * 60)

    start = time.time()
    body = _poc_request_body("0xtest_pure_poc", [1, 2, 3, 4, 5], model)
    response = requests.post(
        f"{server_url}/api/v1/pow/generate",
        json=body,
        timeout=60,
    )
    elapsed = time.time() - start

    result = {"test": "pure_poc", "status_code": response.status_code, "elapsed": elapsed, "success": False}

    if response.status_code == 200:
        data = response.json()
        artifacts = data.get("artifacts", [])
        encoding = data.get("encoding", {})
        k_dim = encoding.get("k_dim", 12)

        valid_artifacts = [a for a in artifacts if _check_artifact(a, k_dim)]
        result["num_artifacts"] = len(artifacts)
        result["num_valid"] = len(valid_artifacts)
        result["success"] = data.get("status") == "completed" and len(valid_artifacts) == 5

        print(f"  Status: {data.get('status')}")
        print(f"  Artifacts: {len(artifacts)} (valid: {len(valid_artifacts)})")
        print(f"  Encoding: {encoding}")
        if artifacts:
            vec = _decode_b64_vector(artifacts[0]["vector_b64"], k_dim)
            print(f"  Sample vector (nonce={artifacts[0]['nonce']}): [{', '.join(f'{v:.4f}' for v in vec[:4])}...]")
        print(f"  Time: {elapsed:.3f}s")
    else:
        print(f"  ERROR: {response.status_code} {response.text[:200]}")

    return result


def test_mixed_batch(server_url: str, model: str) -> dict:
    print("\n" + "=" * 60)
    print("TEST 3: MIXED BATCH (concurrent chat + PoC)")
    print("=" * 60)

    def send_chat():
        return requests.post(
            f"{server_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Count from 1 to 10."}],
                "max_tokens": 50,
            },
            timeout=60,
        )

    def send_poc():
        body = _poc_request_body("0xtest_mixed", list(range(10, 20)), model)
        return requests.post(
            f"{server_url}/api/v1/pow/generate",
            json=body,
            timeout=60,
        )

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        chat_future = executor.submit(send_chat)
        poc_future = executor.submit(send_poc)
        chat_response = chat_future.result()
        poc_response = poc_future.result()
    elapsed = time.time() - start

    result = {"test": "mixed_batch", "elapsed": elapsed, "chat_success": False, "poc_success": False, "success": False}

    if chat_response.status_code == 200:
        content = chat_response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        result["chat_success"] = len(content) > 0
        print(f"  Chat: {content[:80]}")
    else:
        print(f"  Chat ERROR: {chat_response.status_code}")

    if poc_response.status_code == 200:
        poc_data = poc_response.json()
        artifacts = poc_data.get("artifacts", [])
        result["poc_success"] = poc_data.get("status") == "completed" and len(artifacts) > 0
        print(f"  PoC: {len(artifacts)} artifacts")
    else:
        print(f"  PoC ERROR: {poc_response.status_code}")

    result["success"] = result["chat_success"] and result["poc_success"]
    print(f"  Time: {elapsed:.3f}s")
    return result


def test_hook_caching(server_url: str, model: str) -> dict:
    print("\n" + "=" * 60)
    print("TEST 4: HOOK CACHING (same block_hash)")
    print("=" * 60)

    block_hash = "0xtest_hook_caching"
    times = []

    for i in range(3):
        start = time.time()
        body = _poc_request_body(block_hash, [i * 10 + j for j in range(5)], model, block_height=3000 + i)
        response = requests.post(f"{server_url}/api/v1/pow/generate", json=body, timeout=60)
        elapsed = time.time() - start
        times.append(elapsed)
        status = f"{elapsed:.3f}s" if response.status_code == 200 else f"ERROR {response.status_code}"
        print(f"  Request {i + 1}: {status}")

    result = {"test": "hook_caching", "times": times, "success": len(times) == 3}
    if len(times) >= 2:
        print(f"  First: {times[0]:.3f}s  Subsequent avg: {sum(times[1:]) / len(times[1:]):.3f}s")
    return result


def test_different_block_hash(server_url: str, model: str) -> dict:
    print("\n" + "=" * 60)
    print("TEST 5: DIFFERENT BLOCK_HASH")
    print("=" * 60)

    results_list = []
    for i, bh in enumerate(["0xhash_a", "0xhash_b", "0xhash_c"]):
        start = time.time()
        body = _poc_request_body(bh, [1, 2, 3], model, block_height=4000 + i)
        response = requests.post(f"{server_url}/api/v1/pow/generate", json=body, timeout=60)
        elapsed = time.time() - start

        if response.status_code == 200:
            data = response.json()
            artifacts = data.get("artifacts", [])
            b64s = [a.get("vector_b64", "") for a in artifacts[:2]]
            results_list.append({"block_hash": bh, "elapsed": elapsed, "b64s": b64s})
            print(f"  {bh}: {elapsed:.3f}s, first_b64={b64s[0][:20]}..." if b64s else f"  {bh}: no artifacts")
        else:
            print(f"  {bh}: ERROR {response.status_code}")

    result = {"test": "different_block_hash", "results": results_list, "success": len(results_list) == 3}

    # Different block_hash should produce different vectors
    if len(results_list) >= 2 and results_list[0].get("b64s") and results_list[1].get("b64s"):
        differ = results_list[0]["b64s"][0] != results_list[1]["b64s"][0]
        result["vectors_differ"] = differ
        print(f"  Vectors differ: {differ} (expected: True)")

    return result


def test_high_concurrency(server_url: str, model: str, num_chat: int = 10, num_poc: int = 10) -> dict:
    print("\n" + "=" * 60)
    print(f"TEST 6: HIGH CONCURRENCY ({num_chat} chat + {num_poc} PoC)")
    print("=" * 60)

    def send_chat(idx):
        try:
            r = requests.post(
                f"{server_url}/v1/chat/completions",
                json={"model": model, "messages": [{"role": "user", "content": _random_chat_prompt()}], "max_tokens": 150},
                timeout=60,
            )
            return ("chat", idx, r)
        except Exception as e:
            return ("chat", idx, None, str(e))

    def send_poc(idx):
        try:
            body = _poc_request_body(f"0xconcurrency_{idx}", [idx * 100 + j for j in range(5)], model, block_height=5000 + idx)
            r = requests.post(f"{server_url}/api/v1/pow/generate", json=body, timeout=60)
            return ("poc", idx, r)
        except Exception as e:
            return ("poc", idx, None, str(e))

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_chat + num_poc) as executor:
        futures = [executor.submit(send_chat, i) for i in range(num_chat)]
        futures += [executor.submit(send_poc, i) for i in range(num_poc)]
        raw_results = [f.result() for f in concurrent.futures.as_completed(futures)]
    elapsed = time.time() - start

    chat_ok = poc_ok = 0
    for res in raw_results:
        kind, idx = res[0], res[1]
        resp = res[2] if len(res) > 2 else None
        if resp and resp.status_code == 200:
            if kind == "chat":
                chat_ok += 1
            else:
                poc_ok += 1

    print(f"  Chat: {chat_ok}/{num_chat}  PoC: {poc_ok}/{num_poc}  Time: {elapsed:.1f}s")
    return {
        "test": "high_concurrency", "elapsed": elapsed,
        "chat_success": chat_ok, "chat_sent": num_chat,
        "poc_success": poc_ok, "poc_sent": num_poc,
        "success": chat_ok == num_chat and poc_ok == num_poc,
    }


def main():
    parser = argparse.ArgumentParser(description="Test all PoC modes")
    parser.add_argument("--server", default="http://0.0.0.0:8102")
    parser.add_argument("--model", default="Qwen/Qwen2-1.5B-Instruct")
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--num-chat", type=int, default=10)
    parser.add_argument("--num-poc", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()

    server_url = args.server.rstrip("/")
    model = args.model

    print(f"Server: {server_url}  Model: {model}")

    try:
        health = requests.get(f"{server_url}/health", timeout=5)
        if health.status_code != 200:
            print(f"ERROR: Server not healthy ({health.status_code})")
            sys.exit(1)
        print("Server: OK")
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to {server_url}")
        sys.exit(1)

    if args.stress:
        all_passed = True
        for rnd in range(1, args.rounds + 1):
            r = test_high_concurrency(server_url, model, args.num_chat, args.num_poc)
            if not r["success"]:
                all_passed = False
            print(f"  Round {rnd}: {'PASS' if r['success'] else 'FAIL'}")
        sys.exit(0 if all_passed else 1)

    results = [
        test_pure_chat(server_url, model),
        test_pure_poc(server_url, model),
        test_mixed_batch(server_url, model),
        test_hook_caching(server_url, model),
        test_different_block_hash(server_url, model),
        test_high_concurrency(server_url, model, args.num_chat, args.num_poc),
    ]

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
    print("ALL PASSED" if all_passed else "SOME FAILED")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

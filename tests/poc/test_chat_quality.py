#!/usr/bin/env python3
"""Test that chat outputs are not corrupted when mixed with PoC requests.

Usage:
    python tests/poc/test_chat_quality.py --server http://localhost:8102 --model Qwen/Qwen2.5-7B-Instruct
    python tests/poc/test_chat_quality.py --server http://localhost:8102 --model Qwen/Qwen2.5-7B-Instruct --num-tests 40
"""

import argparse
import concurrent.futures
import requests


def test_chat_quality(server_url: str, model: str, num_tests: int = 20):
    """Test chat output quality with concurrent PoC requests."""

    def send_chat(idx: int):
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": f"What is {idx} plus {idx}?"}],
                "max_tokens": 150,
                "temperature": 0.5,
            },
            timeout=60,
        )
        if response.status_code == 200:
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return (idx, content, True)
        return (idx, response.text, False)

    def send_poc(idx: int):
        response = requests.post(
            f"{server_url}/api/v1/pow/generate",
            headers={"Content-Type": "application/json"},
            json={
                "block_hash": f"0xtest_quality_{idx}",
                "public_key": f"0xkey_{idx}",
                "node_id": 0,
                "node_count": 1,
                "nonces": [idx * 10 + j for j in range(5)],
                "block_height": 1000 + idx,
                "params": {"model": model, "seq_len": 256, "k_dim": 12},
                "wait": True,
            },
            timeout=60,
        )
        return response.status_code == 200

    print(f"\nTesting chat output quality with {num_tests} chat + {num_tests} PoC requests...")
    print("=" * 80)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_tests * 2) as executor:
        chat_futures = [executor.submit(send_chat, i) for i in range(num_tests)]
        poc_futures = [executor.submit(send_poc, i) for i in range(num_tests)]

        chat_results = [f.result() for f in chat_futures]
        poc_results = [f.result() for f in poc_futures]

    print("\nChat Outputs:")
    print("-" * 80)

    corrupted = []
    valid = []

    for idx, content, success in sorted(chat_results, key=lambda x: x[0]):
        if not success:
            print(f"  #{idx}: ERROR - {content[:100]}")
            corrupted.append(idx)
            continue

        is_corrupted = False
        corruption_signs = []

        if "OkayOkay" in content or content.count("Okay") > 5:
            is_corrupted = True
            corruption_signs.append("repeated 'Okay'")

        if content.count("[]") > 2 or content.count("()") > 3:
            is_corrupted = True
            corruption_signs.append("excess brackets")

        if "https" in content and "://" in content:
            is_corrupted = True
            corruption_signs.append("random URLs")

        if len(content.strip()) < 5:
            is_corrupted = True
            corruption_signs.append("too short")

        if content.count("\n") > 10:
            is_corrupted = True
            corruption_signs.append("excess newlines")

        alphanum_ratio = sum(c.isalnum() or c.isspace() for c in content) / max(len(content), 1)
        if alphanum_ratio < 0.7:
            is_corrupted = True
            corruption_signs.append(f"garbled ({alphanum_ratio:.1%} readable)")

        if is_corrupted:
            corrupted.append(idx)
            print(f"  #{idx}: CORRUPTED ({', '.join(corruption_signs)})")
            print(f"         Content: {content[:100]}")
        else:
            valid.append(idx)
            expected_result = idx + idx
            if str(expected_result) in content or str(expected_result) in content.replace(" ", ""):
                print(f"  #{idx}: OK - contains result {expected_result}")
            else:
                print(f"  #{idx}: OK (but unexpected content)")
                print(f"         Content: {content[:80]}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Total chat requests: {num_tests}")
    print(f"  Valid outputs: {len(valid)}")
    print(f"  Corrupted outputs: {len(corrupted)}")
    if corrupted:
        print(f"  Corrupted indices: {corrupted}")
    print(f"  PoC requests successful: {sum(poc_results)}/{num_tests}")
    print("=" * 80)

    if corrupted:
        print(f"\nFAILED: {len(corrupted)} chat outputs were corrupted")
        return False
    else:
        print(f"\nPASSED: All chat outputs are valid")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://localhost:8102")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num-tests", type=int, default=20)
    args = parser.parse_args()

    try:
        health = requests.get(f"{args.server}/health", timeout=5)
        if health.status_code != 200:
            print(f"ERROR: Server not healthy")
            return 1
    except Exception as e:
        print(f"ERROR: Cannot connect to server: {e}")
        return 1

    success = test_chat_quality(args.server, args.model, args.num_tests)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

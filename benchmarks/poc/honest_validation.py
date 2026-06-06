#!/usr/bin/env python3
"""
Simple honest-validation test.

Sends identical PoC generation requests to two vLLM servers, decodes the
returned artifact vectors, computes per-nonce L2 distances, and prints a
summary table.

Both servers must already be running. No Docker is started here.

Example:
    python benchmarks/poc/honest_validation.py \
        --server1 http://127.0.0.1:8000 \
        --server2 http://127.0.0.1:8001 \
        --nonces 20 \
        --model RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16
"""

import argparse
import base64

import numpy as np
import requests

API_PREFIX = "/api/v1/pow"

def api_post(url: str, endpoint: str, payload: dict, timeout: int = 600) -> dict:
    full_url = f"{url}{endpoint}"
    r = requests.post(full_url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def stop_server(url: str) -> None:
    try:
        requests.post(f"{url}{API_PREFIX}/stop", timeout=10)
    except Exception:
        pass


def decode_vector(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    return np.frombuffer(raw, dtype="<f2").astype(np.float32)


def generate_artifacts(url: str, nonces: list[int], args: argparse.Namespace) -> dict[int, np.ndarray]:
    stop_server(url)

    payload = {
        "block_hash": args.block_hash,
        "block_height": args.block_height,
        "public_key": args.public_key,
        "node_id": 0,
        "node_count": 1,
        "nonces": nonces,
        "params": {
            "model": args.model,
            "seq_len": args.seq_len,
            "k_dim": args.k_dim,
        },
        "batch_size": args.batch_size,
        "wait": True,
    }

    result = api_post(url, f"{API_PREFIX}/generate", payload, timeout=args.timeout)

    vectors: dict[int, np.ndarray] = {}
    for artifact in result.get("artifacts", []):
        nonce = artifact["nonce"]
        vectors[nonce] = decode_vector(artifact["vector_b64"])
    return vectors


def print_table(rows: list[tuple], headers: list[str], col_widths: list[int]) -> None:
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "  ".join("-" * w for w in col_widths)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare PoC artifacts from two vLLM servers and print L2 distance table."
    )
    parser.add_argument("--server1", required=True, help="URL of first server (e.g. http://127.0.0.1:8000)")
    parser.add_argument("--server2", required=True, help="URL of second server (e.g. http://127.0.0.1:8001)")
    parser.add_argument("--nonces", type=int, default=10, help="Number of nonces to request (default: 10)")
    parser.add_argument("--model", default="RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16", help="Model name")
    parser.add_argument("--seq-len", type=int, default=256, dest="seq_len", help="Sequence length (default: 256)")
    parser.add_argument("--k-dim", type=int, default=12, dest="k_dim", help="K dimension (default: 12)")
    parser.add_argument("--batch-size", type=int, default=16, dest="batch_size", help="Batch size (default: 16)")
    parser.add_argument("--block-hash", default="TEST_BLOCK", dest="block_hash", help="Block hash seed")
    parser.add_argument("--public-key", default="test_pub_key", dest="public_key", help="Public key seed")
    parser.add_argument("--block-height", type=int, default=100, dest="block_height", help="Block height")
    parser.add_argument("--timeout", type=int, default=600, help="Request timeout in seconds (default: 600)")
    args = parser.parse_args()

    nonces = list(range(args.nonces))

    print(f"[1/2] Collecting from {args.server1} ...")
    vecs1 = generate_artifacts(args.server1, nonces, args)
    print(f"      Got {len(vecs1)} artifacts.")

    vecs2 = generate_artifacts(args.server2, nonces, args)
    print(f"      Got {len(vecs2)} artifacts.")
    print()

    common = sorted(set(vecs1) & set(vecs2))
    if not common:
        print("ERROR: No common nonces returned by both servers.")
        return

    dists = np.array([np.linalg.norm(vecs1[n] - vecs2[n]) for n in common])

    rows = [(n, f"{dists[i]:.6f}") for i, n in enumerate(common)]
    print_table(rows, ["Nonce", "L2 Distance"], [10, 14])
    print()

    summary_rows = [
        ("mean",   f"{dists.mean():.6f}"),
        ("std",    f"{dists.std():.6f}"),
        ("min",    f"{dists.min():.6f}"),
        ("max",    f"{dists.max():.6f}"),
        ("median", f"{np.median(dists):.6f}"),
        ("count",  str(len(dists))),
    ]
    print("Summary:")
    print_table(summary_rows, ["Metric", "Value"], [10, 14])

    only_in_1 = set(vecs1) - set(vecs2)
    only_in_2 = set(vecs2) - set(vecs1)
    if only_in_1:
        print(f"\nNonces only in server1 (missing in server2): {sorted(only_in_1)}")
    if only_in_2:
        print(f"\nNonces only in server2 (missing in server1): {sorted(only_in_2)}")


if __name__ == "__main__":
    main()

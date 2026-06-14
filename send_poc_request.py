"""Send PoC requests to a running vLLM server and print the results.

Usage examples
--------------
# Basic: 4 nonces, prefill-only
python send_poc_request.py --url http://localhost:8000 --nonces 0 1 2 3

# With decode steps (requires server started with --poc-decode)
python send_poc_request.py --url http://localhost:8000 --nonces 0 1 2 3 \\
    --max-tokens 5

# Custom block hash / public key
python send_poc_request.py --url http://localhost:8000 --nonces 0 1 2 3 \\
    --block-hash deadbeef1234 --public-key mypubkey \\
    --seq-len 128 --k-dim 16 --max-tokens 3

# Compact output (one line per artifact)
python send_poc_request.py --url http://localhost:8000 --nonces 0 1 2 3 --compact
"""

import argparse
import hashlib
import json3
import os
import sys
import time

import requests


def default_block_hash() -> str:
    return hashlib.sha256(b"poc_test_block").hexdigest()


def default_public_key() -> str:
    return "poc_test_pubkey_0000000000000000"


def send_request(
    url: str,
    block_hash: str,
    public_key: str,
    block_height: int,
    nonces: list[int],
    model: str,
    seq_len: int,
    k_dim: int,
    max_tokens: int,
    timeout: int,
) -> dict:
    payload = {
        "block_hash": block_hash,
        "block_height": block_height,
        "public_key": public_key,
        "node_id": 0,
        "node_count": 1,
        "nonces": nonces,
        "params": {
            "model": model,
            "seq_len": seq_len,
            "k_dim": k_dim
        },
        "max_tokens": max_tokens,
        "wait": True,
    }
    resp = requests.post(
        f"{url.rstrip('/')}/api/v1/pow/generate",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def print_artifact(art: dict, compact: bool, decode_mode: bool) -> None:
    nonce = art.get("nonce", "?")
    sphere_k = art.get("sphere_k", -1)
    sphere_k_steps = art.get("sphere_k_steps", [])
    vector_b64 = art.get("vector_b64", "")
    vec_preview = vector_b64[:24] + "…" if len(vector_b64) > 24 else vector_b64

    if compact:
        if decode_mode and sphere_k_steps:
            print(f"  nonce={nonce:>6}  sphere_k={sphere_k}  "
                  f"k_steps={sphere_k_steps}  vec={vec_preview}")
        else:
            print(f"  nonce={nonce:>6}  sphere_k={sphere_k}  vec={vec_preview}")
        return

    print(f"  nonce            : {nonce}")
    print(f"  sphere_k         : {sphere_k}")
    if decode_mode:
        if sphere_k_steps:
            labels = ["prefill"] + [f"decode{i}" for i in range(1, len(sphere_k_steps))]
            step_str = "  ->  ".join(
                f"{lbl}:{k}" for lbl, k in zip(labels, sphere_k_steps)
            )
            print(f"  sphere_k_steps   : [{step_str}]")
        else:
            print("  sphere_k_steps   : (empty – server may not have --poc-decode enabled)")
    print(f"  vector_b64       : {vec_preview}")
    has_full = art.get("hidden_state_b64")
    has_reduced = art.get("reduced_hidden_state_b64")
    if has_full:
        preview = has_full[:24] + "…"
        print(f"  hidden_state_b64 : {preview}")
    if has_reduced:
        preview = has_reduced[:24] + "…"
        print(f"  reduced_hs_b64   : {preview}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Send PoC requests and print results.")
    parser.add_argument("--url", default="http://localhost:8001",
                        help="vLLM server base URL (default: http://localhost:8001)")
    parser.add_argument("--model", default=None,
                        help="Model name (queried from /v1/models if omitted)")
    parser.add_argument("--block-hash", default=None,
                        help="64-char hex block hash (random default)")
    parser.add_argument("--public-key", default=None,
                        help="Public key string (default: poc_test_pubkey_…)")
    parser.add_argument("--block-height", type=int, default=1,
                        help="Block height (default: 1)")
    parser.add_argument("--nonces", type=int, nargs="+", default=[0, 1, 2, 3],
                        help="Nonce values to submit (default: 0 1 2 3)")
    parser.add_argument("--seq-len", type=int, default=254,
                        help="Prefill sequence length (default: 254)")
    parser.add_argument("--k-dim", type=int, default=12,
                        help="k-dim for Haar rotation (default: 12)")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Decode steps after prefill; 0 = prefill-only (default: 0)")
    parser.add_argument("--timeout", type=int, default=1200,
                        help="HTTP timeout in seconds (default: 120)")
    parser.add_argument("--compact", action="store_true",
                        help="One-line-per-artifact output")
    parser.add_argument("--json", action="store_true",
                        help="Dump raw JSON response and exit")
    args = parser.parse_args()

    url = args.url.rstrip("/")

    # Resolve model name
    model = args.model
    if model is None:
        try:
            r = requests.get(f"{url}/v1/models", timeout=10)
            r.raise_for_status()
            model = r.json()["data"][0]["id"]
        except Exception as exc:
            print(f"[error] Could not fetch model list from {url}/v1/models: {exc}",
                  file=sys.stderr)
            print("        Pass --model <name> explicitly.", file=sys.stderr)
            sys.exit(1)

    block_hash = args.block_hash or default_block_hash()
    public_key = args.public_key or default_public_key()
    decode_mode = args.max_tokens > 0

    print(f"Server      : {url}")
    print(f"Model       : {model}")
    print(f"Block hash  : {block_hash[:32]}…" if len(block_hash) > 32 else
          f"Block hash  : {block_hash}")
    print(f"Public key  : {public_key}")
    print(f"Nonces      : {args.nonces}")
    print(f"seq_len     : {args.seq_len}  k_dim={args.k_dim}  "
          f"max_tokens={args.max_tokens}")
    print(f"Decode mode : {'ON' if decode_mode else 'OFF (prefill-only)'}")
    print()

    t0 = time.perf_counter()
    try:
        result = send_request(
            url=url,
            block_hash=block_hash,
            public_key=public_key,
            block_height=args.block_height,
            nonces=args.nonces,
            model=model,
            seq_len=args.seq_len,
            k_dim=args.k_dim,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
    except requests.HTTPError as exc:
        print(f"[error] HTTP {exc.response.status_code}: {exc.response.text}",
              file=sys.stderr)
        sys.exit(1)
    except requests.ConnectionError:
        print(f"[error] Could not connect to {url}. Is the server running?",
              file=sys.stderr)
        sys.exit(1)

    elapsed = time.perf_counter() - t0

    if args.json:
        print(json.dumps(result, indent=2))
        return

    status = result.get("status", "?")
    artifacts = result.get("artifacts", [])

    print(f"Status      : {status}")
    print(f"Artifacts   : {len(artifacts)}")
    print(f"Elapsed     : {elapsed:.3f}s")
    print()

    if not artifacts:
        print("[warning] No artifacts returned.")
        return

    if args.compact:
        header = f"  {'nonce':>8}  sphere_k"
        if decode_mode:
            header += "  k_steps"
        header += "  vec"
        print(header)
        print("  " + "-" * 60)

    for art in artifacts:
        print_artifact(art, compact=args.compact, decode_mode=decode_mode)

    # Summary statistics when decode mode is on
    # if decode_mode:
    #     all_steps = [art.get("sphere_k_steps", []) for art in artifacts]
    #     all_steps = [s for s in all_steps if s]
    #     if all_steps:
    #         n_steps = len(all_steps[0])
    #         print(f"--- Decode statistics ({n_steps} steps, {len(all_steps)} nonces) ---")
    #         for step_idx in range(n_steps):
    #             label = "prefill" if step_idx == 0 else f"decode{step_idx}"
    #             ks = [s[step_idx] for s in all_steps if step_idx < len(s)]
    #             counts: dict[int, int] = {}
    #             for k in ks:
    #                 counts[k] = counts.get(k, 0) + 1
    #             dist_str = "  ".join(f"k{k}:{cnt}" for k, cnt in sorted(counts.items()))
    #             print(f"  {label:>10}: [{dist_str}]")


if __name__ == "__main__":
    main()

"""Cross-endpoint argmax comparator.

Sends the same (block_hash, nonce) pairs to two separate PoC server
endpoints and checks that argmax_idx agrees between them.

Results (argmax_idx + argmax_logit for each endpoint) are saved to a
JSON file for offline inspection.

Usage:
    python argmax_cross_endpoint.py
    python argmax_cross_endpoint.py --num-hashes 3 --nonces 0 1 2 3
    python argmax_cross_endpoint.py --url-a http://host-a:8001 \\
                                    --url-b http://host-b:8002 \\
                                    --nonces 0 42 --iterations 10
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import requests


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_URL_A      = "http://localhost:8001"
DEFAULT_URL_B      = "http://localhost:8003"
DEFAULT_BLOCK_HASH = "0" * 64
DEFAULT_PUBLIC_KEY = "0" * 64
DEFAULT_MODEL      = "Qwen/Qwen3-0.6B"
DEFAULT_SEQ_LEN    = 256
DEFAULT_K_DIM      = 12
DEFAULT_BATCH_SIZE = 32
DEFAULT_ITERATIONS = 1
DEFAULT_NONCES     = [0]
DEFAULT_OUTPUT     = "argmax_cross_endpoint.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_block_hashes(n: int) -> List[str]:
    return [os.urandom(32).hex() for _ in range(n)]


def fetch_artifacts(
    url: str,
    nonces: List[int],
    block_hash: str,
    public_key: str,
    model: str,
    seq_len: int,
    k_dim: int,
    batch_size: int,
) -> Dict[int, dict]:
    """Return {nonce: artifact_dict} for all requested nonces from one endpoint."""
    endpoint = f"{url.rstrip('/')}/api/v1/pow/generate"
    result: Dict[int, dict] = {}

    for start in range(0, len(nonces), batch_size):
        batch = nonces[start : start + batch_size]
        payload = {
            "block_hash": block_hash,
            "block_height": 0,
            "public_key": public_key,
            "node_id": 0,
            "node_count": 1,
            "nonces": batch,
            "params": {
                "model": model,
                "seq_len": seq_len,
                "k_dim": k_dim,
            },
            "batch_size": batch_size,
            "wait": True,
        }
        resp = requests.post(endpoint, json=payload, timeout=300)
        resp.raise_for_status()
        for art in resp.json().get("artifacts", []):
            result[art["nonce"]] = art

    return result


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(
    url_a: str,
    url_b: str,
    block_hashes: List[str],
    nonces: List[int],
    public_key: str,
    model: str,
    seq_len: int,
    k_dim: int,
    batch_size: int,
    iterations: int,
    output_path: str,
) -> bool:
    n_pairs = len(block_hashes) * len(nonces)
    print(f"Endpoint A : {url_a}")
    print(f"Endpoint B : {url_b}")
    print(f"Hashes     : {len(block_hashes)}")
    print(f"Nonces     : {nonces}")
    print(f"Pairs      : {n_pairs}")
    print(f"Iterations : {iterations}")
    print()

    # results[hash][nonce] = list of per-iteration dicts
    results: Dict[str, Dict[int, List[dict]]] = {
        h: {n: [] for n in nonces} for h in block_hashes
    }

    for it in range(1, iterations + 1):
        print(f"\r  Iteration {it:>{len(str(iterations))}}/{iterations} …", end="", flush=True)

        for h in block_hashes:
            arts_a = fetch_artifacts(
                url=url_a, nonces=nonces, block_hash=h,
                public_key=public_key, model=model,
                seq_len=seq_len, k_dim=k_dim, batch_size=batch_size,
            )
            arts_b = fetch_artifacts(
                url=url_b, nonces=nonces, block_hash=h,
                public_key=public_key, model=model,
                seq_len=seq_len, k_dim=k_dim, batch_size=batch_size,
            )
            for n in nonces:
                a = arts_a.get(n, {})
                b = arts_b.get(n, {})
                results[h][n].append({
                    "iteration": it,
                    "a": {
                        "argmax_idx":   a.get("argmax_idx", -1),
                        "argmax_logit": a.get("argmax_logit", None),
                    },
                    "b": {
                        "argmax_idx":   b.get("argmax_idx", -1),
                        "argmax_logit": b.get("argmax_logit", None),
                    },
                    "match": a.get("argmax_idx", -1) == b.get("argmax_idx", -1),
                })

    print("\n")

    # ── Save JSON ─────────────────────────────────────────────────────────
    serialisable = []
    for h in block_hashes:
        for n in nonces:
            serialisable.append({
                "block_hash": h,
                "nonce": n,
                "iterations": results[h][n],
            })

    with open(output_path, "w") as f:
        json.dump({
            "url_a": url_a,
            "url_b": url_b,
            "model": model,
            "seq_len": seq_len,
            "k_dim": k_dim,
            "entries": serialisable,
        }, f, indent=2)
    print(f"Results saved to {output_path}\n")

    # ── Report ────────────────────────────────────────────────────────────
    total_mismatches = 0
    header = f"{'hash':>14}  {'nonce':>6}  {'match':>8}  " \
             f"{'idx_a':>8}  {'logit_a':>10}  {'idx_b':>8}  {'logit_b':>10}"
    print(header)
    print("-" * len(header))

    for h in block_hashes:
        for n in nonces:
            iters = results[h][n]
            mismatches = [r for r in iters if not r["match"]]
            total_mismatches += len(mismatches)
            ok = len(mismatches) == 0

            # use last iteration values for the summary row
            last = iters[-1]
            la, lb = last["a"], last["b"]
            logit_a = f"{la['argmax_logit']:.5f}" if la["argmax_logit"] is not None else "n/a"
            logit_b = f"{lb['argmax_logit']:.5f}" if lb["argmax_logit"] is not None else "n/a"
            status  = "OK" if ok else f"FAIL({len(mismatches)})"

            print(f"  {h[:12]}  {n:>6}  {status:>8}  "
                  f"{la['argmax_idx']:>8}  {logit_a:>10}  "
                  f"{lb['argmax_idx']:>8}  {logit_b:>10}")

            for m in mismatches:
                ma, mb = m["a"], m["b"]
                la_str = f"{ma['argmax_logit']:.5f}" if ma["argmax_logit"] is not None else "n/a"
                lb_str = f"{mb['argmax_logit']:.5f}" if mb["argmax_logit"] is not None else "n/a"
                print(f"    iter {m['iteration']:>4}:  "
                      f"A idx={ma['argmax_idx']}  logit={la_str}  |  "
                      f"B idx={mb['argmax_idx']}  logit={lb_str}")

    print()
    if total_mismatches == 0:
        print(f"OK  All pairs match across both endpoints.")
        return True

    mismatch_pairs = sum(
        1 for h in block_hashes for n in nonces
        if any(not r["match"] for r in results[h][n])
    )
    print(f"FAIL  {total_mismatches} mismatch(es) in {mismatch_pairs} / {n_pairs} pair(s).")
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare argmax_idx between two PoC endpoints for the same inputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--url-a",      default=DEFAULT_URL_A,
                   help="First endpoint")
    p.add_argument("--url-b",      default=DEFAULT_URL_B,
                   help="Second endpoint")
    p.add_argument("--nonces",     type=int, nargs="+", default=DEFAULT_NONCES,
                   metavar="N", help="One or more nonces")
    p.add_argument("--block-hash", default=DEFAULT_BLOCK_HASH, metavar="HASH",
                   help="Fixed block hash (ignored when --num-hashes is set)")
    p.add_argument("--num-hashes", type=int, default=None, metavar="N",
                   help="Generate N random block hashes")
    p.add_argument("--public-key", default=DEFAULT_PUBLIC_KEY)
    p.add_argument("--model",      default=DEFAULT_MODEL)
    p.add_argument("--seq-len",    type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--k-dim",      type=int, default=DEFAULT_K_DIM)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS,
                   help="How many times to repeat each (hash, nonce) pair")
    p.add_argument("--output",     default=DEFAULT_OUTPUT,
                   help="Path for the JSON results file")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_hashes is not None:
        block_hashes = generate_block_hashes(args.num_hashes)
        print(f"Generated {args.num_hashes} random block hashes:")
        for h in block_hashes:
            print(f"  {h}")
        print()
    else:
        block_hashes = [args.block_hash]

    ok = compare(
        url_a=args.url_a,
        url_b=args.url_b,
        block_hashes=block_hashes,
        nonces=args.nonces,
        public_key=args.public_key,
        model=args.model,
        seq_len=args.seq_len,
        k_dim=args.k_dim,
        batch_size=args.batch_size,
        iterations=args.iterations,
        output_path=args.output,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

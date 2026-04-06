"""Argmax consistency checker.

Sends the same (block_hash, nonce) pairs to the PoC server N times and
verifies that argmax_idx is identical in every iteration for each pair.

On any mismatch the logit values for both the reference and the differing
iteration are printed so you can compare them.

Usage:
    python argmax_consistency.py
    python argmax_consistency.py --iterations 500 --nonces 0 1 2 3
    python argmax_consistency.py --num-hashes 3 --nonces 0 42 --iterations 200
    python argmax_consistency.py --block-hash <HASH> --nonces 0 1
"""
import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import requests


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_URL        = "http://localhost:8001"
DEFAULT_BLOCK_HASH = "0" * 64
DEFAULT_PUBLIC_KEY = "0" * 64
DEFAULT_MODEL      = "RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16"
DEFAULT_SEQ_LEN    = 256
DEFAULT_K_DIM      = 12
DEFAULT_BATCH_SIZE = 32
DEFAULT_ITERATIONS = 1000
DEFAULT_NONCES     = [0]


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
    """Return {nonce: artifact_dict} for all requested nonces."""
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
# Consistency check
# ---------------------------------------------------------------------------

# Key: (hash_short, nonce)
# Value: {"ref_idx": int, "ref_logit": float, "mismatches": list}
CheckState = Dict[Tuple[str, int], dict]


def run_check(
    url: str,
    block_hashes: List[str],
    nonces: List[int],
    public_key: str,
    model: str,
    seq_len: int,
    k_dim: int,
    batch_size: int,
    iterations: int,
) -> bool:
    n_pairs = len(block_hashes) * len(nonces)
    print(f"Hashes     : {len(block_hashes)}")
    print(f"Nonces     : {nonces}")
    print(f"Pairs      : {n_pairs}")
    print(f"Iterations : {iterations}")
    print(f"Total reqs : {n_pairs * iterations}")
    print()

    # State per (hash_short, nonce) pair
    state: CheckState = {}
    for h in block_hashes:
        for n in nonces:
            state[(h[:12], n)] = {"ref_idx": None, "ref_logit": None, "mismatches": []}

    for it in range(1, iterations + 1):
        print(f"\r  Iteration {it:>{len(str(iterations))}}/{iterations} …", end="", flush=True)

        for h in block_hashes:
            arts = fetch_artifacts(
                url=url,
                nonces=nonces,
                block_hash=h,
                public_key=public_key,
                model=model,
                seq_len=seq_len,
                k_dim=k_dim,
                batch_size=batch_size,
            )
            for n in nonces:
                art = arts.get(n, {})
                idx   = art.get("argmax_idx", -1)
                logit = art.get("argmax_logit", None)
                key   = (h[:12], n)
                s     = state[key]

                if s["ref_idx"] is None:
                    s["ref_idx"]   = idx
                    s["ref_logit"] = logit
                    continue

                if idx != s["ref_idx"]:
                    s["mismatches"].append({
                        "iteration": it,
                        "got_idx":   idx,
                        "got_logit": logit,
                    })

    print("\n")

    # ── Report ────────────────────────────────────────────────────────────
    total_mismatches = sum(len(s["mismatches"]) for s in state.values())

    if total_mismatches == 0:
        print(f"OK  All {iterations} iterations are consistent for all {n_pairs} (hash, nonce) pairs.\n")
        print(f"{'hash':>14}  {'nonce':>6}  {'argmax_idx':>10}  {'argmax_logit':>12}")
        print("-" * 50)
        for (h_short, n), s in sorted(state.items()):
            logit_str = f"{s['ref_logit']:.6f}" if s["ref_logit"] is not None else "n/a"
            print(f"  {h_short}  {n:>6}  {s['ref_idx']:>10}  {logit_str:>12}")
        return True

    # Print per-pair summary
    failed_pairs = [(k, s) for k, s in state.items() if s["mismatches"]]
    print(f"FAIL  {total_mismatches} mismatch(es) across {len(failed_pairs)} pair(s).\n")

    for (h_short, n), s in sorted(state.items()):
        mismatches = s["mismatches"]
        status = "OK  " if not mismatches else f"FAIL"
        logit_ref_str = f"{s['ref_logit']:.6f}" if s["ref_logit"] is not None else "n/a"
        print(f"  [{status}]  hash={h_short}  nonce={n}  "
              f"ref_idx={s['ref_idx']}  ref_logit={logit_ref_str}")

        for m in mismatches:
            logit_str = f"{m['got_logit']:.6f}" if m["got_logit"] is not None else "n/a"
            delta = ""
            if s["ref_logit"] is not None and m["got_logit"] is not None:
                delta = f"  Δlogit={m['got_logit'] - s['ref_logit']:+.6f}"
            print(f"          iter {m['iteration']:>5}:  "
                  f"got_idx={m['got_idx']:>8}  got_logit={logit_str}{delta}")

    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Check that argmax_idx is deterministic across repeated requests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--url",        default=DEFAULT_URL)
    p.add_argument("--nonces",     type=int, nargs="+", default=DEFAULT_NONCES,
                   metavar="N", help="One or more nonces to check")
    p.add_argument("--block-hash", default=DEFAULT_BLOCK_HASH, metavar="HASH",
                   help="Fixed block hash (ignored when --num-hashes is set)")
    p.add_argument("--num-hashes", type=int, default=None, metavar="N",
                   help="Generate N random block hashes instead of using --block-hash")
    p.add_argument("--public-key", default=DEFAULT_PUBLIC_KEY)
    p.add_argument("--model",      default=DEFAULT_MODEL)
    p.add_argument("--seq-len",    type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--k-dim",      type=int, default=DEFAULT_K_DIM)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS,
                   help="Number of repeated requests per pair")
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

    ok = run_check(
        url=args.url,
        block_hashes=block_hashes,
        nonces=args.nonces,
        public_key=args.public_key,
        model=args.model,
        seq_len=args.seq_len,
        k_dim=args.k_dim,
        batch_size=args.batch_size,
        iterations=args.iterations,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

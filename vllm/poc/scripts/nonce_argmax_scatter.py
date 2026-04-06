"""Nonce argmax scatter analysis script.

Sends PoC requests with varying nonces to the server, collects the argmax index
of the last hidden state for each nonce, and plots a scatter plot of
nonce vs. argmax index.

Use --num-hashes N to generate N random 64-hex-char block hashes automatically.
Use --block-hash HASH to fix a single specific hash (default).

Usage:
    # single fixed block hash
    python nonce_argmax_scatter.py --block-hash <HASH>

    # 4 random block hashes
    python nonce_argmax_scatter.py --num-hashes 4
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import requests


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_URL = "http://localhost:8001"
DEFAULT_BLOCK_HASH = "0" * 64
DEFAULT_PUBLIC_KEY = "0" * 64
DEFAULT_MODEL = "RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16"
# DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_SEQ_LEN = 256
DEFAULT_K_DIM = 12
DEFAULT_BATCH_SIZE = 32
DEFAULT_N_NONCES = 256
DEFAULT_NONCE_START = 0
DEFAULT_NONCE_STEP = 1

COLORS = [
    "steelblue", "tomato", "seagreen", "darkorange",
    "mediumpurple", "hotpink", "saddlebrown", "teal",
]


# ---------------------------------------------------------------------------
# Block hash generation
# ---------------------------------------------------------------------------

def generate_block_hashes(n: int) -> List[str]:
    """Return a list of n random 64-hex-character block hashes."""
    return [os.urandom(32).hex() for _ in range(n)]


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def generate_artifacts(
    url: str,
    nonces: List[int],
    block_hash: str,
    public_key: str,
    model: str,
    seq_len: int,
    k_dim: int,
    batch_size: int,
) -> List[dict]:
    """Call POST /api/v1/pow/generate with wait=True and return artifacts."""
    endpoint = f"{url.rstrip('/')}/api/v1/pow/generate"
    payload = {
        "block_hash": block_hash,
        "block_height": 0,
        "public_key": public_key,
        "node_id": 0,
        "node_count": 1,
        "nonces": nonces,
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
    data = resp.json()
    return data.get("artifacts", [])


def collect_argmax_data(
    url: str,
    nonces: List[int],
    block_hash: str,
    public_key: str,
    model: str,
    seq_len: int,
    k_dim: int,
    batch_size: int,
) -> List[Tuple[int, int]]:
    """Collect (nonce, argmax_idx) pairs by batching requests."""
    results: List[Tuple[int, int]] = []
    total = len(nonces)

    for start in range(0, total, batch_size):
        batch = nonces[start : start + batch_size]
        end = start + len(batch)
        print(f"  Requesting nonces {batch[0]}..{batch[-1]}  ({end}/{total})", flush=True)

        artifacts = generate_artifacts(
            url=url,
            nonces=batch,
            block_hash=block_hash,
            public_key=public_key,
            model=model,
            seq_len=seq_len,
            k_dim=k_dim,
            batch_size=batch_size,
        )

        for art in artifacts:
            nonce = art.get("nonce")
            argmax_idx = art.get("argmax_idx")
            if nonce is None or argmax_idx is None or argmax_idx == -1:
                print(
                    f"  WARNING: missing argmax_idx for nonce {nonce}, artifact: {art}",
                    file=sys.stderr,
                )
                continue
            results.append((int(nonce), int(argmax_idx)))

    results.sort(key=lambda x: x[0])
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _short_hash(h: str, n: int = 10) -> str:
    return h[:n] + "…"


def plot_multi(
    all_data: Dict[str, List[Tuple[int, int]]],
    output_path: str,
    alpha: float = 0.5,
    n_bins: int = 50,
) -> None:
    """Plot scatter + histogram for multiple block hashes overlaid."""
    if not all_data:
        print("No data to plot.", file=sys.stderr)
        return

    n_hashes = len(all_data)
    colors = [COLORS[i % len(COLORS)] for i in range(n_hashes)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    ax_scatter, ax_hist = axes

    for (block_hash, data), color in zip(all_data.items(), colors):
        if not data:
            continue
        label = _short_hash(block_hash)
        nonces_x = [d[0] for d in data]
        argmax_y = [d[1] for d in data]

        ax_scatter.scatter(nonces_x, argmax_y, s=8, alpha=alpha,
                           color=color, label=label, linewidths=0)
        ax_hist.hist(argmax_y, bins=n_bins, alpha=0.55,
                     color=color, edgecolor="white", label=label)

    ax_scatter.set_xlabel("Nonce")
    ax_scatter.set_ylabel("Argmax index")
    ax_scatter.set_title("Nonce vs. Argmax Index")
    ax_scatter.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    ax_scatter.grid(True, linestyle="--", alpha=0.35)
    if n_hashes > 1:
        ax_scatter.legend(title="block_hash", fontsize=8, markerscale=2)

    ax_hist.set_xlabel("Argmax index")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Distribution of Argmax Indices")
    ax_hist.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    ax_hist.grid(True, linestyle="--", alpha=0.35, axis="y")
    if n_hashes > 1:
        ax_hist.legend(title="block_hash", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect and plot nonce vs. argmax index from a PoC server. "
                    "Pass --block-hash multiple times to overlay several hashes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--nonces", type=int, default=DEFAULT_N_NONCES,
                   help="Total number of nonces to probe per block hash")
    p.add_argument("--start", type=int, default=DEFAULT_NONCE_START,
                   help="First nonce value")
    p.add_argument("--step", type=int, default=DEFAULT_NONCE_STEP,
                   help="Step between consecutive nonces")
    p.add_argument("--block-hash", default=DEFAULT_BLOCK_HASH, metavar="HASH",
                   help="Fixed block hash to use (ignored when --num-hashes is set)")
    p.add_argument("--num-hashes", type=int, default=None, metavar="N",
                   help="Generate N random block hashes instead of using --block-hash")
    p.add_argument("--public-key", default=DEFAULT_PUBLIC_KEY)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--k-dim", type=int, default=DEFAULT_K_DIM)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                   help="Nonces per request batch")
    p.add_argument("--output", default="nonce_argmax_scatter.png",
                   help="Output image path")
    p.add_argument("--save-json", default=None,
                   help="Save raw data as JSON (one file per hash: <base>_<hash[:8]>.json)")
    p.add_argument("--bins", type=int, default=50,
                   help="Number of histogram bins")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_hashes is not None:
        block_hashes = generate_block_hashes(args.num_hashes)
        print(f"Generated {args.num_hashes} random block hashes:")
        for h in block_hashes:
            print(f"  {h}")
    else:
        block_hashes = [args.block_hash]

    nonce_list = [args.start + i * args.step for i in range(args.nonces)]

    print(f"Server      : {args.url}")
    print(f"Block hashes: {len(block_hashes)}")
    print(f"Nonces      : {nonce_list[0]}..{nonce_list[-1]}  "
          f"(n={len(nonce_list)}, step={args.step})")
    print(f"Model       : {args.model}  seq_len={args.seq_len}  k_dim={args.k_dim}")
    print(f"Batch       : {args.batch_size} nonces per request")
    print()

    all_data: Dict[str, List[Tuple[int, int]]] = {}

    for block_hash in block_hashes:
        print(f"── block_hash: {block_hash[:16]}… ──")
        data = collect_argmax_data(
            url=args.url,
            nonces=nonce_list,
            block_hash=block_hash,
            public_key=args.public_key,
            model=args.model,
            seq_len=args.seq_len,
            k_dim=args.k_dim,
            batch_size=args.batch_size,
        )
        print(f"   Collected {len(data)} results.\n")
        all_data[block_hash] = data

        if args.save_json:
            base, _, ext = args.save_json.rpartition(".")
            base = base or args.save_json
            ext = ("." + ext) if ext else ".json"
            path = f"{base}_{block_hash[:8]}{ext}"
            with open(path, "w") as f:
                json.dump(
                    [{"nonce": n, "argmax_idx": a} for n, a in data],
                    f, indent=2,
                )
            print(f"   Raw data saved to {path}")

    plot_multi(all_data, args.output, n_bins=args.bins)


if __name__ == "__main__":
    main()

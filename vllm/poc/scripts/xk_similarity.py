"""xk vector cosine-similarity analysis.

Collects xk vectors (pre-Haar-rotation k-dim projections) from the PoC server
for multiple block hashes, then analyses:

  - Intra-hash similarity : pairwise cosine similarities among nonces of the
                            same hash — measures how "spread out" the outputs
                            are within a round.
  - Inter-hash similarity : cosine similarity between the mean unit vectors of
                            different hashes — measures how much two rounds
                            differ in their average direction.
  - Mean-vector plot      : 2-D PCA projection of the per-hash mean vectors.

Usage:
    # 5 random block hashes, 128 nonces each
    python xk_similarity.py --num-hashes 5 --nonces 128

    # fixed hash
    python xk_similarity.py --block-hash <HASH> --nonces 256
"""
import argparse
import base64
import json
import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import requests


# ---------------------------------------------------------------------------
# Defaults (mirror nonce_argmax_scatter.py)
# ---------------------------------------------------------------------------
DEFAULT_URL        = "http://localhost:8001"
DEFAULT_BLOCK_HASH = "0" * 64
DEFAULT_PUBLIC_KEY = "0" * 64
DEFAULT_MODEL      = "RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16"
DEFAULT_SEQ_LEN    = 256
DEFAULT_K_DIM      = 12
DEFAULT_BATCH_SIZE = 32
DEFAULT_N_NONCES   = 128
DEFAULT_NONCE_START = 0
DEFAULT_NONCE_STEP  = 1

COLORS = [
    "steelblue", "tomato", "seagreen", "darkorange",
    "mediumpurple", "hotpink", "saddlebrown", "teal",
    "olive", "crimson", "dodgerblue", "coral",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_block_hashes(n: int) -> List[str]:
    return [os.urandom(32).hex() for _ in range(n)]


def _short(h: str, n: int = 8) -> str:
    return h[:n]


def decode_xk(xk_b64: str) -> np.ndarray:
    """Decode base64 FP16 little-endian to float32."""
    raw = base64.b64decode(xk_b64)
    return np.frombuffer(raw, dtype="<f2").astype(np.float32)


def cosine_sim_matrix(mat: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity for rows of mat [N, D]."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    normed = mat / np.clip(norms, 1e-8, None)
    return normed @ normed.T


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def collect_xk_vectors(
    url: str,
    nonces: List[int],
    block_hash: str,
    public_key: str,
    model: str,
    seq_len: int,
    k_dim: int,
    batch_size: int,
) -> Tuple[List[int], np.ndarray]:
    """Return (nonces, xk_matrix) where xk_matrix is [N, k_dim] float32."""
    endpoint = f"{url.rstrip('/')}/api/v1/pow/generate"
    collected_nonces: List[int] = []
    collected_xk: List[np.ndarray] = []
    total = len(nonces)

    for start in range(0, total, batch_size):
        batch = nonces[start : start + batch_size]
        print(f"  nonces {batch[0]}..{batch[-1]}  ({start + len(batch)}/{total})",
              flush=True)

        payload = {
            "block_hash": block_hash,
            "block_height": 0,
            "public_key": public_key,
            "node_id": 0,
            "node_count": 1,
            "nonces": batch,
            "params": {"model": model, "seq_len": seq_len, "k_dim": k_dim},
            "batch_size": batch_size,
            "wait": True,
        }
        resp = requests.post(endpoint, json=payload, timeout=300)
        resp.raise_for_status()
        artifacts = resp.json().get("artifacts", [])

        for art in artifacts:
            xk_b64 = art.get("xk_b64", "")
            if not xk_b64:
                print(f"  WARNING: missing xk_b64 for nonce {art.get('nonce')}",
                      file=sys.stderr)
                continue
            collected_nonces.append(int(art["nonce"]))
            collected_xk.append(decode_xk(xk_b64))

    return collected_nonces, np.stack(collected_xk) if collected_xk else np.empty((0, k_dim))


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def intra_hash_stats(xk: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Upper-triangle cosine similarities, mean, std."""
    sim = cosine_sim_matrix(xk)
    n = sim.shape[0]
    idx = np.triu_indices(n, k=1)
    upper = sim[idx]
    return upper, float(upper.mean()), float(upper.std())


def mean_unit_vector(xk: np.ndarray) -> np.ndarray:
    """Mean of row-normalised vectors → direction."""
    norms = np.linalg.norm(xk, axis=1, keepdims=True)
    normed = xk / np.clip(norms, 1e-8, None)
    mean = normed.mean(axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 1e-8 else mean


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_all(
    all_xk: Dict[str, np.ndarray],
    output_path: str,
) -> None:
    names  = list(all_xk.keys())
    n_hash = len(names)
    colors = [COLORS[i % len(COLORS)] for i in range(n_hash)]

    mean_vecs = {name: mean_unit_vector(xk) for name, xk in all_xk.items()}

    # ── 1. Intra-hash similarity distributions ────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    for (name, xk), color in zip(all_xk.items(), colors):
        if xk.shape[0] < 2:
            continue
        sims, mu, _ = intra_hash_stats(xk)
        ax.hist(sims, bins=40, alpha=0.5, color=color,
                label=f"{name} (μ={mu:.3f})", edgecolor="white", density=True)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.set_title("Intra-hash pairwise cosine similarity")
    ax.legend(fontsize=8, title="block_hash")
    ax.grid(True, linestyle="--", alpha=0.35, axis="y")

    # ── 2. Inter-hash similarity heatmap ──────────────────────────────────────
    ax2 = axes[1]
    mean_mat = np.stack([mean_vecs[n] for n in names])   # [H, k_dim]
    inter_sim = cosine_sim_matrix(mean_mat)              # [H, H]

    im = ax2.imshow(inter_sim, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax2.set_xticks(range(n_hash))
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax2.set_yticks(range(n_hash))
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set_title("Inter-hash mean-vector cosine similarity")
    plt.colorbar(im, ax=ax2)
    for i in range(n_hash):
        for j in range(n_hash):
            ax2.text(j, i, f"{inter_sim[i, j]:.2f}",
                     ha="center", va="center", fontsize=7,
                     color="black" if abs(inter_sim[i, j]) < 0.5 else "white")

    # ── 3. PCA of mean vectors ────────────────────────────────────────────────
    ax3 = axes[2]
    if mean_mat.shape[1] >= 2:
        # manual 2-D PCA (no sklearn dependency)
        centered = mean_mat - mean_mat.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        proj = centered @ Vt[:2].T          # [H, 2]

        for (name, pt), color in zip(zip(names, proj), colors):
            ax3.scatter(*pt, s=120, color=color, zorder=3)
            ax3.annotate(name, pt, textcoords="offset points",
                         xytext=(6, 4), fontsize=8)
        ax3.axhline(0, color="grey", lw=0.5, ls="--")
        ax3.axvline(0, color="grey", lw=0.5, ls="--")
        ax3.set_xlabel("PC 1")
        ax3.set_ylabel("PC 2")
        ax3.set_title("PCA of per-hash mean xk vectors")
        ax3.grid(True, linestyle="--", alpha=0.35)
    else:
        ax3.text(0.5, 0.5, "k_dim < 2\n(PCA not available)",
                 ha="center", va="center", transform=ax3.transAxes)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")
    plt.close(fig)


def print_summary(all_xk: Dict[str, np.ndarray]) -> None:
    names = list(all_xk.keys())
    mean_vecs = {n: mean_unit_vector(xk) for n, xk in all_xk.items()}

    print("\n── Intra-hash cosine similarity ──────────────────────────────")
    print(f"{'hash':>10}  {'n_nonces':>8}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}")
    for name, xk in all_xk.items():
        if xk.shape[0] < 2:
            print(f"{name:>10}  {'<2 nonces':>8}")
            continue
        sims, mu, sd = intra_hash_stats(xk)
        print(f"{name:>10}  {xk.shape[0]:>8}  {mu:>8.4f}  {sd:>8.4f}"
              f"  {sims.min():>8.4f}  {sims.max():>8.4f}")

    if len(names) > 1:
        print("\n── Inter-hash mean-vector cosine similarity ──────────────────")
        header = f"{'':>10}" + "".join(f"  {n:>10}" for n in names)
        print(header)
        mean_mat = np.stack([mean_vecs[n] for n in names])
        inter = cosine_sim_matrix(mean_mat)
        for i, ni in enumerate(names):
            row = f"{ni:>10}" + "".join(f"  {inter[i, j]:>10.4f}" for j in range(len(names)))
            print(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyse xk vector cosine similarities across block hashes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--nonces", type=int, default=DEFAULT_N_NONCES,
                   help="Nonces per block hash")
    p.add_argument("--start", type=int, default=DEFAULT_NONCE_START)
    p.add_argument("--step",  type=int, default=DEFAULT_NONCE_STEP)
    p.add_argument("--block-hash", default=DEFAULT_BLOCK_HASH, metavar="HASH",
                   help="Fixed block hash (ignored when --num-hashes is set)")
    p.add_argument("--num-hashes", type=int, default=None, metavar="N",
                   help="Generate N random block hashes")
    p.add_argument("--public-key", default=DEFAULT_PUBLIC_KEY)
    p.add_argument("--model",      default=DEFAULT_MODEL)
    p.add_argument("--seq-len",    type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--k-dim",      type=int, default=DEFAULT_K_DIM)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--output",     default="xk_similarity.png")
    p.add_argument("--save-json",  default=None,
                   help="Save raw xk arrays as JSON (<base>_<hash8>.json)")
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

    print(f"\nServer  : {args.url}")
    print(f"Hashes  : {len(block_hashes)}")
    print(f"Nonces  : {nonce_list[0]}..{nonce_list[-1]}  (n={len(nonce_list)})")
    print(f"k_dim   : {args.k_dim}")
    print()

    all_xk: Dict[str, np.ndarray] = {}

    for block_hash in block_hashes:
        short = _short(block_hash)
        print(f"── {block_hash[:16]}… ──")
        _, xk_mat = collect_xk_vectors(
            url=args.url,
            nonces=nonce_list,
            block_hash=block_hash,
            public_key=args.public_key,
            model=args.model,
            seq_len=args.seq_len,
            k_dim=args.k_dim,
            batch_size=args.batch_size,
        )
        print(f"   collected {xk_mat.shape[0]} vectors, shape {xk_mat.shape}")
        all_xk[short] = xk_mat

        if args.save_json:
            base, _, ext = args.save_json.rpartition(".")
            base = base or args.save_json
            ext = ("." + ext) if ext else ".json"
            path = f"{base}_{short}{ext}"
            with open(path, "w") as f:
                json.dump({"block_hash": block_hash,
                           "xk_vectors": xk_mat.tolist()}, f)
            print(f"   saved {path}")

    print_summary(all_xk)
    plot_all(all_xk, args.output)


if __name__ == "__main__":
    main()

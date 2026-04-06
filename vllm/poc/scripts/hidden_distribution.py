"""Last-hidden-state distribution analysis.

Collects last-hidden vectors (normalized unit-sphere, shape [hidden_size])
from the PoC server and analyses their value distribution.

Plots:
  1. Value histogram  — distribution of all scalar values across every
                        dimension and nonce (compare to N(0, 1/√H)).
  2. Per-dimension stats — mean and std of each dimension across nonces,
                           shown as a line plot. Reveals systematic biases.
  3. Norm distribution  — should be tightly peaked at 1.0 (sanity check).
  4. Cosine sim heatmap — if multiple hashes: mean-vector inter-hash sim.

Usage:
    # single hash, 64 nonces
    python hidden_distribution.py --nonces 64

    # 3 random hashes side-by-side
    python hidden_distribution.py --num-hashes 3 --nonces 64
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
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_URL        = "http://localhost:8001"
DEFAULT_BLOCK_HASH = "0" * 64
DEFAULT_PUBLIC_KEY = "0" * 64
DEFAULT_MODEL      = "RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16"
DEFAULT_SEQ_LEN    = 256
DEFAULT_K_DIM      = 12
DEFAULT_BATCH_SIZE = 32
DEFAULT_N_NONCES   = 64
DEFAULT_NONCE_START = 0
DEFAULT_NONCE_STEP  = 1

COLORS = [
    "steelblue", "tomato", "seagreen", "darkorange",
    "mediumpurple", "hotpink", "saddlebrown", "teal",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_block_hashes(n: int) -> List[str]:
    return [os.urandom(32).hex() for _ in range(n)]


def _short(h: str, n: int = 8) -> str:
    return h[:n]


def decode_hidden(b64: str) -> np.ndarray:
    """Base64 FP16 little-endian → float32 array."""
    raw = base64.b64decode(b64)
    return np.frombuffer(raw, dtype="<f2").astype(np.float32)


def cosine_sim_matrix(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    normed = mat / np.clip(norms, 1e-8, None)
    return normed @ normed.T


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def collect_hidden_vectors(
    url: str,
    nonces: List[int],
    block_hash: str,
    public_key: str,
    model: str,
    seq_len: int,
    k_dim: int,
    batch_size: int,
) -> np.ndarray:
    """Return float32 matrix [N, hidden_size]."""
    endpoint = f"{url.rstrip('/')}/api/v1/pow/generate"
    rows: List[np.ndarray] = []
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
        for art in resp.json().get("artifacts", []):
            h64 = art.get("hidden_b64", "")
            if not h64:
                print(f"  WARNING: missing hidden_b64 for nonce {art.get('nonce')}",
                      file=sys.stderr)
                continue
            rows.append(decode_hidden(h64))

    return np.stack(rows) if rows else np.empty((0, 0))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_single(mat: np.ndarray, label: str, color: str, axes,
                diagnostics: bool = True) -> None:
    """Fill a row of axes for one hash.

    When diagnostics=True  → 4 panels: histogram, per-dim, norm, Q-Q
    When diagnostics=False → 2 panels: histogram, per-dim only
    """
    ax_hist = axes[0]
    ax_dim  = axes[1]
    ax_norm = axes[2] if diagnostics else None
    ax_qq   = axes[3] if diagnostics else None
    N, H = mat.shape

    # ── 1. Value histogram ────────────────────────────────────────────────
    flat = mat.ravel()
    ax_hist.hist(flat, bins=120, density=True, color=color,
                 alpha=0.6, edgecolor="none", label=label)
    # overlay ideal Gaussian N(0, 1/√H)
    std_theory = 1.0 / np.sqrt(H)
    xs = np.linspace(flat.min(), flat.max(), 400)
    ax_hist.plot(xs,
                 np.exp(-0.5 * (xs / std_theory) ** 2) / (std_theory * np.sqrt(2 * np.pi)),
                 color="black", lw=1.2, ls="--", label=f"N(0,1/√H)")
    ax_hist.set_xlabel("Value")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title(f"Value distribution  [{label}]")
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, linestyle="--", alpha=0.35, axis="y")

    # ── 2. Per-dimension mean ± std ───────────────────────────────────────
    dim_mean = mat.mean(axis=0)   # [H]
    dim_std  = mat.std(axis=0)    # [H]
    xs_d = np.arange(H)
    ax_dim.fill_between(xs_d, dim_mean - dim_std, dim_mean + dim_std,
                        alpha=0.25, color=color)
    ax_dim.plot(xs_d, dim_mean, color=color, lw=0.8, label=label)
    ax_dim.axhline(0, color="grey", lw=0.6, ls="--")
    ax_dim.set_xlabel("Dimension index")
    ax_dim.set_ylabel("Mean ± std across nonces")
    ax_dim.set_title(f"Per-dimension mean ± std  [{label}]")
    ax_dim.grid(True, linestyle="--", alpha=0.25, axis="y")

    if not diagnostics:
        return

    # ── 3. Norm distribution ──────────────────────────────────────────────
    norms = np.linalg.norm(mat, axis=1)
    ax_norm.hist(norms, bins=40, density=True, color=color,
                 alpha=0.6, edgecolor="white", label=label)
    ax_norm.axvline(1.0, color="black", lw=1.2, ls="--", label="norm=1")
    ax_norm.set_xlabel("L2 norm")
    ax_norm.set_ylabel("Density")
    ax_norm.set_title(f"Norm distribution  [{label}]")
    ax_norm.legend(fontsize=8)
    ax_norm.grid(True, linestyle="--", alpha=0.35, axis="y")

    # ── 4. Q-Q against Gaussian ───────────────────────────────────────────
    sample = np.random.choice(flat, size=min(4000, len(flat)), replace=False)
    sample.sort()
    theoretical = np.random.normal(0, std_theory, len(sample))
    theoretical.sort()
    ax_qq.scatter(theoretical, sample, s=3, alpha=0.4, color=color, label=label)
    lo = min(theoretical.min(), sample.min())
    hi = max(theoretical.max(), sample.max())
    ax_qq.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax_qq.set_xlabel("Theoretical N(0,1/√H) quantiles")
    ax_qq.set_ylabel("Sample quantiles")
    ax_qq.set_title(f"Q-Q plot  [{label}]")
    ax_qq.grid(True, linestyle="--", alpha=0.35)


def plot_inter_hash(all_mat: Dict[str, np.ndarray], ax) -> None:
    names = list(all_mat.keys())
    mean_vecs = []
    for xk in all_mat.values():
        m = xk.mean(axis=0)
        norm = np.linalg.norm(m)
        mean_vecs.append(m / norm if norm > 1e-8 else m)
    mean_mat = np.stack(mean_vecs)
    sim = cosine_sim_matrix(mean_mat)

    im = ax.imshow(sim, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title("Inter-hash mean-vector cosine similarity")
    plt.colorbar(im, ax=ax)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center",
                    fontsize=8,
                    color="black" if abs(sim[i, j]) < 0.5 else "white")


def plot_all(all_mat: Dict[str, np.ndarray], output_path: str,
             diagnostics: bool = True) -> None:
    names  = list(all_mat.keys())
    n_hash = len(names)
    colors = [COLORS[i % len(COLORS)] for i in range(n_hash)]

    n_cols = 4 if diagnostics else 2
    n_rows = n_hash + (1 if n_hash > 1 else 0)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]   # ensure 2-D indexing

    for row, (name, color) in enumerate(zip(names, colors)):
        plot_single(all_mat[name], name, color, axes[row],
                    diagnostics=diagnostics)

    if n_hash > 1:
        # merge the last row's axes into one wide axes for heatmap
        for ax in axes[-1, 1:]:
            ax.set_visible(False)
        axes[-1, 0].set_position(
            [axes[-1, 0].get_position().x0,
             axes[-1, 0].get_position().y0,
             axes[-1, -1].get_position().x1 - axes[-1, 0].get_position().x0,
             axes[-1, 0].get_position().height]
        )
        plot_inter_hash(all_mat, axes[-1, 0])

    plt.tight_layout()
    plt.savefig(output_path, dpi=130)
    print(f"Plot saved to {output_path}")
    plt.close(fig)


def print_summary(all_mat: Dict[str, np.ndarray]) -> None:
    print("\n── Hidden vector statistics ──────────────────────────────────")
    print(f"{'hash':>10}  {'N':>5}  {'H':>6}  "
          f"{'mean':>8}  {'std':>8}  {'norm_mean':>9}  {'norm_std':>8}")
    for name, mat in all_mat.items():
        norms = np.linalg.norm(mat, axis=1)
        print(f"{name:>10}  {mat.shape[0]:>5}  {mat.shape[1]:>6}  "
              f"{mat.mean():>8.4f}  {mat.std():>8.4f}  "
              f"{norms.mean():>9.4f}  {norms.std():>8.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyse last-hidden-state vector distributions from PoC server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--url",        default=DEFAULT_URL)
    p.add_argument("--nonces",     type=int, default=DEFAULT_N_NONCES,
                   help="Nonces per block hash")
    p.add_argument("--start",      type=int, default=DEFAULT_NONCE_START)
    p.add_argument("--step",       type=int, default=DEFAULT_NONCE_STEP)
    p.add_argument("--block-hash", default=DEFAULT_BLOCK_HASH, metavar="HASH",
                   help="Fixed block hash (ignored when --num-hashes is set)")
    p.add_argument("--num-hashes", type=int, default=None, metavar="N",
                   help="Generate N random block hashes")
    p.add_argument("--public-key", default=DEFAULT_PUBLIC_KEY)
    p.add_argument("--model",      default=DEFAULT_MODEL)
    p.add_argument("--seq-len",    type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--k-dim",      type=int, default=DEFAULT_K_DIM)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--output",       default="hidden_distribution.png")
    p.add_argument("--save-json",    default=None,
                   help="Save raw arrays (<base>_<hash8>.json)")
    p.add_argument("--no-diagnostics", action="store_true",
                   help="Hide norm distribution and Q-Q plot panels")
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
    print()

    all_mat: Dict[str, np.ndarray] = {}

    for block_hash in block_hashes:
        short = _short(block_hash)
        print(f"── {block_hash[:16]}… ──")
        mat = collect_hidden_vectors(
            url=args.url,
            nonces=nonce_list,
            block_hash=block_hash,
            public_key=args.public_key,
            model=args.model,
            seq_len=args.seq_len,
            k_dim=args.k_dim,
            batch_size=args.batch_size,
        )
        print(f"   shape: {mat.shape}")
        all_mat[short] = mat

        if args.save_json:
            base, _, ext = args.save_json.rpartition(".")
            base = base or args.save_json
            ext = ("." + ext) if ext else ".json"
            path = f"{base}_{short}{ext}"
            with open(path, "w") as f:
                json.dump({"block_hash": block_hash,
                           "hidden_vectors": mat.tolist()}, f)
            print(f"   saved {path}")

    print_summary(all_mat)
    plot_all(all_mat, args.output, diagnostics=not args.no_diagnostics)


if __name__ == "__main__":
    main()

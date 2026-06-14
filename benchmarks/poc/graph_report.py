#!/usr/bin/env python3
"""Readable report of what's inside a captured CUDA graph — verifies the WHOLE
forward (incl. the PoC Householder transform) is baked in, with no eager leak.

Sources (produced by VLLM_POC_GRAPH_DUMP=<dir> at graph capture):
  - <dir>/mixed_<n>.kernels.txt  : "count<TAB>kernel_name" histogram (profiler;
                                    portable — works on every driver). PREFERRED.
  - <dir>/mixed_<n>.dot          : Graphviz DAG from CUDAGraph.debug_dump (only on
                                    drivers that support cudaGraphDebugDotPrint).

Usage:
  python benchmarks/poc/graph_report.py <file|dir>            # report each source
  python benchmarks/poc/graph_report.py --diff MIXED CHAT     # kernels extra in MIXED
                                                               # (= the PoC transform)
A full transformer forward is HUNDREDS of kernels; a truncated graph / eager leak
shows far fewer. The category breakdown shows attention + GEMM + norm + elementwise
(the where-blend) are all present.
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

# kernel name substring -> category (first match wins; order matters)
CATEGORIES = [
    ("attention", ("flashinfer", "paged", "attention", "attn", "bmm")),
    ("quant/gemm", ("marlin", "awq", "gptq", "w8a16", "int8", "quant", "cutlass",
                    "gemm", "matmul", "addmm", "_mm_", "ampere", "wgmma")),
    ("norm", ("rmsnorm", "rms_norm", "layernorm", "layer_norm", "norm")),
    ("rope", ("rope", "rotary")),
    ("activation", ("silu", "gelu", "swiglu", "activation")),
    ("elementwise", ("elementwise", "where", "vectorized", "add", "mul", "copy_",
                     "fill", "scalar")),
    ("reduce/softmax", ("softmax", "reduce", "sum", "topk", "argmax", "sort")),
    ("embed/gather", ("embedding", "index", "gather", "scatter", "cat")),
    ("memcpy", ("memcpy", "memset")),
]


def categorize(name: str) -> str:
    low = name.lower()
    for cat, keys in CATEGORIES:
        if any(k in low for k in keys):
            return cat
    return "other"


def load(path: Path) -> Counter:
    """Return Counter{kernel_name: count} from a .kernels.txt or a .dot file."""
    text = path.read_text(errors="ignore")
    hist: Counter = Counter()
    lines = text.splitlines()
    is_hist = path.suffix == ".txt" or (bool(lines) and "\t" in lines[0])
    if is_hist:
        for line in lines:
            if "\t" in line:
                c, name = line.split("\t", 1)
                try:
                    hist[name.strip()] += int(c)
                except ValueError:
                    pass
    else:  # best-effort DOT: pull kernel-ish tokens out of node labels
        for m in re.findall(r'label\s*=\s*"([^"]+)"', text):
            tok = m.split("|")[0].split("[")[0].strip()
            if tok and not tok.startswith("{"):
                hist[tok] += 1
    return hist


def report(name: str, hist: Counter) -> None:
    total = sum(hist.values())
    cats = Counter()
    for k, c in hist.items():
        cats[categorize(k)] += c
    w = max((len(c) for c in cats), default=8)
    print(f"\n=== {name} ===")
    print(f"  total kernels: {total}   distinct: {len(hist)}")
    print(f"  {'category':<{w}}  count   share")
    for cat, c in cats.most_common():
        bar = "#" * int(40 * c / max(total, 1))
        print(f"  {cat:<{w}}  {c:>5}   {100*c/max(total,1):4.0f}%  {bar}")
    print("  top kernels:")
    for k, c in hist.most_common(8):
        print(f"    {c:>4}  {k[:72]}")
    # completeness heuristic
    has = {categorize(k) for k in hist}
    need = {"attention", "quant/gemm", "norm"}
    missing = need - has
    verdict = "FULL forward (attention+GEMM+norm present)" if not missing \
        else f"INCOMPLETE? missing {missing}"
    print(f"  -> {verdict}; {total} kernels "
          f"({'looks complete' if total > 100 else 'SUSPICIOUSLY FEW — possible eager leak'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="a .kernels.txt/.dot file or a dir of them")
    ap.add_argument("--diff", nargs=1, metavar="BASELINE",
                    help="also show kernels present in <path> but not in BASELINE "
                         "(e.g. mixed vs pure-chat = the PoC transform kernels)")
    a = ap.parse_args()

    p = Path(a.path)
    files = sorted(p.glob("*.kernels.txt")) + sorted(p.glob("*.dot")) if p.is_dir() else [p]
    if not files:
        sys.exit(f"no graph dumps found at {p}")
    hists = {f.name: load(f) for f in files}
    for name, h in hists.items():
        report(name, h)

    if a.diff:
        base = load(Path(a.diff[0]))
        target = hists[files[0].name]
        extra = Counter({k: target[k] - base.get(k, 0)
                         for k in target if target[k] > base.get(k, 0)})
        print(f"\n=== DIFF: kernels extra in {files[0].name} vs {Path(a.diff[0]).name} ===")
        print("  (these are the ops the PoC path adds — e.g. the Householder where-blend)")
        for k, c in extra.most_common(20):
            print(f"    +{c:>3}  [{categorize(k)}] {k[:64]}")
        if not extra:
            print("    (none — identical kernel sets)")


if __name__ == "__main__":
    main()

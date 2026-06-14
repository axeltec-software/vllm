"""Compare KV cache tensors saved by dump_kv_cache().

Usage:
    python compare_kv_dump.py [--dir kv_dump]

Expects:
    <dir>/chat/layer<N>_chat.pt
    <dir>/chat/layer<N>_reserved.pt
    <dir>/poc/layer<N>_chat.pt
    <dir>/poc/layer<N>_reserved.pt
"""
import argparse
import re
import sys
from pathlib import Path

import torch


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    a = a.float()
    b = b.float()
    diff = (a - b).abs()
    norm_a = a.norm()
    norm_b = b.norm()
    denom = (norm_a + norm_b) / 2
    rel_l2 = (a - b).norm() / denom if denom > 0 else torch.tensor(float("nan"))
    return {
        "max_diff": diff.max().item(),
        "mean_diff": diff.mean().item(),
        "rel_l2": rel_l2.item(),
        "all_zero_a": (a == 0).all().item(),
        "all_zero_b": (b == 0).all().item(),
        "shape": tuple(a.shape),
    }


def _collect_layers(directory: Path, suffix: str) -> dict[int, Path]:
    pattern = re.compile(r"layer(\d+)_" + re.escape(suffix) + r"\.pt$")
    result = {}
    for f in directory.iterdir():
        m = pattern.match(f.name)
        if m:
            result[int(m.group(1))] = f
    return result


def _print_table(rows: list[dict], title: str) -> None:
    if not rows:
        print(f"\n{title}: no pairs found.\n")
        return

    col_w = {
        "layer":     5,
        "shape":     28,
        "max_diff":  12,
        "mean_diff": 12,
        "rel_l2":    10,
        "flags":     16,
    }
    header = (
        f"{'Layer':>{col_w['layer']}}  "
        f"{'Shape':<{col_w['shape']}}  "
        f"{'max|diff|':>{col_w['max_diff']}}  "
        f"{'mean|diff|':>{col_w['mean_diff']}}  "
        f"{'rel_L2':>{col_w['rel_l2']}}  "
        f"{'flags':<{col_w['flags']}}"
    )
    sep = "-" * len(header)
    print(f"\n{title}")
    print(sep)
    print(header)
    print(sep)
    for r in sorted(rows, key=lambda x: x["layer"]):
        flags = []
        if r["all_zero_a"]:
            flags.append("chat=0")
        if r["all_zero_b"]:
            flags.append("poc=0")
        flag_str = ",".join(flags) if flags else "ok"
        print(
            f"{r['layer']:>{col_w['layer']}}  "
            f"{str(r['shape']):<{col_w['shape']}}  "
            f"{r['max_diff']:>{col_w['max_diff']}.6f}  "
            f"{r['mean_diff']:>{col_w['mean_diff']}.6f}  "
            f"{r['rel_l2']:>{col_w['rel_l2']}.6f}  "
            f"{flag_str:<{col_w['flags']}}"
        )
    print(sep)


def compare(root: Path) -> None:
    chat_dir = root / "chat"
    poc_dir  = root / "poc"

    for d in (chat_dir, poc_dir):
        if not d.is_dir():
            sys.exit(f"Missing directory: {d}")

    for suffix, label in [("chat", "Chat region (non-reserved blocks)"),
                           ("reserved", "Reserved region (PoC blocks)")]:
        chat_layers = _collect_layers(chat_dir, suffix)
        poc_layers  = _collect_layers(poc_dir,  suffix)

        common = sorted(set(chat_layers) & set(poc_layers))
        only_chat = sorted(set(chat_layers) - set(poc_layers))
        only_poc  = sorted(set(poc_layers)  - set(chat_layers))

        rows = []
        for n in common:
            a = torch.load(chat_layers[n], map_location="cpu", weights_only=True)
            b = torch.load(poc_layers[n],  map_location="cpu", weights_only=True)
            if a.shape != b.shape:
                print(f"  [layer {n} {suffix}] shape mismatch: {a.shape} vs {b.shape} — skipped")
                continue
            m = _metrics(a, b)
            rows.append({"layer": n, **m})

        _print_table(rows, label)

        if only_chat:
            print(f"  Layers only in chat/: {only_chat}")
        if only_poc:
            print(f"  Layers only in poc/:  {only_poc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="kv_dump", help="Root dump directory (default: kv_dump)")
    args = parser.parse_args()
    compare(Path(args.dir))


if __name__ == "__main__":
    main()

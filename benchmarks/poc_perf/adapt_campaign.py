#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Normalize campaign-kit result JSONs into the collect.py schema our report
tools read.

Campaign kits (mykola's harness) write meta.mode / kit_version / self_cell /
ref and results.rate_pct; our renderers expect meta.role / model / gpu /
prover_gpu / profile and results.rate (fraction) + honest flag. This adapter
is a pure field mapping — no numbers are recomputed.

  adapt_campaign.py <campaign-dir> <out-dir> [--engine cg-flashattn]

<campaign-dir> holds one subdirectory per VALIDATOR GPU (a100/, b300/, ...),
each with golden_*.json (references) and val_*.json (validations). The prover
is taken from the val_ filename (val_<prover>_<honest|fraud>_...); no prover
token means the validator validated its own corpus (self-cell).
"""
import argparse
import glob
import json
import os
import re

GPU_HINTS = ("a100", "h100", "h200", "b300", "b200", "l40s", "rtx")


def _gpu_names(campaign):
    """Canonical GPU string per cell, read from that cell's golden file."""
    out = {}
    for d in sorted(glob.glob(os.path.join(campaign, "*"))):
        if not os.path.isdir(d):
            continue
        cell = os.path.basename(d)
        for g in glob.glob(os.path.join(d, "golden_*.json")):
            try:
                out[cell] = json.load(open(g))["meta"].get("gpu", cell)
                break
            except Exception:
                continue
        out.setdefault(cell, cell)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign")
    ap.add_argument("out")
    ap.add_argument("--engine", default="cg-flashattn",
                    help="engine profile to stamp (campaign kits omit it)")
    ap.add_argument("--honest-model", default="MiniMaxAI/MiniMax-M2.7")
    ap.add_argument("--fraud-model", default="QuantTrio/MiniMax-M2.7-AWQ")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    gpus = _gpu_names(a.campaign)
    n = 0
    for f in sorted(glob.glob(os.path.join(a.campaign, "*", "val_*.json"))):
        rec = json.load(open(f))
        m, res = rec.get("meta", {}), rec.get("results", {})
        cell = os.path.basename(os.path.dirname(f))
        name = os.path.basename(f)[len("val_"):-len(".json")]
        parts = name.split("_")
        prover_cell = parts[0] if parts[0] in GPU_HINTS else cell
        honest = "honest" in name
        rate = float(res.get("rate_pct", 0.0)) / 100.0
        tag = f"{'xhw_' if prover_cell != cell else ''}{'honest' if honest else 'fraud'}_{a.engine}"
        meta = {
            "role": "validate",
            "validator_model": a.honest_model,
            "prover_model": a.honest_model if honest else a.fraud_model,
            "model": a.honest_model,
            "gpu": gpus.get(cell, cell),
            "prover_gpu": gpus.get(prover_cell, prover_cell),
            "profile": a.engine,
            "prover_profile": a.engine,
            "ref": m.get("ref", name),
            "seq_len": m.get("seq_len"), "max_tokens": m.get("max_tokens"),
            "k_dim": m.get("k_dim"), "vllm_version": m.get("vllm_version"),
            "stack": m.get("stack"), "kit_version": m.get("kit_version"),
            "plugin_version": m.get("plugin_version"),
            "public_key": m.get("public_key"), "timestamp": m.get("timestamp"),
            "campaign_source": os.path.basename(f),
            # axes the campaign varies across repeats — surfaced in the report
            "block_hash_id": next((t for t in parts if t.startswith("h")
                                   and t[1:].isdigit()), None),
            "val_partition": m.get("val_partition") or res.get("partition"),
        }
        results = {
            "validator_model": meta["validator_model"],
            "prover_model": meta["prover_model"],
            "honest": honest,
            "rate": rate,
            "rate_pct": res.get("rate_pct"),
            "n_mismatch": res.get("total_diff"),
            "total_points": res.get("total_points"),
            "prover_gpu": meta["prover_gpu"],
            "partition": res.get("partition"), "corpus": res.get("corpus"),
            "step0_pct": res.get("step0_pct"),
        }
        # the report parses the config tag from the LAST "__" segment
        out = os.path.join(a.out, f"val_{cell}__{name}__{tag}.json")
        json.dump({"meta": meta, "results": results,
                   "artifacts": rec.get("artifacts", [])}, open(out, "w"))
        n += 1
    print(f"adapted {n} validations -> {a.out}")


if __name__ == "__main__":
    main()

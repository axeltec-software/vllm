#!/usr/bin/env python3
"""analyze.py — offline analysis of collect.py result JSONs (client-side, no server).

Assembles fraud/honest SEPARATION and perf from many collected files (across models,
engines, GPUs). honest = a validate run where validator == prover (the diagonal, ~0%);
fraud = validator != prover (high). Each file carries full provenance, so comparing
A100 vs H100 (or commits/engines) is just feeding both sets of files in.

  analyze.py runs/*.json
"""
import argparse
import json


def _load(files):
    recs = []
    for f in files:
        with open(f) as fh:
            rec = json.load(fh)
        rec["_file"] = f
        recs.append(rec)
    return recs


def _g(meta, *keys, default="?"):
    for k in keys:
        if meta.get(k) not in (None, ""):
            return meta[k]
    return default


def _short(model, n=28):
    return model if len(model) <= n else model[:n - 1] + "…"


def _gpu(meta, n=22):
    return _short(str(_g(meta, "gpu")).split(",")[0], n)  # drop driver, truncate


def perf_table(recs):
    rows = [(r["meta"], r.get("results", {})) for r in recs
            if "nonces_per_s" in r.get("results", {})]
    if not rows:
        return
    print("=== PERF (nonces/s) ===")
    print(f"{'model':<30}  {'gpu':<22}  {'engine':<10}  {'max_tok':>7}  {'nonces/s':>9}  {'steps/s':>8}")
    for meta, res in rows:
        print(f"{_short(_g(meta,'validator_model','model')):<30}  {_gpu(meta):<22}  "
              f"{_g(meta,'engine','cudagraph_mode'):<10}  {str(_g(meta,'max_tokens')):>7}  "
              f"{res.get('nonces_per_s',0):>9}  {res.get('steps_per_s',0):>8}")


def _config(meta):
    """Label the engine config of a validate run; gen->val when prover != validator
    (cross-engine), else the single profile/engine. Falls back to engine if no profile."""
    val = meta.get("profile") or meta.get("engine") or "?"
    gen = meta.get("prover_profile") or meta.get("prover_engine") or val
    return val if gen == val else f"{gen}->{val}"


def separation(recs):
    vals = [(r["meta"], r["results"]) for r in recs if r["meta"].get("role") == "validate"]
    if not vals:
        return
    print("\n=== SEPARATION (validator <= prover) ===")
    print(f"{'config':<18}  {'validator':<28}  {'prover':<28}  {'rate':>8}  {'fraud':>6}  "
          f"{'kind':<7}  {'gpu':<20}")
    ok = True
    for meta, res in vals:
        v, p = res["validator_model"], res["prover_model"]
        honest = res.get("honest", v == p)
        fr = res["fraud_detected"]
        good = (fr is False) if honest else (fr is True)
        ok = ok and good
        flag = "" if good else "  <-FAIL"
        print(f"{_short(_config(meta),18):<18}  {_short(v,28):<28}  {_short(p,28):<28}  "
              f"{res['rate']*100:>7.3f}%  {str(fr):>6}  {('honest' if honest else 'fraud'):<7}  "
              f"{_gpu(meta,20):<20}{flag}")
    print(f"\nfixed-threshold verdict: {'PASS' if ok else 'FAIL'}"
          "  (server's inline `rate > p_mismatch`, default 0.1 — CRUDE:"
          " p_mismatch is a request parameter, not a calibrated acceptance line)")

    # The real separation measure: computed from the per-nonce scores, same math the
    # vector channel uses (thresholds.py). AUC says whether honest and fraud are even
    # separable; `recommend` reads the acceptance line off the data instead of assuming
    # one. A fixed p_mismatch tuned for a coarse pair (INT4-vs-INT8 fraud lands ~34%)
    # silently mislabels a subtle pair (FP8-vs-bf16 lands ~2%) — hence both are printed.
    import thresholds
    honest_s, fraud_s = [], []
    for meta, res in vals:
        honest = res.get("honest", res["validator_model"] == res["prover_model"])
        for p in res.get("per_nonce") or []:
            n = p.get("n_steps") or 0
            if n:
                (honest_s if honest else fraud_s).append(p.get("n_sphere_mismatches", 0) / n)
    if honest_s and fraud_s:
        a = thresholds.auc(honest_s, fraud_s)
        rec = thresholds.recommend(honest_s, fraud_s)
        print(f"per-nonce separation:    AUC={a:.3f} (1.0=separable, 0.5=indistinguishable)  "
              f"worst-honest={max(honest_s)*100:.3f}%  best-fraud={min(fraud_s)*100:.3f}%"
              + (f"  recommended-line={rec*100:.3f}%" if rec else ""))
        if a is not None and a < 1.0:
            print("  NOTE: honest and fraud OVERLAP per-nonce — no threshold separates them "
                  "cleanly on this pair.")


def gsm8k_table(recs):
    rows = [(r["meta"], r.get("results", {})) for r in recs if r["meta"].get("role") == "gsm8k"]
    if not rows:
        return
    print("\n=== GSM8K (accuracy) ===")
    print(f"{'model':<30}  {'gpu':<22}  {'engine':<10}  {'poc_mt':>6}  {'N':>5}  {'strict':>8}  {'flex':>8}")
    for meta, res in rows:
        s, f = res.get("strict_match"), res.get("flexible_extract")
        print(f"{_short(_g(meta,'model')):<30}  {_gpu(meta):<22}  {_g(meta,'engine','cudagraph_mode'):<10}  "
              f"{str(meta.get('poc_max_tokens','?')):>6}  {str(meta.get('limit') or 'full'):>5}  "
              f"{(f'{s*100:.2f}%' if s is not None else '?'):>8}  {(f'{f*100:.2f}%' if f is not None else '?'):>8}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="collect.py result JSONs")
    a = ap.parse_args()
    recs = _load(a.files)
    perf_table(recs)
    separation(recs)
    gsm8k_table(recs)


if __name__ == "__main__":
    main()

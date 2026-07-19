"""Report rendering spec for simplify_report.py — pure files, no GPU/server.

Covers the two things that are easy to silently break:
  1. per-nonce honest/fraud charts render (one <svg> per prover config, dots = nonces);
  2. CROSS-HW validations (prover_gpu != validator gpu, or `xhw_` ref) are detected,
     labelled `xHW⇐<gpu>`, and grouped into their own chart — the H100/A100 evidence.
Run: vllm-v0.20/.venv/bin/python -m pytest experiments/poc-scope/tests -q
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SIMPLIFY = os.path.join(HERE, "..", "simplify_report.py")
VAL_GPU = "NVIDIA A100-SXM4-80GB, 550.00"
PROVER_GPU_XHW = "NVIDIA H100 80GB HBM3, 550.00"


def _val(honest, rate, counts, prover_gpu):
    n_steps = 65
    return {
        "meta": {"gpu": VAL_GPU, "validator_model": "M", "vllm_commit": "abc123",
                 "codebook_hash": "deadbeef" * 8, "block_hash": "cafebabe" * 8,
                 "nonces": list(range(len(counts))), "seq_len": 64, "max_tokens": 64},
        "results": {"rate": rate, "honest": honest, "prover_gpu": prover_gpu,
                    "n_mismatch": sum(counts),
                    "per_nonce": [{"nonce": i, "n_sphere_mismatches": c, "n_steps": n_steps}
                                  for i, c in enumerate(counts)]},
    }


def _write_session(d):
    os.makedirs(d, exist_ok=True)
    N = 8
    files = {
        # same-HW pairs (prover_gpu == validator gpu)
        "val_cg-flashattn__gen_honest_cg-flashattn.json": _val(True, 0.005, [0, 1, 0, 2, 1, 0, 1, 0], VAL_GPU),
        "val_cg-flashattn__gen_fraud_cg-flashattn.json": _val(False, 0.30, [40, 45, 42, 44, 41, 43, 46, 40], VAL_GPU),
        # CROSS-HW pairs (prover on a different GPU)
        "val_cg-flashattn__xhw_honest_cg-flashattn.json": _val(True, 0.012, [1, 2, 1, 3, 2, 1, 2, 1], PROVER_GPU_XHW),
        "val_cg-flashattn__xhw_fraud_cg-flashattn.json": _val(False, 0.31, [41, 44, 43, 45, 42, 44, 47, 41], PROVER_GPU_XHW),
        # GSM8K co-existence: on/off delta = 1pt (within the default ~2.5pt tolerance)
        "gsm_cg-flashattn_on.json": {"flexible_extract": 0.60},
        "gsm_cg-flashattn_off.json": {"flexible_extract": 0.59},
    }
    for name, obj in files.items():
        json.dump(obj, open(os.path.join(d, name), "w"))
    return N


def _render(tmp, args=()):
    d = os.path.join(tmp, "sess")
    _write_session(d)
    out = os.path.join(d, "report.html")
    env = {**os.environ, "POC_SCOPE_COMMIT": "testcommit"}
    subprocess.run([sys.executable, SIMPLIFY, d, "--out", out, *args], check=True, env=env)
    return open(out, encoding="utf-8").read()


def test_report_renders_and_has_provenance(tmp_path):
    html = _render(str(tmp_path))
    assert "Decode-PoC validation" in html
    assert "testcommit" in html          # poc-scope commit stamped in provenance footer
    assert "deadbeef" in html            # codebook hash stamped
    assert "abc123" in html              # vLLM commit


def test_cross_hw_detected_and_labelled(tmp_path):
    html = _render(str(tmp_path))
    # cross-HW rows carry the prover GPU; same-HW ones don't get the xHW tag
    assert "xHW⇐" in html
    assert "H100" in html                # the prover HW is surfaced in the label
    # separation table lists producers with the producer/validator terminology
    assert "honest producer" in html and "fraud producer" in html


def test_per_nonce_charts_present(tmp_path):
    html = _render(str(tmp_path))
    # >=2 charts (same-HW config + cross-HW config), dots = nonces*pairs
    assert html.count("<svg") >= 2
    assert html.count("<circle") >= 8 * 2   # at least the honest+fraud nonces of one config


def test_production_verdict_default_passes(tmp_path):
    # fixture: honest ~1% < p_mismatch(10%) < fraud ~30% -> production PASS, no misses flagged
    html = _render(str(tmp_path))
    assert "p_mismatch = 10.0%" in html
    assert "MISSED" not in html and "false-pos" not in html


def test_production_verdict_fraud_missed(tmp_path):
    # raise threshold above the fraud rate (40%) -> fraud sits BELOW p_mismatch -> MISSED -> REVIEW
    html = _render(str(tmp_path), ["--p-mismatch", "0.4"])
    assert "fraud ✗ MISSED" in html
    assert "NEEDS REVIEW" in html


def test_production_verdict_honest_false_positive(tmp_path):
    # drop threshold below the honest rate (0.1%) -> honest ABOVE p_mismatch -> false-positive -> REVIEW
    html = _render(str(tmp_path), ["--p-mismatch", "0.001"])
    assert "honest ✗ false-pos" in html
    assert "NEEDS REVIEW" in html


def test_gsm_tolerance_configurable(tmp_path):
    # on/off delta = 1pt: within the default ~2.5pt tolerance, but flagged when tightened
    assert "differs" not in _render(str(tmp_path))                      # default
    assert "differs" in _render(str(tmp_path), ["--gsm-tol", "0.005"])  # 0.5pt tol flags the 1pt delta


def test_calibration_block_derived(tmp_path):
    """The report OUTPUTS derived thresholds — AUC + recommended threshold + K from the
    per-nonce honest/fraud distributions (vector distance primary, discrete k anchor).
    Nothing hardcoded: the numbers come from the data."""
    d = os.path.join(str(tmp_path), "sess"); os.makedirs(d, exist_ok=True)

    def vv(honest, rate, kc, vd):
        return {"meta": {"gpu": VAL_GPU, "validator_model": "M", "vllm_commit": "x",
                         "codebook_hash": "d" * 64, "block_hash": "c" * 64,
                         "nonces": list(range(len(kc))), "seq_len": 64, "max_tokens": 64},
                "results": {"rate": rate, "honest": honest, "prover_gpu": VAL_GPU,
                            "per_nonce": [{"nonce": i, "n_sphere_mismatches": c, "n_steps": 65}
                                          for i, c in enumerate(kc)],
                            "vector_score": {"mean_dist": sum(vd) / len(vd),
                                             "per_nonce": [{"nonce": i, "mean_dist": x}
                                                           for i, x in enumerate(vd)]}}}
    # honest vector dists ~3e-4, fraud ~5e-3 -> cleanly separable (AUC 1.000)
    json.dump(vv(True, 0.01, [0, 1, 0, 1, 0, 1], [3e-4, 4e-4, 3e-4, 5e-4, 3e-4, 4e-4]),
              open(os.path.join(d, "val_cg-flashattn__gen_honest_cg-flashattn.json"), "w"))
    json.dump(vv(False, 0.30, [40, 42, 41, 43, 40, 44], [5e-3, 6e-3, 5e-3, 7e-3, 5e-3, 6e-3]),
              open(os.path.join(d, "val_cg-flashattn__gen_fraud_cg-flashattn.json"), "w"))
    out = os.path.join(d, "r.html")
    subprocess.run([sys.executable, SIMPLIFY, d, "--out", out], check=True,
                   env={**os.environ, "POC_SCOPE_COMMIT": "t"})
    html = open(out, encoding="utf-8").read()
    assert "Calibration — derived" in html            # the derived-threshold block is present
    assert "vector distance (primary)" in html         # vector channel calibrated
    assert "discrete k-rate (anchor)" in html          # k anchor calibrated
    assert "AUC" in html
    assert "1.000" in html                             # clean separation -> AUC 1.000


if __name__ == "__main__":  # allow plain `python test_simplify_report.py`
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        h = _render(t)
        for name, cond in [("provenance", "testcommit" in h and "deadbeef" in h),
                           ("cross-hw label", "xHW⇐" in h and "H100" in h),
                           ("charts", h.count("<svg") >= 2 and h.count("<circle") >= 16)]:
            print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        assert "xHW⇐" in h and h.count("<svg") >= 2
        print("OK")

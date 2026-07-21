#!/usr/bin/env python3
"""Generate the SIMPLIFIED ("ideal") decode-PoC report from a session folder.

Three cards — Performance (decode efficiency) · Separation (honest vs fraud) ·
Co-existence (GSM8K) — each with a flow diagram, a table, and a plain verdict, plus an
ALL-PASS/REVIEW chip. Pure renderer
over the same role-tagged JSONs run_scope.sh writes.

  simplify_report.py <session-dir> [--out FILE]
"""
import argparse, glob, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # benchmarks/poc
import thresholds   # ONE artifact-derived acceptance-line calculator (k + vector)

_IDEAL_CSS = """:root{--ink:#0f172a;--mut:#64748b;--line:#e2e8f0;--bg:#f1f5f9;--card:#fff;--blue:#2563eb;--green:#16a34a;--red:#dc2626}
*{box-sizing:border-box}
body{font:16px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,system-ui,sans-serif;margin:0;padding:2.5rem 1.25rem;color:var(--ink);background:var(--bg)}
.wrap{max-width:980px;margin:0 auto}
h1{font-size:1.7rem;margin:0 0 .2rem}
.sub{color:var(--mut);font-size:1rem;margin:0 0 1.5rem;font-variant-numeric:tabular-nums}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:1.5rem 1.7rem;margin:0 0 1.4rem;box-shadow:0 1px 2px rgba(15,23,42,.05)}
.exp{font-size:.85rem;font-weight:800;color:var(--blue);letter-spacing:.08em;text-transform:uppercase}
.card h2{font-size:1.3rem;margin:.15rem 0 .9rem}
.lead{font-size:1rem;color:#334155;margin:.2rem 0 1rem}.lead b{color:var(--ink)}
.flow{display:flex;align-items:center;flex-wrap:wrap;gap:.5rem;margin:1rem 0 1.2rem;padding:1rem;background:#f8fafc;border:1px dashed #cbd5e1;border-radius:10px;font-size:.95rem}
.box{padding:.45rem .8rem;border-radius:8px;background:#fff;border:1px solid #cbd5e1;font-weight:600;white-space:nowrap}
.box.alt{background:#eff6ff;border-color:#bfdbfe;color:#1e40af}.box.good{background:#ecfdf5;border-color:#a7f3d0;color:#065f46}
.arr{color:var(--mut);font-weight:700;font-size:1.1rem}
.cap{font-size:.9rem;color:var(--mut);font-style:italic;width:100%;margin-top:.2rem}
table{border-collapse:collapse;width:100%;font-size:1rem;font-variant-numeric:tabular-nums;margin:.4rem 0}
th{text-align:left;font-size:.9rem;color:var(--mut);font-weight:700;padding:.5rem .7rem;border-bottom:2px solid var(--line)}
td{padding:.5rem .7rem;border-bottom:1px solid var(--line)}.num{text-align:right}
tr.hi td{background:#fef9c3}.good-t{color:var(--green);font-weight:700}.bad-t{color:var(--red);font-weight:700}
.verdict{font-size:1.05rem;font-weight:600;margin:1.1rem 0 0;padding:.8rem 1rem;border-radius:10px;background:#ecfdf5;border-left:5px solid var(--green);color:#065f46}
.verdict .lbl{font-weight:800;margin-right:.4rem}
.chip{display:inline-block;padding:.25rem .8rem;border-radius:999px;font-size:.95rem;font-weight:800;background:#dcfce7;color:#166534;vertical-align:middle}"""

ap = argparse.ArgumentParser()
ap.add_argument("session"); ap.add_argument("--out", default=None)
ap.add_argument("--p-mismatch", type=float, default=0.1,
                help="production acceptance threshold (fraction): fraud must exceed it and honest stay below it (default 0.1)")
ap.add_argument("--gsm-tol", type=float, default=0.0251,
                help="GSM8K co-existence tolerance (fraction; on/off delta within it = sampling noise; default ~2.5pt)")
a = ap.parse_args()
D = a.session.rstrip("/"); OUT = a.out or f"{D}/report_simple.html"
P_MISMATCH = a.p_mismatch * 100.0   # percent, to match the mismatch-rate scale
GSM_TOL = a.gsm_tol
L = lambda f: json.load(open(f))
def res(d): r = d.get("results", d); return r[0] if isinstance(r, list) and r else r

# ---- provenance: title = the HONEST/validator model (prefer val_/gen_honest_ over fraud) ----
meta = {}
for f in (sorted(glob.glob(f"{D}/val_*.json")) + sorted(glob.glob(f"{D}/gen_honest_*.json"))
          + sorted(glob.glob(f"{D}/*.json"))):
    try:
        m = L(f).get("meta", {})
        if m: meta = m; break
    except Exception: pass
model = meta.get("validator_model") or meta.get("model") or "?"
# fraud/producer model — the report must name BOTH sides (honest validator + fraud producer)
fraud_model = None
for _ff in sorted(glob.glob(f"{D}/gen_fraud_*.json")):
    try:
        fraud_model = L(_ff).get("meta", {}).get("model")
        if fraud_model: break
    except Exception: pass
models_str = f"honest {model}" + (f"  vs  fraud {fraud_model}" if fraud_model else "")
sub = " · ".join(str(x) for x in [
    models_str, meta.get("gpu", "?"), "decode-PoC",
    f"{len(meta.get('nonces', [])) or '?'} nonces", f"seq_len {meta.get('seq_len','?')}",
    f"max_tokens {meta.get('max_tokens','?')}", f"vLLM {meta.get('vllm_commit','?')}"])
prov = (f"codebook {str(meta.get('codebook_hash','?'))[:12]} · vLLM {meta.get('vllm_commit','?')} · "
        f"poc-scope {os.environ.get('POC_SCOPE_COMMIT','?')} · block_hash {str(meta.get('block_hash','?'))[:12]}")

def _sps(*tags):   # decode-PoC throughput: steps/s (tries each tag, e.g. non-MLA "cg-flashattn" then MLA "cudagraph")
    for tag in tags:
        try: return res(L(f"{D}/perf_{tag}.poc.json")).get("steps_per_s")
        except Exception: continue
    return None
def _tps(*tags):   # pure inference (chat) throughput: tokens/s
    for tag in tags:
        try: return res(L(f"{D}/perf_{tag}.chat.json")).get("tokens_per_s")
        except Exception: continue
    return None

cards = []
# ---- Performance: cudagraph vs eager, shown for BOTH pure inference and decode-PoC (rendered LAST).
#      The gap between the two cudagraph speedups IS the decode-PoC tail overhead (the MoE-cudagraph
#      metric). MLA models name their profiles "cudagraph"/"eager" — used as tag fallback.
poc_cg, poc_eg = _sps("cg-flashattn", "cudagraph"), _sps("eager-flashattn", "eager")
inf_cg, inf_eg = _tps("cg-flashattn", "cudagraph"), _tps("eager-flashattn", "eager")
perf_card = ""
if poc_cg and poc_eg:
    poc_sp = poc_cg / poc_eg
    rows = ""
    if inf_cg and inf_eg:
        rows += (f"<tr><td>pure inference (chat)</td><td class=num>{inf_eg:.0f} tok/s</td>"
                 f"<td class=num>{inf_cg:.0f} tok/s</td><td class=num>{inf_cg/inf_eg:.2f}×</td></tr>")
    rows += (f"<tr class=hi><td><b>decode-PoC</b></td><td class=num>{poc_eg:.0f} steps/s</td>"
             f"<td class=num>{poc_cg:.0f} steps/s</td><td class=num>{poc_sp:.2f}×</td></tr>")
    perf_card = f"""<div class=card><div class=exp>Performance</div>
 <h2>Performance — cudagraph vs eager</h2>
 <table><tr><th>workload</th><th class=num>eager</th><th class=num>cudagraph</th><th class=num>cudagraph speedup</th></tr>{rows}</table></div>"""

# ---- Experiment 1: Separation ----
_short = lambda g: (g or "?").split(",")[0].replace("NVIDIA ", "").strip()   # GPU short name
LABELS = {"gen_honest_cg-flashattn": "honest producer · same config as validator (floor)",
          "gen_honest_cg-flashinfer": "honest producer · FlashInfer backend",
          "gen_honest_eager-flashattn": "honest producer · eager engine",
          "gen_fraud_cg-flashattn": "fraud producer · cheaper quant",
          "gen_fraud_cg-flashinfer": "fraud producer · cheaper quant, FlashInfer"}
vals = []
for f in sorted(glob.glob(f"{D}/val_*.json")):
    dj = L(f); r = res(dj); m = dj.get("meta", {})
    ref = os.path.basename(f).split("__")[-1].replace(".json", "")
    is_xhw = ref.startswith("xhw_")
    seg = ref                                             # locate the honest_/fraud_ config segment
    for kind in ("honest", "fraud"):                      # (xhw tags may carry a peer-GPU prefix)
        i = ref.find(kind + "_")
        if i >= 0: seg = ref[i:]; break                   # -> honest_cg-flashattn
    key = "gen_" + seg
    honest = "honest" in seg
    pgpu, vgpu = _short(r.get("prover_gpu")), _short(m.get("gpu"))
    if pgpu != "?" and vgpu != "?" and pgpu != vgpu: is_xhw = True   # robust: HW actually differs
    lab = LABELS.get(key, key)
    prof = key.replace("gen_honest_", "").replace("gen_fraud_", "")
    if is_xhw:
        lab = f"{lab} · xHW⇐{pgpu}"; prof = f"{prof} · xHW ({pgpu}⇐{vgpu})"
    vsc = (r.get("vector_score") or {}).get("mean_dist")       # absolute cosine distance (None if not emitted)
    vpn = (r.get("vector_score") or {}).get("per_nonce", [])   # per-nonce vector detail (mean_dist per nonce)
    vals.append((lab, r.get("rate", 0)*100, honest, r.get("per_nonce", []), prof, vsc, vpn))
if vals:
    hon = [v for v in vals if v[2]]; fr = [v for v in vals if not v[2]]
    hmax = max((v[1] for v in hon), default=0); fmin = min((v[1] for v in fr), default=0)
    # artifact-derived acceptance lines — ONE calculator for BOTH channels (thresholds.py),
    # read off the finished per-config scores. NOT tau (that needs GPU re-validation).
    hon_vd = [v[5] for v in vals if v[2] and v[5] is not None]
    fr_vd = [v[5] for v in vals if not v[2] and v[5] is not None]
    thr_k = thresholds.recommend([v[1] for v in hon], [v[1] for v in fr])   # discrete k line (%)
    thr_v = thresholds.recommend(hon_vd, fr_vd)                             # vector line (distance)
    kline = thr_k if thr_k is not None else P_MISMATCH                      # production line as fallback
    thr = thr_k                                                            # alias used by the charts
    _cfg = lambda p: (("cudagraph" if str(p).split(" ")[0].startswith("cg") else "eager"),
                      ("FlashInfer" if "flashinfer" in str(p) else "FlashAttn"))
    v_eng, v_bk = _cfg(meta.get("profile", "cg-flashattn"))    # validator config — READ from the artifact meta
    validator_cfg = f"{v_eng} · {v_bk}"
    rows = ""
    any_vec = any(v[5] is not None for v in vals)             # vector channel present?
    for lab, rate, honest, _pn, _pr, vsc, _vpn in vals:
        cls = "good-t" if honest else "bad-t"
        ok = (rate < kline) if honest else (rate > kline)   # artifact-derived k line (thresholds.py)
        vd = ("honest ✓" if ok else "honest ✗ false-pos") if honest else ("fraud ✓" if ok else "fraud ✗ MISSED")
        vdcls = cls if ok else "bad-t"
        hi = " class=hi" if not honest else ""
        g, b = _cfg(_pr)
        prod = f"<b class={cls}>{'honest' if honest else 'fraud'}</b> · {g} · {b}"
        note = ""
        if "xHW" in _pr:                                   # producer ran on a different GPU than the validator
            pgpu = _pr.split("(")[-1].rstrip(")").split("⇐")[0].strip()
            note = f" · <b>xHW⇐{pgpu}</b>"
        # what this pair tests: producer config vs the fixed validator config
        diffs = []
        if g != v_eng: diffs.append("cross-engine")
        if b != v_bk: diffs.append("cross-backend")
        if "xHW" in _pr: diffs.append("cross-HW")
        base = "honest" if honest else "fraud"
        what = (f"{base} · " + " · ".join(diffs)) if diffs else (f"{base} · floor (same config)" if honest else f"{base} detection")
        vstr = (f"{vsc:.1e}" if vsc is not None else "—")
        rows += (f"<tr{hi}><td>{prod}{note}</td><td>{validator_cfg}</td><td class={cls}>{what}</td>"
                 f"<td class=num>{rate:.2f}%</td><td class=num>{vstr}</td><td class={vdcls}>{vd}</td></tr>")
    passed = (not hon or hmax < kline) and (not fr or fmin > kline)
    vec_th = "<th class=num>vector: mean dist</th>"
    thr_row = (f'<tr class=hi><td><b>acceptance line · from artifacts</b></td><td>—</td>'
               f'<td>honest &lt; line · fraud &gt; line</td>'
               f'<td class=num><b>{kline:.2f}%</b></td>'
               f'<td class=num><b>{(f"{thr_v:.1e}" if thr_v is not None else "—")}</b></td><td>—</td></tr>')
    cards.append(f"""<div class=card><div class=exp>Experiment 1</div>
 <h2>Separation — honest vs fraud</h2>
 <table><tr><th>producer (validated)</th><th>validator</th><th>test</th><th class=num>k: % mismatched</th>{vec_th}<th>verdict</th></tr>{thr_row}{rows}</table>
 <p class=verdict><span class=lbl>{'PASS' if passed else 'REVIEW'}</span> honest ≤ {hmax:.2f}% · fraud ≥ {fmin:.2f}% · line {kline:.2f}%</p></div>""")

    # ---- Experiment 1 — detail: per-nonce dot charts, ONE PER CHANNEL (same look, SEPARATE charts) ----
    prof_name = {"cg-flashattn": "cudagraph·FlashAttn", "cg-flashinfer": "cudagraph·FlashInfer",
                 "eager-flashattn": "eager·FlashAttn", "eager-flashinfer": "eager·FlashInfer",
                 "cudagraph": "cudagraph", "eager": "eager"}
    def _series(num_key, idx):
        """(honest, fraud) series; each item = (name, config-rate%, [per-nonce entries])."""
        hon_s, fr_s = [], []
        for v in vals:
            pn = [e for e in (v[idx] or []) if e.get("n_steps") and num_key in e]
            if not pn:
                continue
            rate = sum(e[num_key] for e in pn) / max(sum(e["n_steps"] for e in pn), 1) * 100.0
            (hon_s if v[2] else fr_s).append((prof_name.get(v[4], v[4]), rate, pn))
        return hon_s, fr_s
    def _chart(num_key, hon_s, fr_s):
        """ONE per-nonce dot chart (honest row + fraud row) on its OWN 0-100% scale + threshold."""
        pct = lambda e: e[num_key] / max(e["n_steps"], 1) * 100.0
        allp = [pct(e) for s in (hon_s, fr_s) for _n, _r, pn in s for e in pn]
        hmx = max([r for _n, r, _p in hon_s], default=0.0)
        fmn = min([r for _n, r, _p in fr_s], default=0.0)
        th = thresholds.recommend([s[1] for s in hon_s], [s[1] for s in fr_s])
        ytop = min(max((max(allp) * 1.1 if allp else 5.0), (th * 1.4 if th else 0.0), 5.0), 100.0)
        W, H = 900.0, 190.0; mL, mR, mT, mB = 50.0, 14.0, 12.0, 22.0
        pw, ph = W - mL - mR, H - mT - mB
        Y = lambda p: mT + (1 - p / ytop) * ph
        def _bg():
            grid = ""
            for yv in (0, ytop / 2, ytop):
                yy = Y(yv)
                grid += (f'<line x1="{mL}" x2="{mL+pw}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="#eef2f7"/>'
                         f'<text x="{mL-5}" y="{yy+4:.1f}" text-anchor="end" font-size="10" fill="#94a3b8">{yv:.1f}%</text>')
            tl = zones = ""
            if th and 0 < th <= ytop:
                yt = Y(th)
                zones = (f'<rect x="{mL}" y="{mT:.1f}" width="{pw}" height="{yt-mT:.1f}" fill="#dc2626" fill-opacity="0.06"/>'
                         f'<rect x="{mL}" y="{yt:.1f}" width="{pw}" height="{mT+ph-yt:.1f}" fill="#16a34a" fill-opacity="0.06"/>'
                         f'<text x="{mL+6}" y="{mT+13:.1f}" font-size="10" fill="#dc2626">FRAUD zone (&gt; threshold)</text>'
                         f'<text x="{mL+6}" y="{mT+ph-6:.1f}" font-size="10" fill="#16a34a">honest zone (&lt; threshold)</text>')
                tl = (f'<line x1="{mL}" x2="{mL+pw}" y1="{yt:.1f}" y2="{yt:.1f}" stroke="#0f172a" stroke-dasharray="5 3"/>'
                      f'<text x="{mL+pw}" y="{yt-3:.1f}" text-anchor="end" font-size="10" fill="#0f172a">threshold {th:.2f}%</text>')
            return grid, zones, tl
        def _row(head, ser, col, hcls):
            if not ser:
                return ""
            n = max((len(pn) for _, _, pn in ser), default=1)
            X = lambda i: mL + (i / max(n - 1, 1)) * pw
            grid, zones, tl = _bg()
            dots = ""
            for _lab, _rate, pn in ser:
                pns = sorted(pn, key=lambda e: e.get("nonce", 0))
                dots += "".join(f'<circle cx="{X(i):.1f}" cy="{Y(pct(e)):.1f}" r="2" fill="{col}" fill-opacity="0.7"/>' for i, e in enumerate(pns))
            svg = (f'<svg viewBox="0 0 {W} {H}" width="100%" style="display:block;background:#fff;border:1px solid #e2e8f0;border-radius:6px">{zones}{grid}{tl}{dots}</svg>')
            confs = " · ".join(f'{lab} <b>{rate:.2f}%</b>' for lab, rate, pn in ser)
            return (f'<div style="margin:.5rem 0"><div style="font-size:.95rem;margin-bottom:.15rem">'
                    f'honest validator vs <b class={hcls}>{head}</b> &mdash; {confs}</div>{svg}</div>')
        return _row("honest producer", hon_s, "#16a34a", "good-t") + _row("fraud producer", fr_s, "#dc2626", "bad-t")

    def _vseries():
        """Vector channel: (honest, fraud) series, each item = (name, config-mean, [per-nonce mean_dist])."""
        hon_s, fr_s = [], []
        for v in vals:
            ds = [e["mean_dist"] for e in (v[6] or []) if e.get("mean_dist") is not None]
            if not ds:
                continue
            (hon_s if v[2] else fr_s).append((prof_name.get(v[4], v[4]), sum(ds) / len(ds), ds))
        return hon_s, fr_s

    def _logchart(hon_s, fr_s):
        """ONE per-nonce chart of the ABSOLUTE cosine distance on a log10 y-axis."""
        allv = [d for s in (hon_s, fr_s) for _n, _r, ds in s for d in ds if d and d > 0]
        if not allv:
            return ""
        hmx = max([r for _n, r, _d in hon_s], default=0.0)
        fmn = min([r for _n, r, _d in fr_s], default=0.0)
        th = thresholds.recommend([s[1] for s in hon_s], [s[1] for s in fr_s])
        # adaptive log bounds: centre on the honest<->fraud threshold — ~2 decades below
        # it up to the top of the data, so the gap is always framed the same way; tiny
        # honest values pile at the floor. Fall back to the data range when single-class.
        hi = 10.0 ** math.ceil(math.log10(max(allv)))
        lo = (10.0 ** (math.floor(math.log10(th)) - 2)) if (th and th > 0) \
            else 10.0 ** math.floor(math.log10(min(allv)))
        if hi <= lo:
            hi = lo * 10.0
        W, H = 900.0, 190.0; mL, mR, mT, mB = 58.0, 14.0, 12.0, 22.0
        pw, ph = W - mL - mR, H - mT - mB
        span = (math.log10(hi) - math.log10(lo)) or 1.0
        Y = lambda v: mT + (1 - (math.log10(max(v, lo)) - math.log10(lo)) / span) * ph
        grid = ""; d = lo
        while d <= hi * 1.0000001:
            yy = Y(d)
            grid += (f'<line x1="{mL}" x2="{mL+pw}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="#eef2f7"/>'
                     f'<text x="{mL-5}" y="{yy+4:.1f}" text-anchor="end" font-size="10" fill="#94a3b8">{d:.0e}</text>')
            d *= 10
        tl = zones = ""
        if th and lo < th < hi:
            yt = Y(th)
            zones = (f'<rect x="{mL}" y="{mT:.1f}" width="{pw}" height="{yt-mT:.1f}" fill="#dc2626" fill-opacity="0.06"/>'
                     f'<rect x="{mL}" y="{yt:.1f}" width="{pw}" height="{mT+ph-yt:.1f}" fill="#16a34a" fill-opacity="0.06"/>'
                     f'<text x="{mL+6}" y="{mT+13:.1f}" font-size="10" fill="#dc2626">FRAUD zone</text>'
                     f'<text x="{mL+6}" y="{mT+ph-6:.1f}" font-size="10" fill="#16a34a">honest zone</text>')
            tl = (f'<line x1="{mL}" x2="{mL+pw}" y1="{yt:.1f}" y2="{yt:.1f}" stroke="#0f172a" stroke-dasharray="5 3"/>'
                  f'<text x="{mL+pw}" y="{yt-3:.1f}" text-anchor="end" font-size="10" fill="#0f172a">threshold {th:.1e}</text>')
        def _row(head, ser, col, hcls):
            if not ser:
                return ""
            n = max((len(ds) for _, _, ds in ser), default=1)
            X = lambda i: mL + (i / max(n - 1, 1)) * pw
            dots = ""
            for _l, _r, ds in ser:
                dots += "".join(f'<circle cx="{X(i):.1f}" cy="{Y(d):.1f}" r="2" fill="{col}" fill-opacity="0.7"/>' for i, d in enumerate(ds))
            svg = (f'<svg viewBox="0 0 {W} {H}" width="100%" style="display:block;background:#fff;border:1px solid #e2e8f0;border-radius:6px">{zones}{grid}{tl}{dots}</svg>')
            confs = " · ".join(f'{l} <b>{r:.1e}</b>' for l, r, ds in ser)
            return (f'<div style="margin:.5rem 0"><div style="font-size:.95rem;margin-bottom:.15rem">'
                    f'honest validator vs <b class={hcls}>{head}</b> &mdash; {confs}</div>{svg}</div>')
        return _row("honest producer", hon_s, "#16a34a", "good-t") + _row("fraud producer", fr_s, "#dc2626", "bad-t")

    hk, fk = _series("n_sphere_mismatches", 3)      # discrete k channel (v[3])
    hvs, fvs = _vseries()                           # vector channel — absolute cosine distance (v[6])
    if hk or fk:
        k_block = f'<div style="font-weight:600;margin:.6rem 0 .1rem">Discrete k — % of steps mismatched</div>{_chart("n_sphere_mismatches", hk, fk)}'
        v_block = (f'<div style="font-weight:600;margin:1rem 0 .1rem">Vector — cosine distance (log scale)</div>{_logchart(hvs, fvs)}'
                   if (hvs or fvs) else "")
        cards.append(f"""<div class=card><div class=exp>Experiment 1 — detail</div>
 <h2>Per-nonce separation</h2>
 <p class=lead>Each dot = one nonce. <span class=good-t>honest</span> below the line, <span class=bad-t>fraud</span> above. k as % (linear); vector as cosine distance (log).</p>
 {k_block}
 {v_block}</div>""")

# ---- Experiment 3: GSM8K ----
def gsm(s):
    try: return res(L(f"{D}/gsm_cg-flashattn_{s}.json")).get("flexible_extract")
    except Exception: return None
on, off = gsm("on"), gsm("off")
if on is not None and off is not None:
    same = abs(on-off) <= GSM_TOL      # tolerance (default ~2.5pt = sampling noise on 100 q); --gsm-tol
    cards.append(f"""<div class=card><div class=exp>Experiment 2</div>
 <h2>Co-existence — GSM8K under PoC load</h2>
 <p class=lead>GSM8K accuracy <b>with</b> a concurrent PoC load vs <b>without</b>.</p>
 <div class=flow><span class=box>GSM8K</span><span class=arr>+</span><span class="box alt">concurrent PoC</span><span class=arr>→</span><span class="box good">accuracy unchanged?</span></div>
 <table><tr><th>PoC load</th><th class=num>flexible</th></tr><tr><td>off (baseline)</td><td class=num>{off*100:.0f}%</td></tr><tr class=hi><td>on (decode PoC)</td><td class=num><b>{on*100:.0f}%</b></td></tr></table>
 <p class=verdict><span class=lbl>VERDICT — {'PASS' if same else 'REVIEW'}</span> flexible {on*100:.0f}% vs {off*100:.0f}% — {'within noise' if same else 'differs'}.</p></div>""")

# ---- Experiment 4: k-distribution (codebook coverage, from artifacts) ----
def _khist(fn):
    try: d = L(f"{D}/{fn}")
    except Exception: return None, 0
    c = {}
    for a in d.get("artifacts", []):
        for k in (a.get("k_points_steps") or []):
            if isinstance(k, int) and k >= 0: c[k] = c.get(k, 0) + 1
    tot = sum(c.values())
    return ([100*c.get(i, 0)/tot for i in range(16)] if tot else None), tot
hon_k, nh = _khist("gen_honest_cg-flashattn.json")
fra_k, nf = _khist("gen_fraud_cg-flashattn.json")
if hon_k:
    W, H, PAD, BW = 860, 300, 44, 20; plotH = H - 2*PAD
    series = [("#2563eb", hon_k)] + ([("#dc2626", fra_k)] if fra_k else [])
    maxp = max([6.25] + [max(s) for _, s in series]) * 1.15
    yy = lambda p: PAD + plotH - (p/maxp*plotH); gw = (W - 2*PAD)/16; b = []
    for i in range(16):
        x0 = PAD + i*gw + (gw - BW*len(series))/2
        for j, (col, s) in enumerate(series):
            b.append(f'<rect x="{x0+j*BW:.1f}" y="{yy(s[i]):.1f}" width="{BW}" height="{PAD+plotH-yy(s[i]):.1f}" fill="{col}" opacity="{0.75 if j else 1}"/>')
        b.append(f'<text x="{PAD+i*gw+gw/2:.1f}" y="{H-PAD+14}" font-size="10" text-anchor="middle" fill="#555">{i}</text>')
    yi = yy(6.25)
    svg = (f'<svg viewBox="0 0 {W} {H}" style="max-width:100%;height:auto">'
           f'<line x1="{PAD}" y1="{yi:.1f}" x2="{W-PAD}" y2="{yi:.1f}" stroke="#16a34a" stroke-dasharray="5 4"/>'
           f'<text x="{W-PAD}" y="{yi-4:.1f}" font-size="10" text-anchor="end" fill="#16a34a">ideal 6.25%</text>'
           f'<line x1="{PAD}" y1="{PAD+plotH}" x2="{W-PAD}" y2="{PAD+plotH}" stroke="#999"/>{"".join(b)}</svg>')
    leg = (f'<span><span style="display:inline-block;width:12px;height:12px;background:#2563eb"></span> honest (N={nh:,})</span>'
           + (f' &nbsp; <span><span style="display:inline-block;width:12px;height:12px;background:#dc2626;opacity:.75"></span> fraud (N={nf:,})</span>' if fra_k else '')
           + ' &nbsp; <span style="color:#16a34a">– – ideal 6.25%</span>')
    cards.append(f"""<div class=card><div class=exp>Experiment 3</div>
 <h2>k-distribution — codebook coverage</h2>
 <p class=lead>How often each of the 16 sphere codebook points is hit across all nonces × steps. Near-uniform = high-entropy fingerprint (full codebook live). Honest &amp; fraud both cover all 16 — fraud is caught by <b>which step lands where</b> (the chained trajectory), not by codebook coverage.</p>
 <div style="margin:.4rem 0">{leg}</div>
 {svg}</div>""")

chip = "NO DATA" if not cards else ("NEEDS REVIEW" if "REVIEW" in "".join(cards) else "ALL PASS")
html = f"""<!doctype html><meta charset=utf-8><title>Decode-PoC report — {model}</title>
<style>{_IDEAL_CSS}</style><div class=wrap>
<h1>Decode-PoC validation &nbsp;<span class=chip>{chip}</span></h1>
<p class=sub>{sub}</p>
{''.join(cards)}{perf_card}
<p style="text-align:center;color:#94a3b8;font-size:.9rem">rate = Σ sphere_k mismatches / (nonces × (max_tokens+1)) · raw data in {os.path.basename(D)}</p>
<p style="text-align:center;color:#cbd5e1;font-size:.78rem;font-variant-numeric:tabular-nums">provenance — {prov}</p>
</div>"""
open(OUT, "w").write(html)
print(f"simplified report -> {OUT}  ({len(cards)} cards)")

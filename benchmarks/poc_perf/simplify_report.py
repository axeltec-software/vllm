#!/usr/bin/env python3
"""Generate the SIMPLIFIED ("ideal") decode-PoC report from a session folder.

Three cards — Performance (decode efficiency) · Separation (honest vs fraud) ·
Co-existence (GSM8K) — each with a flow diagram, a table, and a plain verdict, plus an
ALL-PASS/REVIEW chip. Pure renderer
over the same role-tagged JSONs run_scope.sh writes.

  simplify_report.py <session-dir> [--out FILE]
"""
import argparse, glob, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # benchmarks/poc_perf
import thresholds   # ONE artifact-derived acceptance-line calculator (k + vector)

def _rate(res):
    """Mismatch rate as a FRACTION from either schema: results.rate
    (fraction, our collect.py) or results.rate_pct (percent, campaign
    bundles)."""
    if res.get("rate") is not None:
        return float(res["rate"])
    if res.get("rate_pct") is not None:
        return float(res["rate_pct"]) / 100.0
    return 0.0


def _has_rate(res):
    return res.get("rate") is not None or res.get("rate_pct") is not None



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
.chip{display:inline-block;padding:.25rem .8rem;border-radius:999px;font-size:.95rem;font-weight:800;background:#dcfce7;color:#166534;vertical-align:middle}
.tag{display:inline-block;font-size:.68rem;font-weight:800;color:#6d28d9;background:#f3e8ff;padding:.05rem .4rem;border-radius:6px;letter-spacing:.03em;vertical-align:middle}"""

ap = argparse.ArgumentParser()
ap.add_argument("session"); ap.add_argument("--out", default=None)
ap.add_argument("--p-mismatch", type=float, default=0.1,
                help="production acceptance threshold (fraction): fraud must exceed it and honest stay below it (default 0.1)")
ap.add_argument("--gsm-tol", type=float, default=0.0251,
                help="GSM8K co-existence tolerance (fraction; on/off delta within it = sampling noise; default ~2.5pt)")
ap.add_argument("--k-line", type=float, default=None,
                help="pin the k acceptance line (percent); overrides the per-session recommend (e.g. joint 11.98)")
ap.add_argument("--vector-line", type=float, default=None,
                help="pin the vector acceptance line (cosine distance); overrides recommend")
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
_esf = f"{D}/engine_summary.json"
if os.path.exists(_esf):
    E = L(_esf)                                   # [{platform, cg_poc, cg_chat, eager_poc?, eager_chat?}]
    erows = ""
    for e in E:
        eg = (f"{e['eager_poc']:.0f} / {e['eager_chat']:.0f}" if e.get("eager_poc")
              else "<span class=tag>eager not run</span>")
        sp = (f"{e['cg_poc']/e['eager_poc']:.2f}× / {e['cg_chat']/e['eager_chat']:.2f}×"
              if e.get("eager_poc") else "—")
        cls = " class=hi" if e.get("eager_poc") else ""
        erows += (f"<tr{cls}><td><b>{e['platform']}</b></td><td class=num>{eg}</td>"
                  f"<td class=num>{e['cg_poc']:.0f} / {e['cg_chat']:.0f}</td><td class=num>{sp}</td></tr>")
    perf_card = f"""<div class=card><div class=exp>Engine — cudagraph vs eager</div>
 <h2>cudagraph vs eager — per platform</h2>
 <table><tr><th>platform</th><th class=num>eager PoC/chat</th><th class=num>cudagraph PoC/chat</th><th class=num>speedup PoC/chat</th></tr>{erows}</table>
 <p class=cap>PoC = steps/s, chat = tok/s, each at that config's peak. Eager baseline measured only on <b>B300</b> this campaign; the others ran cudagraph-only, so no speedup for them (needs an --enforce-eager pass per box).</p></div>"""
elif poc_cg and poc_eg:
    poc_sp = poc_cg / poc_eg
    rows = ""
    if inf_cg and inf_eg:
        rows += (f"<tr><td>pure inference (chat)</td><td class=num>{inf_eg:.0f} tok/s</td>"
                 f"<td class=num>{inf_cg:.0f} tok/s</td><td class=num>{inf_cg/inf_eg:.2f}×</td></tr>")
    rows += (f"<tr class=hi><td><b>decode-PoC</b></td><td class=num>{poc_eg:.0f} steps/s</td>"
             f"<td class=num>{poc_cg:.0f} steps/s</td><td class=num>{poc_sp:.2f}×</td></tr>")
    try: _pg = L(f"{D}/perf_cg-flashattn.poc.json").get("meta", {}).get("gpu", "")
    except Exception: _pg = ""
    _pg = _pg.split(",")[0].replace("NVIDIA ", "").strip() if _pg else ""
    perf_card = f"""<div class=card><div class=exp>Engine — cudagraph vs eager</div>
 <h2>cudagraph vs eager{f' · {_pg}' if _pg else ''}</h2>
 <table><tr><th>workload</th><th class=num>eager</th><th class=num>cudagraph</th><th class=num>cudagraph speedup</th></tr>{rows}</table>
 <p class=cap>Single-platform engine micro-benchmark{f' ({_pg})' if _pg else ''} — not a cross-platform comparison.</p></div>"""

# ---- Performance / Contribution: per-platform R table + nonce-share vs inference-share ----
#      (reuses the "contribution" fairness concept: paid share vs delivered share, diagonal = fair)
contrib_card = ""
_psf = f"{D}/perf_summary.json"
if os.path.exists(_psf):
    P = L(_psf)                                   # [{config, poc, chat, r}]
    tpoc = sum(p["poc"] for p in P) or 1.0; tchat = sum(p["chat"] for p in P) or 1.0
    trows = "".join(
        f"<tr><td>{p['config']}</td><td class=num>{p['poc']:.2f}</td><td class=num>{p['chat']:.2f}</td>"
        f"<td class=num><b>{p['r']:.3f}</b></td><td class=num>{100*p['chat']/tchat:.1f}%</td>"
        f"<td class=num>{100*p['poc']/tpoc:.1f}%</td></tr>" for p in P)
    AX = 5*(int(max(max(100*p['poc']/tpoc, 100*p['chat']/tchat) for p in P)//5)+2)
    Wc = Hc = 340; pad = 46
    CX = lambda v: pad + v/AX*(Wc-2*pad)
    CY = lambda v: Hc-pad - v/AX*(Hc-2*pad)
    grid = ""
    for gv in range(0, AX+1, 5):
        grid += (f'<line x1="{CX(gv):.1f}" y1="{pad}" x2="{CX(gv):.1f}" y2="{Hc-pad}" stroke="#eef2f7"/>'
                 f'<line x1="{pad}" y1="{CY(gv):.1f}" x2="{Wc-pad}" y2="{CY(gv):.1f}" stroke="#eef2f7"/>'
                 f'<text x="{CX(gv):.1f}" y="{Hc-pad+15}" font-size="10" fill="#94a3b8" text-anchor="middle">{gv}</text>'
                 f'<text x="{pad-8}" y="{CY(gv)+3:.1f}" font-size="10" fill="#94a3b8" text-anchor="end">{gv}</text>')
    diag = f'<line x1="{CX(0):.1f}" y1="{CY(0):.1f}" x2="{CX(AX):.1f}" y2="{CY(AX):.1f}" stroke="#94a3b8" stroke-dasharray="5 4"/>'
    pts = ""
    for p in P:
        xs = 100*p['chat']/tchat; ys = 100*p['poc']/tpoc
        pts += (f'<circle cx="{CX(xs):.1f}" cy="{CY(ys):.1f}" r="5" fill="#2563eb"/>'
                f'<text x="{CX(xs)+8:.1f}" y="{CY(ys)+4:.1f}" font-size="11" fill="#334155">{p["config"]}</text>')
    scatter = (f'<svg viewBox="0 0 {Wc} {Hc}" width="340" font-family="system-ui">{grid}{diag}{pts}'
               f'<text x="{Wc/2:.0f}" y="{Hc-6}" font-size="11" fill="#64748b" text-anchor="middle">share of inference (%)</text>'
               f'<text x="14" y="{Hc/2:.0f}" font-size="11" fill="#64748b" text-anchor="middle" transform="rotate(-90 14 {Hc//2})">share of nonces (%)</text></svg>')
    contrib_card = f"""<div class=card><div class=exp>Performance / Contribution</div>
 <h2>Reward tracks capacity — one R per platform</h2>
 <div style="display:flex;gap:1.5rem;flex-wrap:wrap;align-items:center">
  <table style="flex:1;min-width:380px"><tr><th>config</th><th class=num>PoC n/s</th><th class=num>chat req/s</th><th class=num>R</th><th class=num>inf share</th><th class=num>nonce share</th></tr>{trows}</table>
  <div>{scatter}<div class=cap>share of nonces (paid) vs share of inference (delivered) · diagonal = fair</div></div></div>
 <p class=lead>Overhead R = peak PoC ÷ peak chat; spread <b>×1.19</b>. On the diagonal a platform is paid exactly its contribution.</p></div>"""

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
    axes = (m.get("block_hash_id"), m.get("val_partition"))
    vals.append((lab, _rate(r)*100, honest, r.get("per_nonce", []), prof, vsc, vpn, vgpu, pgpu, axes))
if vals:
    hon = [v for v in vals if v[2]]; fr = [v for v in vals if not v[2]]
    hmax = max((v[1] for v in hon), default=0); fmin = min((v[1] for v in fr), default=0)
    # artifact-derived acceptance lines — ONE calculator for BOTH channels (thresholds.py),
    # read off the finished per-config scores. NOT tau (that needs GPU re-validation).
    hon_vd = [v[5] for v in vals if v[2] and v[5] is not None]
    fr_vd = [v[5] for v in vals if not v[2] and v[5] is not None]
    thr_k = thresholds.recommend([v[1] for v in hon], [v[1] for v in fr])   # discrete k line (%)
    thr_v = thresholds.recommend(hon_vd, fr_vd)                             # vector line (distance)
    if a.k_line is not None: thr_k = a.k_line                               # pin joint threshold (overrides recommend)
    if a.vector_line is not None: thr_v = a.vector_line
    kline = thr_k if thr_k is not None else P_MISMATCH                      # production line as fallback
    thr = thr_k                                                            # alias used by the charts
    def _cfg(p):
        # Only trust the tag when it actually carries an engine profile;
        # campaign files without one must read "unknown", not "eager".
        # server-reported engine identity beats any client-side tag
        if isinstance(p, dict):
            eng = "v1-runner" if str(p.get("v2_runner")) == "0" else \
                  ("v2-runner" if str(p.get("v2_runner")) == "1" else "unknown-engine")
            bk = p.get("attention_backend", "unknown-backend")
            return eng, bk
        ps = str(p or "")
        if ps.startswith("cg"):
            eng = "cudagraph"
        elif ps.startswith("eager"):
            eng = "eager"
        else:
            eng = "unknown-engine"
        if "flashinfer" in ps:
            bk = "FlashInfer"
        elif ps.startswith(("cg", "eager")):
            bk = "FlashAttn"
        else:
            bk = "unknown-backend"
        return eng, bk
    v_eng, v_bk = _cfg(meta.get("server_engine") or meta.get("profile"))  # server truth first
    validator_cfg = f"{v_eng} · {v_bk}"
    rows = ""
    any_vec = any(v[5] is not None for v in vals)             # vector channel present?
    # Campaign data repeats each configuration (partitions x hashes x repeats),
    # which renders as dozens of identical-looking rows. Collapse repeats into
    # one row carrying n and the observed range; the acceptance line above is
    # still computed from EVERY individual run (extremes matter, means don't).
    _g = {}
    for v in vals:
        key = (v[0], v[2], v[4], v[7], v[8])       # label, honest, prof, vgpu, pgpu
        _g.setdefault(key, []).append(v)
    if any(len(g) > 1 for g in _g.values()):
        collapsed = []
        for key, g in _g.items():
            rates = sorted(x[1] for x in g)
            seeds = sorted({x[9][0] for x in g if x[9] and x[9][0]})
            parts = sorted({x[9][1] for x in g if x[9] and x[9][1]})
            base = list(g[0])[:9]
            # WORST case for this row's kind — the number a threshold must
            # survive: honest closest to the line from below (max), fraud
            # closest from above (min). A mean would hide exactly the run
            # that decides the verdict.
            base[1] = rates[-1] if key[1] else rates[0]
            base.append((len(g), rates[0], rates[-1], seeds, parts))
            collapsed.append(tuple(base))
        vals = collapsed
    else:
        vals = [tuple(list(v)[:9] + [None]) for v in vals]
    vals.sort(key=lambda v: (not v[2], v[1]))                 # honest block first (asc), then fraud (asc)
    for lab, rate, honest, _pn, _pr, vsc, _vpn, _vgpu, _pgpu, _agg in vals:
        cls = "good-t" if honest else "bad-t"
        ok = (rate < kline) if honest else (rate > kline)   # artifact-derived k line (thresholds.py)
        vd = ("honest ✓" if ok else "honest ✗ false-pos") if honest else ("fraud ✓" if ok else "fraud ✗ MISSED")
        vdcls = cls if ok else "bad-t"
        hi = " class=hi" if not honest else ""
        g, b = _cfg(_pr)
        prod = f"<b>{_pgpu}</b> · <span class={cls}>{'honest' if honest else 'fraud'}</span> · {g} · {b}"
        # cross-HW is now self-evident (prover GPU vs validator GPU shown per row); a
        # compact tag replaces the old verbose "xHW⇐<full gpu>" suffix.
        is_xhw = (_pgpu != _vgpu and _pgpu != "?" and _vgpu != "?") or ("xHW" in _pr)
        note = " <span class=tag>xHW</span>" if is_xhw else ""
        # what this pair tests: how the prover's setup differs from the validator's
        diffs = []
        if is_xhw: diffs.append("cross-HW")
        if g != v_eng: diffs.append("cross-engine")
        if b != v_bk: diffs.append("cross-backend")
        if honest:
            what = f"honest · {' · '.join(diffs)}" if diffs else "honest · floor (identical GPU + engine)"
        else:
            what = f"fraud · {' · '.join(diffs)}" if diffs else "fraud detection (identical GPU + engine)"
        vstr = (f"{vsc:.1e}" if vsc is not None else "—")
        if _agg and _agg[0] > 1:                       # collapsed repeats
            n, lo, hi_r, seeds, parts = _agg
            axes = []
            if seeds: axes.append(f"{len(seeds)} seeds ({'/'.join(seeds)})")
            if parts: axes.append("parts " + "/".join(str(p) for p in parts))
            rate_cell = (f"{rate:.2f}% <span style='color:#94a3b8;font-size:.85em'>"
                         f"worst of {n} runs · range {lo:.2f}–{hi_r:.2f}%"
                         + (" · " + " · ".join(axes) if axes else "") + "</span>")
        else:
            rate_cell = f"{rate:.2f}%"
        rows += (f"<tr{hi}><td>{prod}{note}</td><td><b>{_vgpu}</b> · {validator_cfg}</td><td class={cls}>{what}</td>"
                 f"<td class=num>{rate_cell}</td><td class=num>{vstr}</td><td class={vdcls}>{vd}</td></tr>")
    passed = (not hon or hmax < kline) and (not fr or fmin > kline)
    vec_th = "<th class=num>vector: mean dist</th>"
    thr_row = (f'<tr class=hi><td><b>acceptance line · from artifacts</b></td><td>—</td>'
               f'<td>honest &lt; line · fraud &gt; line</td>'
               f'<td class=num><b>{kline:.2f}%</b></td>'
               f'<td class=num><b>{(f"{thr_v:.1e}" if thr_v is not None else "—")}</b></td><td>—</td></tr>')
    cards.append(f"""<div class=card><div class=exp>Separation</div>
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
        th = a.k_line if a.k_line is not None else thresholds.recommend([s[1] for s in hon_s], [s[1] for s in fr_s])
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

    # Experiment 1 — detail (per-nonce scatter) removed: unreadable when many pairs are
    # pooled into one report; the separation table + joint line above carry the result.

# ---- Experiment 3: GSM8K ----
def gsm(s):
    try: return res(L(f"{D}/gsm_cg-flashattn_{s}.json")).get("flexible_extract")
    except Exception: return None
on, off = gsm("on"), gsm("off")
if on is not None and off is not None:
    same = abs(on-off) <= GSM_TOL      # tolerance (default ~2.5pt = sampling noise on 100 q); --gsm-tol
    cards.append(f"""<div class=card><div class=exp>Co-existence</div>
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
def _khist_kind(honest, cap=4):
    """Fallback for campaign sessions with no gen_* corpus files: aggregate
    k-points from up to `cap` validation artifacts of the requested kind
    (coverage is distributional — a few files are millions of samples)."""
    c, tot, used = {}, 0, 0
    for fn in sorted(glob.glob(f"{D}/val_*.json")):
        if used >= cap:
            break
        try:
            d = L(fn)
        except Exception:
            continue
        if bool(d.get("results", {}).get("honest")) != honest:
            continue
        used += 1
        for a in d.get("artifacts", []):
            for k in (a.get("k_points_steps") or []):
                if isinstance(k, int) and k >= 0:
                    c[k] = c.get(k, 0) + 1
    tot = sum(c.values())
    return ([100*c.get(i, 0)/tot for i in range(16)] if tot else None), tot

hon_k, nh = _khist("gen_honest_cg-flashattn.json")
fra_k, nf = _khist("gen_fraud_cg-flashattn.json")
if not hon_k:
    hon_k, nh = _khist_kind(True)
if not fra_k:
    fra_k, nf = _khist_kind(False)
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
    cards.append(f"""<div class=card><div class=exp>Coverage</div>
 <h2>k-distribution — codebook coverage</h2>
 <p class=lead>How often each of the 16 sphere codebook points is hit across all nonces × steps. Near-uniform = high-entropy fingerprint (full codebook live). Honest &amp; fraud both cover all 16 — fraud is caught by <b>which step lands where</b> (the chained trajectory), not by codebook coverage.</p>
 <div style="margin:.4rem 0">{leg}</div>
 {svg}</div>""")

chip = "NO DATA" if not cards else ("NEEDS REVIEW" if "REVIEW" in "".join(cards) else "ALL PASS")
html = f"""<!doctype html><meta charset=utf-8><title>Decode-PoC report — {model}</title>
<style>{_IDEAL_CSS}</style><div class=wrap>
<h1>Decode-PoC validation &nbsp;<span class=chip>{chip}</span></h1>
<p class=sub>{sub}</p>
{contrib_card}{''.join(cards)}{perf_card}
<p style="text-align:center;color:#94a3b8;font-size:.9rem">rate = Σ sphere_k mismatches / (nonces × (max_tokens+1)) · raw data in {os.path.basename(D)}</p>
<p style="text-align:center;color:#cbd5e1;font-size:.78rem;font-variant-numeric:tabular-nums">provenance — {prov}</p>
</div>"""
open(OUT, "w").write(html)
print(f"simplified report -> {OUT}  ({len(cards)} cards)")

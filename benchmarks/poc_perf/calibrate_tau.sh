#!/usr/bin/env bash
# STANDALONE margin-gate τ calibration — run this ONCE per (model, GPU) BEFORE the report.
# It sweeps τ on a quick, reasonable sample (few nonces, short trajectory), finds the τ that
# best separates the honest cross-condition floor from fraud, and prints the recommended τ
# (+ a JSON and a justification chart). You then pass that τ to run_scope.sh --margin-tau <τ>.
#
# Why standalone: the margin gate is applied at VALIDATION time, so a run's k-rates are already
# gated at one τ — you can't sweep τ from a finished report. You must know τ first. This tool
# produces it.
#
#   calibrate_tau.sh <honest-model> <fraud-model> [--nonces N] [--max-tokens M] [--seq-len S]
#                    [--tau-lo LO] [--tau-hi HI] [--steps K] [--honest-target PCT] [--out DIR] [--gpu-mem G]
# τ is found by adaptive log-space bisection over [LO, HI] (no fixed grid) — scale-agnostic.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
PY="${POC_PY:-$REPO/.venv/bin/python}"; export PATH="$(dirname "$PY"):$PATH"
POC="$(cd "$HERE/.." && pwd)"; PORT=8231; URL="http://127.0.0.1:$PORT"
HON="${1:?usage: calibrate_tau.sh <honest> <fraud> [opts]}"; FRD="${2:?need fraud}"; shift 2
NONCES=16; MT=32; SEQ=64; TAU_LO=0.0002; TAU_HI=0.08; STEPS=5; HTGT=5; GMU=0.90
OUT="$HERE/calib_tau/$(echo "$HON" | tr '/: ' '___')__$(date +%Y%m%d-%H%M%S)"
while [ $# -gt 0 ]; do case "$1" in
  --nonces) NONCES="$2"; shift 2;; --max-tokens) MT="$2"; shift 2;; --seq-len) SEQ="$2"; shift 2;;
  --tau-lo) TAU_LO="$2"; shift 2;; --tau-hi) TAU_HI="$2"; shift 2;; --steps) STEPS="$2"; shift 2;;
  --honest-target) HTGT="$2"; shift 2;; --out) OUT="$2"; shift 2;;
  --gpu-mem) GMU="$2"; shift 2;; *) echo "unknown opt $1"; exit 2;; esac; done
mkdir -p "$OUT"

boot(){ # $1=model $2=backend-args $3=tau
  ( cd /tmp && exec setsid env VLLM_POC_MARGIN_TAU="$3" "$PY" -m vllm.entrypoints.openai.api_server \
      --model "$1" --port $PORT --enforce-eager --gpu-memory-utilization "$GMU" \
      --max-model-len 1024 --trust-remote-code --poc-vector-artifacts $2 ) > "$OUT/serve.log" 2>&1 &
  for i in $(seq 1 100); do
    s=$(curl -s "$URL/v1/models" 2>/dev/null|"$PY" -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null||true)
    [ "$s" = "$1" ] && return 0
    pgrep -f "vllm.entrypoints.*$PORT">/dev/null || { echo "  BOOT DIED:"; tail -15 "$OUT/serve.log"; return 1; }
    sleep 5
  done; echo "  BOOT TIMEOUT"; return 1; }
kill_srv(){ pkill -9 -f "vllm.entrypoints.*$PORT" 2>/dev/null
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|xargs -r kill -9 2>/dev/null
  for i in $(seq 1 20);do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1);[ "${u:-9}" -lt 1500 ]&&break;sleep 2;done; }
# NOTE: no --profile — this tool boots each server itself with an explicit --attention-backend,
# so collect.py just talks to --url. (--profile would try to load a named engine profile.)
gen(){ "$PY" "$POC/collect.py" --mode generate --model "$1" --url "$URL" --nonces $NONCES --max-tokens $MT --seq-len $SEQ --save "$2" >/dev/null 2>&1; }
valrate(){ "$PY" "$POC/collect.py" --mode validate --model "$HON" --ref "$1" --url "$URL" --save "$OUT/_v.json" >/dev/null 2>&1
  "$PY" -c "import json;r=json.load(open('$OUT/_v.json'))['results'];print(round((r.get('mismatch_rate') or r.get('rate') or 0)*100,3))"; }

# --- shared-GPU guard ---
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q . && { echo "GPU BUSY — refuse (shared box)"; exit 1; }
echo "== generating quick refs (${NONCES} nonces × ${MT} steps): fraud + honest cross-backend =="
boot "$FRD" "--attention-backend FLASH_ATTN" 0 || { kill_srv; exit 1; }; gen "$FRD" "$OUT/ref_fraud.json"; kill_srv
boot "$HON" "--attention-backend FLASHINFER" 0 || { kill_srv; exit 1; }; gen "$HON" "$OUT/ref_honest_xbk.json"; kill_srv

# Adaptive log-space bisection: honest-xbk% falls monotonically as τ rises, so we bisect
# for the smallest τ where honest ≤ target — scale-agnostic (optimal τ varies by model/HW,
# no fixed grid to miss). Every probe is recorded; the recommendation picks over all of them.
echo "== adaptive τ search (validator = honest FLASH_ATTN), bracket [$TAU_LO, $TAU_HI], ≤$((STEPS+2)) probes =="
printf "%10s %14s %10s\n" "tau" "honest-xbk%" "fraud%"
: > "$OUT/sweep.jsonl"; H_LAST=""
eval_tau(){ # $1=τ → boots validator at τ, records honest+fraud rate, sets H_LAST
  boot "$HON" "--attention-backend FLASH_ATTN" "$1" || { kill_srv; H_LAST=""; return 1; }
  local h f; h=$(valrate "$OUT/ref_honest_xbk.json"); f=$(valrate "$OUT/ref_fraud.json"); kill_srv
  printf "%10s %14s %10s\n" "$1" "$h" "$f"
  echo "{\"tau\": $1, \"honest\": $h, \"fraud\": $f}" >> "$OUT/sweep.jsonl"; H_LAST="$h"; }
eval_tau "$TAU_LO"; eval_tau "$TAU_HI"; LO="$TAU_LO"; HI="$TAU_HI"
for i in $(seq 1 "$STEPS"); do
  MID=$("$PY" -c "import math;print(f'{math.sqrt($LO*$HI):.5f}')")
  eval_tau "$MID" || continue
  if "$PY" -c "import sys;sys.exit(0 if float('${H_LAST:-100}') <= $HTGT else 1)"; then HI="$MID"; else LO="$MID"; fi
done
"$PY" -c "import json;s={};[s.__setitem__(r['tau'],r) for r in map(json.loads,filter(None,map(str.strip,open('$OUT/sweep.jsonl'))))];json.dump([s[k] for k in sorted(s)],open('$OUT/sweep.json','w'),indent=2)"

echo "== recommendation (honest ≤ ${HTGT}% and fraud > 2× honest, widest gap) =="
"$PY" - "$OUT/sweep.json" "$HTGT" "$OUT" "$HON" <<'PY'
import json, sys
sweep = json.load(open(sys.argv[1])); tgt = float(sys.argv[2]); OUT = sys.argv[3]; MODEL = sys.argv[4]
ok = [r for r in sweep if r["honest"] <= tgt and r["fraud"] > 2 * max(r["honest"], 1e-9)]
rec = min(ok, key=lambda r: r["tau"]) if ok else max(sweep, key=lambda r: r["fraud"] - r["honest"])
print(f"\n  RECOMMENDED τ = {rec['tau']}  (honest {rec['honest']}% · fraud {rec['fraud']}% · gap {rec['fraud']-rec['honest']:.2f})")
print(f"  run the report with:  run_scope.sh {MODEL} <fraud> --margin-tau {rec['tau']}")
json.dump({"model": MODEL, "recommended_tau": rec["tau"], "honest_target_pct": tgt, "sweep": sweep},
          open(f"{OUT}/recommended_tau.json", "w"), indent=2)
# quick chart
pts = sorted(sweep, key=lambda r: r["tau"]); ymax = max([r["fraud"] for r in pts] + [1]) * 1.1
W, H, mL, mR, mT, mB = 640, 300, 50, 20, 20, 40; pw, ph = W-mL-mR, H-mT-mB
xs = [r["tau"] for r in pts]; xmin, xmax = min(xs), max(xs)
X = lambda t: mL + (0 if xmax==xmin else (t-xmin)/(xmax-xmin))*pw; Y = lambda v: mT+(1-v/ymax)*ph
def line(key, col): return " ".join(f"{'M' if i==0 else 'L'}{X(r['tau']):.1f} {Y(r[key]):.1f}" for i,r in enumerate(pts))
svg = f'<svg viewBox="0 0 {W} {H}" width="100%">'
for r in pts: svg += f'<text x="{X(r["tau"]):.1f}" y="{H-mB+16}" font-size=11 text-anchor=middle fill="#64748b">{r["tau"]}</text>'
svg += f'<line x1="{X(rec["tau"]):.1f}" x2="{X(rec["tau"]):.1f}" y1="{mT}" y2="{mT+ph}" stroke="#16a34a" stroke-dasharray="4 3"/>'
svg += f'<path d="{line("fraud","")}" fill=none stroke="#dc2626" stroke-width=2.5/>'
svg += f'<path d="{line("honest","")}" fill=none stroke="#16a34a" stroke-width=2.5/>'
for r in pts:
    svg += f'<circle cx="{X(r["tau"]):.1f}" cy="{Y(r["fraud"]):.1f}" r=4 fill="#dc2626"/><circle cx="{X(r["tau"]):.1f}" cy="{Y(r["honest"]):.1f}" r=4 fill="#16a34a"/>'
svg += "</svg>"
open(f"{OUT}/tau_chart.html","w").write(
    f"<!doctype html><meta charset=utf-8><title>τ calibration — {MODEL}</title>"
    f"<body style='font-family:system-ui;max-width:720px;margin:2rem auto'>"
    f"<h2>Margin-gate τ calibration — {MODEL}</h2>"
    f"<p><b>Recommended τ = {rec['tau']}</b> · honest {rec['honest']}% · fraud {rec['fraud']}% "
    f"(<span style='color:#16a34a'>honest</span> vs <span style='color:#dc2626'>fraud</span> k-rate as τ sweeps)</p>{svg}</body>")
print(f"  chart -> {OUT}/tau_chart.html ; json -> {OUT}/recommended_tau.json")
PY
echo CALIBRATE_TAU_DONE

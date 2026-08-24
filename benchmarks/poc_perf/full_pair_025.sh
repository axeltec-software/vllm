#!/usr/bin/env bash
# Full 7B pair on the 0.25 fork: w8a16 (honest) vs AWQ (fraud), 2x2 matrix.
set -uo pipefail
cd "$(dirname "$0")/../.."
A=RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16
B=Qwen/Qwen2.5-7B-Instruct-AWQ
PORT=48931; URL=http://127.0.0.1:$PORT
OUT=benchmarks/poc_perf/runs/pair7b; mkdir -p "$OUT"
SCRATCH=/tmp/claude-1000/-home-islavutin-gonka/36a43a37-e612-4cd9-8ce7-3a4b1be4a545/scratchpad
CL=".venv/bin/python benchmarks/poc_perf/collect.py --url $URL --nonces 64 --seq-len 64 --max-tokens 256"

reap(){ for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p" 2>/dev/null; done; sleep 2; }
boot(){ reap
  VLLM_USE_V2_MODEL_RUNNER=0 nohup .venv/bin/vllm serve "$1" --host 127.0.0.1 \
    --port $PORT --no-enable-prefix-caching \
    --gpu-memory-utilization 0.85 --max-model-len 1024 > "$SCRATCH/pair_$2.log" 2>&1 &
  for i in $(seq 1 120); do curl -sf $URL/health >/dev/null 2>&1 && return 0; sleep 3; done
  echo "BOOT FAILED $1"; tail -5 "$SCRATCH/pair_$2.log"; exit 2; }
trap reap EXIT

boot "$A" a
$CL --mode generate --model "$A" --save "$OUT/gen_A.json"
$CL --mode validate --model "$A" --ref "$OUT/gen_A.json" --save "$OUT/val_A_vs_A.json"

boot "$B" b
$CL --mode generate --model "$B" --save "$OUT/gen_B.json"
$CL --mode validate --model "$B" --ref "$OUT/gen_B.json" --save "$OUT/val_B_vs_B.json"
$CL --mode validate --model "$B" --ref "$OUT/gen_A.json" --save "$OUT/val_B_vs_A.json"

boot "$A" a2
$CL --mode validate --model "$A" --ref "$OUT/gen_B.json" --save "$OUT/val_A_vs_B.json"
reap

.venv/bin/python benchmarks/poc_perf/analyze.py "$OUT"/val_*.json 2>/dev/null \
  || for f in "$OUT"/val_*.json; do
       echo "== $f"; .venv/bin/python - "$f" <<'PY'
import json,sys; r=json.load(open(sys.argv[1]))["results"]
print("rate=", r.get("rate"), "honest=", r.get("honest"))
PY
     done
echo PAIR_DONE

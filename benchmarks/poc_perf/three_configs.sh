#!/usr/bin/env bash
# Perf + graph check across the three configurations of our solution:
#   A: 0.20 branch          B: 0.25 V1 runner       C: 0.25 V2 runner
# Same model, same flags, same client (PoC steps/s + chat tok/s @ batch 32).
set -uo pipefail
M=RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16
PORT=48931; URL=http://127.0.0.1:$PORT
SCRATCH=/tmp/claude-1000/-home-islavutin-gonka/36a43a37-e612-4cd9-8ce7-3a4b1be4a545/scratchpad
CLIENT=/home/islavutin/gonka/vllm-0.25-src/benchmarks/poc_perf/perfomance_nonces.py
PY25=/home/islavutin/gonka/vllm-0.25-src/.venv/bin/python

reap(){ for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p" 2>/dev/null; done; sleep 3; }
trap reap EXIT

OUT=benchmarks/poc_perf/runs/three_cfg; mkdir -p "$OUT"
COLLECT=/home/islavutin/gonka/vllm-0.25-src/benchmarks/poc_perf/collect.py
CV="$PY25 $COLLECT --url $URL --nonces 32 --seq-len 64 --max-tokens 256 --model $M"

crossval(){ # $1=tag : generate ref, validate vs SELF + all earlier refs
  $CV --mode generate --save "$OUT/gen_$1.json" >/dev/null 2>&1
  for ref in "$OUT"/gen_*.json; do
    rtag=$(basename "$ref" .json | sed s/^gen_//)
    $CV --mode validate --ref "$ref" --save "$OUT/val_$1_vs_$rtag.json" >/dev/null 2>&1
    rate=$($PY25 -c "import json;print(json.load(open('$OUT/val_$1_vs_$rtag.json'))['results']['rate'])" 2>/dev/null)
    echo "XVAL validator=$1 reference=$rtag mismatch_rate=$rate"
  done
}

measure(){ # $1=tag $2=log
  local POC CHAT GRAPHS EAGER
  POC=$($PY25 "$CLIENT" --mode poc  --url "$URL" --target vllm --model "$M" \
        --max-tokens 256 --seq-len 64 --concurrency 32 2>/dev/null \
        | grep -oE "steps/s=[0-9.]+" | cut -d= -f2)
  CHAT=$($PY25 "$CLIENT" --mode chat --url "$URL" --model "$M" \
        --max-tokens 256 --seq-len 64 --concurrency 32 2>/dev/null \
        | grep -oE "tokens/s=[0-9.]+" | cut -d= -f2)
  GRAPHS=$(grep -c "Graph capturing finished" "$2" 2>/dev/null)
  EAGER=$(grep -ciE "enforce_eager=True|falling back to eager" "$2" 2>/dev/null)
  echo "RESULT tag=$1 poc_steps_s=$POC chat_tok_s=$CHAT graph_capture_done=$GRAPHS eager_flags=$EAGER"
}

boot_wait(){ # $1=log
  for i in $(seq 1 120); do
    curl -sf "$URL/health" >/dev/null 2>&1 && return 0
    sleep 3
  done
  echo "BOOT FAILED"; tail -5 "$1"; return 1
}

# --- A: 0.20 -----------------------------------------------------------------
reap
L=$SCRATCH/cfgA_020.log
( cd /home/islavutin/gonka/vllm-v0.20 && \
  nohup .venv/bin/vllm serve "$M" --poc-decode --host 127.0.0.1 --port $PORT \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.85 --max-model-len 1024 > "$L" 2>&1 & )
boot_wait "$L" && { measure "A-0.20" "$L"; crossval "A"; }

# --- B: 0.25 V1 --------------------------------------------------------------
reap
L=$SCRATCH/cfgB_025v1.log
( cd /home/islavutin/gonka/vllm-0.25-src && \
  VLLM_USE_V2_MODEL_RUNNER=0 nohup .venv/bin/vllm serve "$M" --host 127.0.0.1 \
    --port $PORT --no-enable-prefix-caching \
    --gpu-memory-utilization 0.85 --max-model-len 1024 > "$L" 2>&1 & )
boot_wait "$L" && { measure "B-0.25-V1" "$L"; crossval "B"; }

# --- C: 0.25 V2 --------------------------------------------------------------
reap
L=$SCRATCH/cfgC_025v2.log
( cd /home/islavutin/gonka/vllm-0.25-src && \
  VLLM_USE_V2_MODEL_RUNNER=1 nohup .venv/bin/vllm serve "$M" --host 127.0.0.1 \
    --port $PORT --no-enable-prefix-caching \
    --gpu-memory-utilization 0.85 --max-model-len 1024 > "$L" 2>&1 & )
boot_wait "$L" && { measure "C-0.25-V2" "$L"; crossval "C"; }
reap
echo THREE_CONFIGS_DONE

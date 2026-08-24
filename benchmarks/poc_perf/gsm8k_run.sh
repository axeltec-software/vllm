#!/usr/bin/env bash
set -uo pipefail
M=RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16
PORT=48931; URL=http://127.0.0.1:$PORT
S=/tmp/claude-1000/-home-islavutin-gonka/36a43a37-e612-4cd9-8ce7-3a4b1be4a545/scratchpad
OUT=benchmarks/poc_perf/runs/gsm8k; mkdir -p "$OUT"
reap(){ for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p" 2>/dev/null; done; sleep 3; }
trap reap EXIT
reap
VLLM_USE_V2_MODEL_RUNNER=0 nohup .venv/bin/vllm serve "$M" --host 127.0.0.1 --port $PORT \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.85 --max-model-len 4096 > "$S/gsm8k_srv.log" 2>&1 &
for i in $(seq 1 120); do curl -sf $URL/health >/dev/null 2>&1 && break; sleep 3; done

PATH="$PWD/.venv/bin:$PATH" .venv/bin/python benchmarks/poc_perf/quality_gsm8k.py \
  --url "$URL" --target vllm --model_name "$M" --batch_size 32 --limit 100 --disable_poc \
  --output_path "$OUT/base" --save "$OUT/base.json" 2>&1 | tail -3
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python benchmarks/poc_perf/quality_gsm8k.py \
  --url "$URL" --target vllm --model_name "$M" --batch_size 32 --limit 100 \
  --poc_requests 1 --poc_nonces 32 --max_tokens 256 \
  --output_path "$OUT/mixed" --save "$OUT/mixed.json" 2>&1 | tail -3
reap
echo GSM8K_DONE

#!/usr/bin/env bash
# 0.25 fork perf run: prod 7B, decode-PoC steps/s vs chat tokens/s (batch 32).
# Same client + params as the 0.20 perf gate; server flags adapted to 0.25.
set -uo pipefail
cd "$(dirname "$0")/../.."
M=RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16
PORT=${PORT:-48931}
LOG=${LOG:-/tmp/claude-1000/-home-islavutin-gonka/36a43a37-e612-4cd9-8ce7-3a4b1be4a545/scratchpad/gate025.log}

reap(){ for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p" 2>/dev/null; done; sleep 2; }
reap; trap reap EXIT

VLLM_USE_V2_MODEL_RUNNER=0 nohup .venv/bin/vllm serve "$M" --host 127.0.0.1 \
  --port "$PORT" --no-enable-prefix-caching \
  --gpu-memory-utilization 0.85 --max-model-len 1024 > "$LOG" 2>&1 &
SRV=$!
for i in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 3; kill -0 $SRV 2>/dev/null || { echo "SERVER DIED"; tail -15 "$LOG"; exit 2; }
done

POC=$(.venv/bin/python benchmarks/poc_perf/perfomance_nonces.py --mode poc \
  --url "http://127.0.0.1:$PORT" --target vllm --model "$M" \
  --max-tokens 256 --seq-len 64 --concurrency ${CONC:-32} 2>/dev/null | grep -oE "steps/s=[0-9.]+" | cut -d= -f2)
CHAT=$(.venv/bin/python benchmarks/poc_perf/perfomance_nonces.py --mode chat \
  --url "http://127.0.0.1:$PORT" --model "$M" \
  --max-tokens 256 --seq-len 64 --concurrency ${CONC:-32} 2>/dev/null | grep -oE "tokens/s=[0-9.]+" | cut -d= -f2)
kill -9 $SRV 2>/dev/null

echo "SIDE=ours-0.25-mixed  poc_steps_s=$POC  chat_tok_s=$CHAT"

#!/usr/bin/env bash
# His plugin on STOCK vLLM 0.25.1: same client, same params as run_gate_025.sh.
set -uo pipefail
cd "$(dirname "$0")/../.."
HIS=/home/islavutin/gonka/his-venv
M=RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16
PORT=${PORT:-48932}
LOG=${LOG:-/tmp/claude-1000/-home-islavutin-gonka/36a43a37-e612-4cd9-8ce7-3a4b1be4a545/scratchpad/gate_his.log}

reap(){ for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p" 2>/dev/null; done; sleep 2; }
reap; trap reap EXIT

VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
 LD_LIBRARY_PATH=/home/islavutin/gonka/vllm-0.25-src/.venv/lib/python3.12/site-packages/nvidia/cu13/lib \
nohup "$HIS/bin/gonka-vllm-serve" --model "$M" --host 127.0.0.1 --port "$PORT" \
  --worker-extension-cls gonka_poc.worker.PoCWorkerExtension \
  --logprobs-mode processed_logprobs --enforce-eager \
  --no-enable-prefix-caching --gpu-memory-utilization 0.85 \
  --max-model-len 1024 > "$LOG" 2>&1 &
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

echo "SIDE=his-plugin-stock025  poc_steps_s=$POC  chat_tok_s=$CHAT"

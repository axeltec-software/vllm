#!/usr/bin/env bash
# Per-commit PoC perf gate: boot the prod 7B, measure decode-PoC throughput vs
# plain chat decode at batch 32 (same decode-step unit), print the OVERHEAD ratio.
# Target: ~1.00x (PoC nonce == chat request in cost). Run BEFORE every push.
#
#   bash benchmarks/poc/perf_gate.sh [MAX_OVERHEAD]   # default gate 1.05x
#
# Exit non-zero if overhead > MAX_OVERHEAD. Root cause of >1x is the eager
# per-step seeded-RNG dispatch flood (host stall, not GPU compute) — see KB.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH="$PWD/.venv/bin:$PATH" VLLM_POC_MIXED_DECODE=1
M=RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16
PORT=${PORT:-48931}
GATE=${1:-1.05}
LOG=$(mktemp)

reap(){ pkill -9 -f "VLLM::EngineCore" 2>/dev/null; pkill -9 -f "vllm serve.*$PORT" 2>/dev/null; }
reap; sleep 3
# --no-enable-prefix-caching: PoC nonces prefill unique seeded vectors and can never
# hit the prefix cache; off keeps the chat baseline from getting free cached prefills
# (apples-to-apples, the true uncached cost).
nohup .venv/bin/vllm serve "$M" --host 127.0.0.1 --port "$PORT" --poc-decode \
  --no-enable-prefix-caching --gpu-memory-utilization 0.85 --max-model-len 1024 > "$LOG" 2>&1 &
SRV=$!
for i in $(seq 1 90); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 3; kill -0 $SRV 2>/dev/null || { echo "SERVER DIED"; tail -15 "$LOG"; reap; exit 2; }
done

# seq_len=64 matches the reference fidelity run (runs/from_image__RTX4000__…, meta.seq_len=64).
# Do NOT drop --seq-len on EITHER line: the tool defaults to 256, so omitting it on chat gave
# chat a 4x longer prefill than PoC -> chat tok/s depressed -> overhead ratio (chat/poc)
# understated -> the gate passed PoC runs it should have failed. Both modes must prefill the
# same length (see KB: the "20% overhead" was exactly this class of apples-to-oranges).
POC=$(.venv/bin/python benchmarks/poc/perfomance_nonces.py --mode poc  --url "http://127.0.0.1:$PORT" --target vllm --model "$M" --max-tokens 256 --seq-len 64 2>/dev/null | grep -oE "steps/s=[0-9.]+" | cut -d= -f2)
CHAT=$(.venv/bin/python benchmarks/poc/perfomance_nonces.py --mode chat --url "http://127.0.0.1:$PORT"               --model "$M" --max-tokens 256 --seq-len 64 2>/dev/null | grep -oE "tokens/s=[0-9.]+" | cut -d= -f2)
kill -9 $SRV 2>/dev/null; reap

echo "commit=$(git rev-parse --short HEAD)  poc_steps_s=$POC  chat_tok_s=$CHAT"
awk -v p="$POC" -v c="$CHAT" -v g="$GATE" 'BEGIN{
  if(p==""||c==""){print "MEASURE FAIL"; exit 2}
  o=c/p; printf "overhead = %.3fx  (gate <= %.2fx, target ~1.00x)\n", o, g;
  if(o>g){print "GATE FAIL"; exit 1} else print "GATE PASS"
}'

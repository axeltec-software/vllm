#!/usr/bin/env bash
# Run perfomance_nonces.py with a fixed nonces × max_tokens sweep and save
# results to the storage folder.
#
# Environment variables
# ---------------------
# MODEL          HuggingFace model id  (default: RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16)
# SEQ_LEN        seq_len passed to PoC  (default: 256)
# SERVER_URL     Full base URL of an existing server; omit to auto-launch
# SERVER_ARGS    Extra 'vllm serve' flags used when auto-launching
#                (default: "--gpu-memory-utilization 0.9 --max-model-len 4096")
# NONCES         Space-separated n_nonces values  (default: "32 64")
# MAX_TOKENS     Space-separated max_tokens values (default: "0 32 64 128 256")
# DURATION       Seconds per config  (default: 60)
# WARMUP         Warmup requests before benchmarking  (default: 3)
#
# Examples
# --------
#   # Auto-launch server, defaults:
#   bash tests_poc/scripts/run_nonces_benchmark.sh
#
#   # Existing server:
#   SERVER_URL=http://localhost:8100 \
#       bash tests_poc/scripts/run_nonces_benchmark.sh
#
#   # Custom sweep:
#   NONCES="16 32 64" MAX_TOKENS="0 64 256" DURATION=30 \
#       bash tests_poc/scripts/run_nonces_benchmark.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
BENCH_SCRIPT="$SCRIPT_DIR/../benchmarks/perfomance_nonces.py"

MODEL="${MODEL:-RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16}"
SEQ_LEN="${SEQ_LEN:-256}"
SERVER_URL="${SERVER_URL:-}"
SERVER_ARGS="${SERVER_ARGS:---gpu-memory-utilization 0.9 --max-model-len 4096}"
NONCES="${NONCES:-32 64}"
MAX_TOKENS="${MAX_TOKENS:-0 32 64 128 256}"
DURATION="${DURATION:-60}"
WARMUP="${WARMUP:-3}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
STORAGE_DIR="$SCRIPT_DIR/../storage/nonces_benchmark"
mkdir -p "$STORAGE_DIR"

OUTPUT_JSON="$STORAGE_DIR/results_${TIMESTAMP}.json"
OUTPUT_MD="$STORAGE_DIR/results_${TIMESTAMP}.md"

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "Model      : $MODEL"
log "seq_len    : $SEQ_LEN"
log "Nonces     : $NONCES"
log "Max tokens : $MAX_TOKENS"
log "Duration   : ${DURATION}s per config"
log "Warmup     : $WARMUP requests"
log "Output     : $OUTPUT_JSON"
echo ""

SERVER_FORWARD=()
if [ -n "$SERVER_URL" ]; then
    SERVER_FORWARD=(--url "$SERVER_URL")
else
    SERVER_FORWARD=(--server-args "$SERVER_ARGS")
    log "Server     : auto-launch"
fi

read -ra NONCES_ARR    <<< "$NONCES"
read -ra MAX_TOKENS_ARR <<< "$MAX_TOKENS"

"$PYTHON" "$BENCH_SCRIPT" \
    --model "$MODEL" \
    --seq-len "$SEQ_LEN" \
    --duration "$DURATION" \
    --warmup "$WARMUP" \
    --nonces "${NONCES_ARR[@]}" \
    --max-tokens "${MAX_TOKENS_ARR[@]}" \
    --output "$OUTPUT_JSON" \
    --md-output "$OUTPUT_MD" \
    "${SERVER_FORWARD[@]}"

echo ""
echo "════════════════════════════════════════════"
echo "  Results JSON : $OUTPUT_JSON"
echo "  Results MD   : $OUTPUT_MD"
echo "════════════════════════════════════════════"

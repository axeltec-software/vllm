#!/usr/bin/env bash
# Run quality_gsm8k.py across a batch_size × poc_requests matrix.
#
# Order of runs:
#   1. Baseline (--disable_poc)
#   2. For each batch_size, for each poc_requests value
#
# Server lifecycle is fully delegated to quality_gsm8k.py:
#   - Pass SERVER_HOST + SERVER_PORT to connect to an existing server.
#   - Omit SERVER_PORT (and optionally set SERVER_ARGS) to auto-launch per run.
#
# Environment variables
# ---------------------
# MODEL          HuggingFace model id  (default: RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16)
# SERVER_HOST    Server IP / hostname  (default: 127.0.0.1)
# SERVER_PORT    Existing server port; omit to let quality_gsm8k auto-launch
# SERVER_ARGS    Extra 'vllm serve' flags used when auto-launching
#                (default: "--gpu-memory-utilization 0.5 --max-model-len 4096")
# TASKS          lm-eval task(s)       (default: gsm8k)
# BATCH_SIZES    Space-separated list  (default: "1 4 8")
# POC_REQUESTS   Space-separated list  (default: "1 2 4")
#
# Examples
# --------
#   # Auto-launch per run, defaults:
#   bash tests_poc/scripts/run_quality_test.sh
#
#   # Local existing server:
#   SERVER_PORT=8100 BATCH_SIZES="4 8" POC_REQUESTS="1 4" \
#       bash tests_poc/scripts/run_quality_test.sh
#
#   # Remote server:
#   SERVER_HOST=10.0.0.5 SERVER_PORT=8100 \
#       bash tests_poc/scripts/run_quality_test.sh
#
#   # Custom model, auto-launch:
#   MODEL=Qwen/Qwen3-0.6B SERVER_ARGS="--gpu-memory-utilization 0.3" \
#       bash tests_poc/scripts/run_quality_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
EVAL_SCRIPT="$SCRIPT_DIR/../benchmarks/quality_gsm8k.py"

MODEL="${MODEL:-RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-}"
SERVER_ARGS="${SERVER_ARGS:---gpu-memory-utilization 0.9 --max-model-len 4096}"
TASKS="${TASKS:-gsm8k}"

read -ra BATCH_SIZES  <<< "${BATCH_SIZES:-8 16}"
read -ra POC_REQUESTS <<< "${POC_REQUESTS:-1 4}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
STORAGE_DIR="$SCRIPT_DIR/../storage/eval_results"
LOG_DIR="$SCRIPT_DIR/../storage/logs/$TIMESTAMP"
mkdir -p "$STORAGE_DIR" "$LOG_DIR"

SERVER_FORWARD=()
if [ -n "$SERVER_PORT" ]; then
    SERVER_FORWARD=(--host "$SERVER_HOST" --port "$SERVER_PORT")
else
    SERVER_FORWARD=(--server-args "$SERVER_ARGS")
fi

log() { echo "[$(date +%H:%M:%S)] $*"; }

PASS=0
FAIL=0

_run() {
    local label="$1"; shift
    local log_file="$LOG_DIR/${label}.log"
    log "[$label] Starting ..."
    if "$PYTHON" "$EVAL_SCRIPT" \
            --model_name "$MODEL" \
            --output_path "$STORAGE_DIR" \
            --tasks "$TASKS" \
            "${SERVER_FORWARD[@]}" \
            "$@" 2>&1 | tee "$log_file"; then
        log "[$label] Done."
        (( PASS++ )) || true
    else
        log "[$label] FAILED."
        (( FAIL++ )) || true
    fi
}

if [ -n "$SERVER_PORT" ]; then
    SERVER_DISPLAY="$SERVER_HOST:$SERVER_PORT"
else
    SERVER_DISPLAY="auto-launch"
fi

log "Model      : $MODEL"
log "Tasks      : $TASKS"
log "Batches    : ${BATCH_SIZES[*]}"
log "PoC reqs   : ${POC_REQUESTS[*]}"
log "Server     : $SERVER_DISPLAY"
log "Output     : $STORAGE_DIR"
echo ""

_run "baseline_bs${BATCH_SIZES[0]}" \
    --batch_size "${BATCH_SIZES[0]}" \
    --disable_poc

for bs in "${BATCH_SIZES[@]}"; do
    for poc in "${POC_REQUESTS[@]}"; do
        _run "bs${bs}_poc${poc}" \
            --batch_size "$bs" \
            --poc_requests "$poc"
    done
done

echo ""
log "Regenerating comparison table ..."
"$PYTHON" "$EVAL_SCRIPT" \
    --table-only \
    --output_path "$STORAGE_DIR" \
    --table-output "$STORAGE_DIR/results_table.md"

echo ""
echo "════════════════════════════════════════════"
printf "  Runs passed : %-4s  failed : %-4s\n" "$PASS" "$FAIL"
echo "  Results     : $STORAGE_DIR/results_table.md"
echo "  Logs        : $LOG_DIR"
echo "════════════════════════════════════════════"

[ "$FAIL" -eq 0 ]

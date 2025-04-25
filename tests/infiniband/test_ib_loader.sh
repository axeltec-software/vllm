#!/bin/bash
#
# This script orchestrates the launch of two VLLM instances to test
# the 'infiniband' weight loading feature.
#
# It performs the following steps:
# 1. Checks for a minimum number of available GPUs.
# 2. Starts a sender VLLM server on GPUs 0,1 to act as the weights source.
# 3. Waits for the sender server to become healthy by polling its /health endpoint.
# 4. Starts a receiver VLLM server on GPUs 2,3, configured to load weights
#    from the sender server using the 'infiniband' method.
# 5. Waits for the receiver server to become healthy by polling its /health endpoint.
# 6. Reads logs of the receiver server looking for SUCCESS_MESSAGE.
# 7. Cleans up all background VLLM processes on exit (success, failure, or interrupt).
#

# Exit immediately if a command exits with a non-zero status.
set -e
# Ensure the cleanup function is called on script exit.
set -o pipefail

# --- Configuration ---
REQUIRED_GPUS=4
SERVER1_PORT=8000
SERVER2_PORT=8001
LOG_DIR="/tmp/vllm_logs"
SERVER1_LOG="$LOG_DIR/server1.log"
SERVER2_LOG="$LOG_DIR/server2.log"
HEALTH_CHECK_TIMEOUT_SECONDS=180
HEALTH_CHECK_INTERVAL_SECONDS=5
SUCCESS_MESSAGE="Using weight source override:"

# --- Cleanup Function ---
# This function is automatically called when the script exits, ensuring no
# orphaned processes are left behind.
cleanup() {
    echo "--- Cleaning up background processes ---"
    # Use pkill to find and kill the processes by a unique part of their command line.
    # The `|| true` prevents the script from failing if the process is already gone.
    pkill -f "vllm.entrypoints.openai.api_server.*--port $SERVER1_PORT" || true
    pkill -f "vllm.entrypoints.openai.api_server.*--port $SERVER2_PORT" || true
    echo "Cleanup complete."
}

# Register the cleanup function to be called on the EXIT signal.
trap cleanup EXIT

echo "--- 1. Checking for available GPUs ---"
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -n 1 || echo "0")
if [ "$GPU_COUNT" -lt "$REQUIRED_GPUS" ]; then
    echo "❌ FAILURE: Insufficient GPUs. Required: $REQUIRED_GPUS, Found: $GPU_COUNT"
    exit 1
fi
echo "Found $GPU_COUNT GPUs. Proceeding."

mkdir -p "$LOG_DIR"
echo "Logs will be stored in $LOG_DIR"

echo "--- 2. Starting sender VLLM server on port $SERVER1_PORT (GPUs 0,1) ---"
VLLM_LOGGING_LEVEL="DEBUG" NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 python3 -m vllm.entrypoints.openai.api_server \
    --model neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 \
    --tensor-parallel-size 2 \
    --enforce-eager \
    --host 0.0.0.0 \
    --port $SERVER1_PORT > "$SERVER1_LOG" 2>&1 &
echo "Sender server started. Logs are being written to $SERVER1_LOG"

echo "--- 3. Waiting for sender server to become healthy (timeout: ${HEALTH_CHECK_TIMEOUT_SECONDS}s) ---"
start_time=$(date +%s)
while true; do
    if curl --silent --fail "http://0.0.0.0:$SERVER1_PORT/health"; then
        current_time=$(date +%s)
        elapsed=$((current_time - start_time))
        echo "Sender server is healthy after $elapsed seconds."
        break
    fi

    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    if [ $elapsed -ge $HEALTH_CHECK_TIMEOUT_SECONDS ]; then
        echo "❌ FAILURE: Timeout waiting for sender server to become healthy."
        echo "--- Last 50 lines from sender server log ($SERVER1_LOG): ---"
        tail -n 50 "$SERVER1_LOG"
        echo "--------------------------------------------------------"
        exit 1
    fi
    sleep $HEALTH_CHECK_INTERVAL_SECONDS
done

echo "--- 4. Starting receiver VLLM server on port $SERVER2_PORT (GPUs 2,3) ---"
VLLM_LOGGING_LEVEL="DEBUG" NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=2,3 python3 -m vllm.entrypoints.openai.api_server \
    --load-format infiniband \
    --tensor-parallel-size 2 \
    --model neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 \
    --model-loader-extra-config "{\"weights_source_override\": \"http://0.0.0.0:$SERVER1_PORT\"}" \
    --enforce-eager \
    --port $SERVER2_PORT > "$SERVER2_LOG" 2>&1 &

echo "Receiver server started. Logs are being written to $SERVER2_LOG"

echo "--- 5. Waiting for receiver server to become healthy (timeout: ${HEALTH_CHECK_TIMEOUT_SECONDS}s) ---"
start_time=$(date +%s)
while true; do
    if curl --silent --fail "http://localhost:$SERVER2_PORT/health"; then
        current_time=$(date +%s)
        elapsed=$((current_time - start_time))
        echo "Receiver server is healthy after $elapsed seconds."
        break
    fi

    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    if [ $elapsed -ge $HEALTH_CHECK_TIMEOUT_SECONDS ]; then
        echo "❌ FAILURE: Timeout waiting for receiver server to become healthy."
        echo "--- Last 50 lines from receiver server log ($SERVER2_LOG): ---"
        tail -n 50 "$SERVER2_LOG"
        echo "--------------------------------------------------------"
        exit 1
    fi
    sleep $HEALTH_CHECK_INTERVAL_SECONDS
done

echo "--- 6. Checking logs for Infiniband confirmation (timeout: ${LOG_CHECK_TIMEOUT_SECONDS}s) ---"
if [ -s "$SERVER2_LOG" ] && grep -q "$SUCCESS_MESSAGE" "$SERVER2_LOG"; then
    echo "✅ SUCCESS: Infiniband weights discovery confirmed in logs of receiver server."
    grep "$SUCCESS_MESSAGE" "$SERVER2_LOG"
    exit 0
else
    echo "❌ FAILURE: Infiniband weights discovery not found in logs of receiver server."
    echo "----------------------------------------------------"
fi

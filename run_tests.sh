#!/bin/bash

export HUGGING_FACE_HUB_TOKEN=""

DRAFT_DIR="$(cd "$(dirname "$0")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$DRAFT_DIR/test_logs/$TIMESTAMP"
PYTEST="$DRAFT_DIR/.venv/bin/pytest"
TESTS_DIR="$DRAFT_DIR/tests"

mkdir -p "$LOG_DIR"

SUMMARY_LOG="$LOG_DIR/summary.log"
RESULTS_FILE="$LOG_DIR/results.csv"

# CSV header
echo "folder,status,passed,failed,errors,skipped,exit_code,duration_s,log_file" > "$RESULTS_FILE"

TOTAL_PASSED=0
TOTAL_FAILED=0
TOTAL_ERRORS=0
TOTAL_SKIPPED=0

echo "Test run started at $(date)" | tee "$SUMMARY_LOG"
echo "Logs: $LOG_DIR" | tee -a "$SUMMARY_LOG"
printf "%-30s %-10s %7s %7s %7s %7s  %s\n" "FOLDER" "STATUS" "PASSED" "FAILED" "ERRORS" "SKIPPED" "LOG" | tee -a "$SUMMARY_LOG"
echo "$(printf '%0.s-' {1..100})" | tee -a "$SUMMARY_LOG"

run_tests() {
    local label="$1"
    local target="$2"
    local log_file="$LOG_DIR/${label}.log"

    timeout 600 "$PYTEST" $target -v --tb=short --no-header -q 2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}

    # Parse the pytest summary line, e.g.: "1 failed, 7 passed, 4 warnings in 53.06s"
    local summary_line
    summary_line=$(grep -E "^=+ .*(passed|failed|error)" "$log_file" | tail -1)

    local passed=0 failed=0 errors=0 skipped=0 duration=0

    [[ "$summary_line" =~ ([0-9]+)\ passed    ]] && passed="${BASH_REMATCH[1]}"
    [[ "$summary_line" =~ ([0-9]+)\ failed    ]] && failed="${BASH_REMATCH[1]}"
    [[ "$summary_line" =~ ([0-9]+)\ error     ]] && errors="${BASH_REMATCH[1]}"
    [[ "$summary_line" =~ ([0-9]+)\ skipped   ]] && skipped="${BASH_REMATCH[1]}"
    [[ "$summary_line" =~ in\ ([0-9]+\.[0-9]+)s ]] && duration="${BASH_REMATCH[1]}"

    local status
    if   [ $exit_code -eq 124 ];                    then status="TIMEOUT"
    elif grep -q "Segmentation fault\|dumped core" "$log_file"; then status="SEGFAULT"
    elif [ $exit_code -eq 0 ];                      then status="PASSED"
    elif [ "$passed" -eq 0 ] && [ "$failed" -eq 0 ] && [ "$errors" -eq 0 ]; then status="NO_TESTS"
    else                                                  status="FAILED"
    fi

    TOTAL_PASSED=$(( TOTAL_PASSED + passed ))
    TOTAL_FAILED=$(( TOTAL_FAILED + failed ))
    TOTAL_ERRORS=$(( TOTAL_ERRORS + errors ))
    TOTAL_SKIPPED=$(( TOTAL_SKIPPED + skipped ))

    echo "$label,$status,$passed,$failed,$errors,$skipped,$exit_code,$duration,$log_file" >> "$RESULTS_FILE"

    printf "%-30s %-10s %7s %7s %7s %7s  %s\n" \
        "$label" "$status" "$passed" "$failed" "$errors" "$skipped" "$(basename "$log_file")" \
        | tee -a "$SUMMARY_LOG"
}

run_tests "top_level_files" "$TESTS_DIR/test_*.py"

# Subdirectories
for dir in \
    basic_correctness \
    benchmarks \
    compile \
    config \
    cuda \
    detokenizer \
    distributed \
    engine \
    entrypoints \
    evals \
    kernels \
    kv_transfer \
    lora \
    model_executor \
    models \
    multimodal \
    plugins \
    plugins_tests \
    quantization \
    reasoning \
    samplers \
    standalone_tests \
    tokenization \
    tools \
    tool_use \
    tpu \
    transformers_utils \
    utils_ \
    v1 \
    vllm_test_utils \
    weight_loading
do
    if [ -d "$TESTS_DIR/$dir" ]; then
        run_tests "$dir" "$TESTS_DIR/$dir"
    fi
done

echo "$(printf '%0.s-' {1..100})" | tee -a "$SUMMARY_LOG"
printf "%-30s %-10s %7s %7s %7s %7s\n" \
    "TOTAL" "" "$TOTAL_PASSED" "$TOTAL_FAILED" "$TOTAL_ERRORS" "$TOTAL_SKIPPED" \
    | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"
echo "Test run finished at $(date)" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"
echo "Human-readable summary : $SUMMARY_LOG"
echo "Parseable CSV results  : $RESULTS_FILE"

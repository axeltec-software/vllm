#!/bin/bash

export HUGGING_FACE_HUB_TOKEN=""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VLLM_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TESTS_DIR="$VLLM_DIR/tests"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$TESTS_DIR/storage/blocking_tests/$TIMESTAMP"
BLOCKING_TESTS="$SCRIPT_DIR/blocking_tests.txt"
export PATH="$VLLM_DIR/.venv/bin:$PATH"

mkdir -p "$LOG_DIR"

SUMMARY_LOG="$LOG_DIR/summary.log"
RESULTS_FILE="$LOG_DIR/results.csv"

echo "label,command,status,passed,failed,errors,skipped,exit_code,duration_s,log_file" > "$RESULTS_FILE"

TOTAL_PASSED=0
TOTAL_FAILED=0
TOTAL_ERRORS=0
TOTAL_SKIPPED=0

echo "Blocking test run started at $(date)" | tee "$SUMMARY_LOG"
echo "Logs: $LOG_DIR" | tee -a "$SUMMARY_LOG"
printf "%-45s %-10s %7s %7s %7s %7s  %s\n" "LABEL" "STATUS" "PASSED" "FAILED" "ERRORS" "SKIPPED" "LOG" | tee -a "$SUMMARY_LOG"
echo "$(printf '%0.s-' {1..110})" | tee -a "$SUMMARY_LOG"

run_test() {
    local label="$1"
    local cmd="$2"
    local safe_label
    safe_label="${label}__$(echo "$cmd" | md5sum | cut -c1-6)"
    local log_file="$LOG_DIR/${safe_label}.log"

    local tmp_out
    tmp_out=$(mktemp)
    timeout 600 bash -c "cd '$TESTS_DIR' && $cmd" > "$tmp_out" 2>&1
    local exit_code=$?

    # Log file: failures, errors, assertion lines, and summary — skip PASSED/code context
    grep -E "^(FAILED|ERROR|E   |E$|short test summary|=+)" "$tmp_out" > "$log_file" 2>/dev/null || true
    rm -f "$tmp_out"

    local summary_line
    summary_line=$(grep -E "^=+ .*(passed|failed|error|warning)" "$log_file" | tail -1)

    local passed=0 failed=0 errors=0 skipped=0 duration=0

    [[ "$summary_line" =~ ([0-9]+)\ passed    ]] && passed="${BASH_REMATCH[1]}"
    [[ "$summary_line" =~ ([0-9]+)\ failed    ]] && failed="${BASH_REMATCH[1]}"
    [[ "$summary_line" =~ ([0-9]+)\ error     ]] && errors="${BASH_REMATCH[1]}"
    [[ "$summary_line" =~ ([0-9]+)\ skipped   ]] && skipped="${BASH_REMATCH[1]}"
    [[ "$summary_line" =~ in\ ([0-9]+\.[0-9]+)s ]] && duration="${BASH_REMATCH[1]}"

    local status
    if   [ "$exit_code" -eq 124 ];                                                then status="TIMEOUT"
    elif grep -q "Segmentation fault\|dumped core" "$log_file";                   then status="SEGFAULT"
    elif [ "$exit_code" -eq 0 ];                                                  then status="PASSED"
    elif [ "$passed" -eq 0 ] && [ "$failed" -eq 0 ] && [ "$errors" -eq 0 ];      then status="NO_TESTS"
    else                                                                               status="FAILED"
    fi

    TOTAL_PASSED=$(( TOTAL_PASSED + passed ))
    TOTAL_FAILED=$(( TOTAL_FAILED + failed ))
    TOTAL_ERRORS=$(( TOTAL_ERRORS + errors ))
    TOTAL_SKIPPED=$(( TOTAL_SKIPPED + skipped ))

    echo "$label,\"$cmd\",$status,$passed,$failed,$errors,$skipped,$exit_code,$duration,$log_file" >> "$RESULTS_FILE"

    # Full table row to summary log only
    printf "%-45s %-10s %7s %7s %7s %7s  %s\n" \
        "${label:0:44}" "$status" "$passed" "$failed" "$errors" "$skipped" "$(basename "$log_file")" \
        >> "$SUMMARY_LOG"

    printf "%-45s %-10s %7s %7s %7s %7s  %s\n" \
        "${label:0:44}" "$status" "$passed" "$failed" "$errors" "$skipped" "$(basename "$log_file")"
}

while IFS=$'\t' read -r label cmd; do
    [[ -z "$label" || "$label" == \#* ]] && continue
    run_test "$label" "$cmd"
done < "$BLOCKING_TESTS"

echo "$(printf '%0.s-' {1..110})" | tee -a "$SUMMARY_LOG"
printf "TOTAL  passed=%-6s failed=%-6s errors=%-6s skipped=%s\n" \
    "$TOTAL_PASSED" "$TOTAL_FAILED" "$TOTAL_ERRORS" "$TOTAL_SKIPPED" \
    | tee -a "$SUMMARY_LOG"
printf "%-45s %-10s %7s %7s %7s %7s\n" \
    "TOTAL" "" "$TOTAL_PASSED" "$TOTAL_FAILED" "$TOTAL_ERRORS" "$TOTAL_SKIPPED" \
    >> "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"
echo "Blocking test run finished at $(date)" | tee -a "$SUMMARY_LOG"
echo "Human-readable summary : $SUMMARY_LOG"
echo "Parseable CSV results  : $RESULTS_FILE"

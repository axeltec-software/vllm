#!/usr/bin/env bash
# Run all PoC tests under tests_poc/tests/.
#
# Unit tests (no server) and integration tests (each file launches its own
# PoCTestServer automatically) are run as separate suites.
#
# Examples
# --------
#   bash tests_poc/scripts/run_poc_tests.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TESTS_POC_DIR="$SCRIPT_DIR/.."
PYTEST="$ROOT_DIR/.venv/bin/pytest"
TESTS_DIR="$TESTS_POC_DIR/tests"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$TESTS_POC_DIR/storage/logs/poc_tests/$TIMESTAMP"
mkdir -p "$LOG_DIR"

SUMMARY_LOG="$LOG_DIR/summary.log"
RESULTS_FILE="$LOG_DIR/results.csv"

echo "label,status,passed,failed,errors,skipped,exit_code,duration_s,log_file" > "$RESULTS_FILE"

TOTAL_PASSED=0
TOTAL_FAILED=0
TOTAL_ERRORS=0
TOTAL_SKIPPED=0

echo "PoC test run started at $(date)" | tee "$SUMMARY_LOG"
echo "Tests : $TESTS_DIR" | tee -a "$SUMMARY_LOG"
echo "Logs  : $LOG_DIR" | tee -a "$SUMMARY_LOG"
printf "%-30s %-10s %7s %7s %7s %7s  %s\n" "LABEL" "STATUS" "PASSED" "FAILED" "ERRORS" "SKIPPED" "LOG" | tee -a "$SUMMARY_LOG"
echo "$(printf '%0.s-' {1..100})" | tee -a "$SUMMARY_LOG"

run_suite() {
    local label="$1"
    shift
    local extra_args=("$@")
    local log_file="$LOG_DIR/${label}.log"

    set +e
    timeout 1800 "$PYTEST" "$TESTS_DIR" -v --tb=long "${extra_args[@]}" > "$log_file" 2>&1
    local exit_code=$?
    set -e

    local summary_line
    summary_line=$(grep -E "^=+ .*(passed|failed|error|warning)" "$log_file" | tail -1)

    local passed=0 failed=0 errors=0 skipped=0 duration=0
    [[ "$summary_line" =~ ([0-9]+)\ passed  ]] && passed="${BASH_REMATCH[1]}"
    [[ "$summary_line" =~ ([0-9]+)\ failed  ]] && failed="${BASH_REMATCH[1]}"
    [[ "$summary_line" =~ ([0-9]+)\ error   ]] && errors="${BASH_REMATCH[1]}"
    [[ "$summary_line" =~ ([0-9]+)\ skipped ]] && skipped="${BASH_REMATCH[1]}"
    [[ "$summary_line" =~ in\ ([0-9]+\.[0-9]+)s ]] && duration="${BASH_REMATCH[1]}"

    local status
    if   [ "$exit_code" -eq 124 ];                                              then status="TIMEOUT"
    elif grep -q "Segmentation fault\|dumped core" "$log_file" 2>/dev/null;     then status="SEGFAULT"
    elif [ "$exit_code" -eq 0 ];                                                then status="PASSED"
    elif [ "$passed" -eq 0 ] && [ "$failed" -eq 0 ] && [ "$errors" -eq 0 ];    then status="NO_TESTS"
    else                                                                             status="FAILED"
    fi

    TOTAL_PASSED=$(( TOTAL_PASSED + passed ))
    TOTAL_FAILED=$(( TOTAL_FAILED + failed ))
    TOTAL_ERRORS=$(( TOTAL_ERRORS + errors ))
    TOTAL_SKIPPED=$(( TOTAL_SKIPPED + skipped ))

    echo "$label,$status,$passed,$failed,$errors,$skipped,$exit_code,$duration,$log_file" >> "$RESULTS_FILE"

    printf "%-30s %-10s %7s %7s %7s %7s  %s\n" \
        "${label:0:29}" "$status" "$passed" "$failed" "$errors" "$skipped" "$(basename "$log_file")" \
        | tee -a "$SUMMARY_LOG"

    return "$exit_code"
}

OVERALL_EXIT=0

run_suite "unit"        -m "not integration" || OVERALL_EXIT=1
run_suite "integration" -m integration       || OVERALL_EXIT=1

echo "$(printf '%0.s-' {1..100})" | tee -a "$SUMMARY_LOG"
printf "%-30s %-10s %7s %7s %7s %7s\n" \
    "TOTAL" "" "$TOTAL_PASSED" "$TOTAL_FAILED" "$TOTAL_ERRORS" "$TOTAL_SKIPPED" \
    | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"
echo "PoC test run finished at $(date)" | tee -a "$SUMMARY_LOG"
echo "Human-readable summary : $SUMMARY_LOG"
echo "Parseable CSV          : $RESULTS_FILE"

exit "$OVERALL_EXIT"

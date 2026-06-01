# PoC Test Suite

## Structure

```
tests_poc/
├── tests/
│   ├── test_all_modes.py        # Chat, PoC, mixed batches, concurrency, truncation
│   ├── test_blocking.py         # wait/blocking params, exclusive mode
│   ├── test_chat_quality.py     # Chat output integrity under PoC load
│   ├── unit/
│   │   ├── test_layer_hook_math.py   # Householder reflection math, hook lifecycle
│   │   └── test_distributed_poc.py  # Sphere codebook, TP/PP interface contracts
│   └── integration/
│       ├── test_artifact_validity.py    # Vector content: norm, finiteness, determinism
│       ├── test_kv_cache_integrity.py   # KV cache not corrupted by PoC
│       └── test_poc_decode.py           # k_points_steps length, range, entropy
├── benchmarks/
│   ├── perfomance_nonces.py     # Nonces/s throughput sweep
│   ├── quality_gsm8k.py         # lm-eval accuracy under concurrent PoC
│   └── poc_validation.py        # Script to collect k mismatches
└── scripts/
    ├── run_poc_tests.sh          
    ├── run_blocking_tests.sh     # Run vLLM blocking test suite from blocking_tests.txt
    ├── run_quality_test.sh       # Run quality_gsm8k across batch_size × poc_requests
    ├── run_nonces_benchmark.sh   # Run perfomance_nonces with configured sweep
    ├── blocking_tests.txt        
    └── collect_validation.sh     # Collect artifacts from poc_validation.py 
```

Logs and benchmark results are written to `storage/`.


## Running Tests

### All PoC tests (unit + integration)

```bash
bash tests_poc/scripts/run_poc_tests.sh
```

Console shows a compact summary table; full output (tracebacks included) goes to
`tests_poc/storage/logs/poc_tests/<timestamp>/`.


## Benchmarks

### Nonces/s throughput sweep

```bash
# Auto-launch server:
bash tests_poc/scripts/run_nonces_benchmark.sh

# Existing server:
SERVER_URL=http://localhost:8100 bash tests_poc/scripts/run_nonces_benchmark.sh

# Custom sweep:
NONCES="16 32 64" MAX_TOKENS="0 64 256" DURATION=30 \
    bash tests_poc/scripts/run_nonces_benchmark.sh
```

Results saved to `tests_poc/storage/nonces_benchmark/results_<timestamp>.{json,md}`.

### GSM8k quality under PoC load

```bash
# Auto-launch server:
bash tests_poc/scripts/run_quality_test.sh

# Existing server:
SERVER_PORT=8100 bash tests_poc/scripts/run_quality_test.sh

# Custom sweep:
SERVER_PORT=8100 BATCH_SIZES="4 8" POC_REQUESTS="1 4" \
    bash tests_poc/scripts/run_quality_test.sh
```

Runs a baseline (no PoC) then a matrix of `BATCH_SIZES × POC_REQUESTS` combinations.
Results and a comparison table are saved to `tests_poc/storage/eval_results/`.


## Blocking Tests (vLLM CI suite)


```bash
uv pip install -r requirements/test.in --no-build-isolation
bash tests_poc/scripts/run_blocking_tests.sh
```

Test commands are read from `scripts/blocking_tests.txt`.
Console shows one row per test; errors go to per-test log files under
`tests_poc/scripts/test_logs/blocking_tests/<timestamp>/`.


## PoC-decode (k mismatches rate)

To collect mismatched data for PoC decode:

```
bash scripts/collect_validation.sh --honest
bash scripts/collect_validation.sh --fraud
```

Use `notebooks/validation.ipynb` to plot the results. 
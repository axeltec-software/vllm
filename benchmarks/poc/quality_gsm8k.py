import argparse
import asyncio
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import aiohttp
import numpy as np


async def _send_poc_request(
    base_url: str,
    model: str,
    block_hash: str,
    nonces: list[int],
    public_key: str = "test_node",
    max_tokens: int = 0,
) -> tuple[dict[str, Any] | None, float]:
    """Send one PoC generate request and return (result, elapsed_seconds).

    max_tokens > 0 runs decode PoC (the proposal's purpose); 0 = prefill-only.
    Proposal API: max_tokens lives under params.
    """
    url = f"{base_url}/api/v1/pow/generate"
    payload = {
        "block_hash": block_hash,
        "block_height": 100,
        "public_key": public_key,
        "node_id": 0,
        "node_count": 1,
        "nonces": nonces,
        "params": {"model": model, "seq_len": 256, "k_dim": 12,
                   "max_tokens": max_tokens},
        "wait": True,
    }
    t0 = time.time()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                result = await resp.json() if resp.status == 200 else {"error": f"status {resp.status}"}
    except Exception as exc:
        result = {"error": str(exc)}
    return result, time.time() - t0


async def _poc_sender_loop(
    base_url: str,
    model: str,
    interval_seconds: float,
    requests_per_interval: int,
    stop_event: asyncio.Event,
    poc_artifacts: list[dict[str, Any]],
    poc_times: list[float],
    max_tokens: int = 0,
    nonces_per_request: int = 32,
) -> None:
    """Continuously send PoC requests until *stop_event* is set.

    Each request carries ``nonces_per_request`` nonces (prod PoC batch = 32),
    and ``requests_per_interval`` such requests fire concurrently per interval.
    """
    nonce_counter = 0
    request_count = 0
    print(
        f"[PoC] Starting "
        f"(interval={interval_seconds}s, {requests_per_interval} req/interval)\n"
    )

    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
            break
        except asyncio.TimeoutError:
            pass

        batch_start = request_count + 1
        tasks = []
        for _ in range(requests_per_interval):
            nonces = list(range(nonce_counter, nonce_counter + nonces_per_request))
            nonce_counter += nonces_per_request
            request_count += 1
            tasks.append((request_count, f"block_{request_count}", nonces))

        print(f"[PoC] Sending batch #{batch_start}–{request_count}")
        results = await asyncio.gather(*[
            _send_poc_request(base_url, model, block_hash, nonces,
                              max_tokens=max_tokens)
            for _, block_hash, nonces in tasks
        ])

        for (req_id, block_hash, nonces), (result, elapsed) in zip(tasks, results):
            poc_artifacts.append({
                "request_id": req_id,
                "timestamp": time.time(),
                "block_hash": block_hash,
                "nonces": nonces,
                "result": result,
                "elapsed_time": elapsed,
            })
            poc_times.append(elapsed)

        success = sum(1 for r, _ in results if r and "error" not in r)
        print(f"[PoC] Batch done: {success}/{len(results)} OK\n")

    print(f"[PoC] Stopped: {len(poc_artifacts)} requests sent")


async def _stream_process_output(process: subprocess.Popen, log_file: Path) -> int:
    """Stream stdout of *process* to console and *log_file*; return exit code."""
    with open(log_file, "w", buffering=1) as log_f:
        while True:
            return_code = process.poll()
            line = process.stdout.readline()
            if line:
                print(line, end="", flush=True)
                log_f.write(line)
            if return_code is not None:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log_f.write(line)
                break
            await asyncio.sleep(0.01)
    return return_code


def _parse_gsm8k_results(output_path: Path, since: float) -> dict[str, Any] | None:
    """Find the most recent gsm8k results JSON written after *since*."""
    candidates = [
        f
        for d in output_path.glob("*/")
        for f in d.glob("results_*.json")
        if f.stat().st_mtime >= since
    ]
    if not candidates:
        return None

    results_file = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        with open(results_file) as f:
            data = json.load(f)
        gsm8k = data.get("results", {}).get("gsm8k", {})
        if gsm8k:
            return {
                "strict_match": gsm8k.get("exact_match,strict-match"),
                "flexible_extract": gsm8k.get("exact_match,flexible-extract"),
                "results_file": str(results_file),
            }
    except Exception as exc:
        print(f"Error parsing gsm8k results: {exc}")
    return None


def _parse_run_stats(stats_file: Path) -> dict[str, Any]:
    """Extract display-ready fields from a ``run_stats.json`` file."""
    with open(stats_file) as f:
        data = json.load(f)

    model = data.get("model_name", "Unknown")
    if "/" in model:
        model = model.split("/")[-1]

    folder = stats_file.parent.name
    # poc_requests/poc_max_tokens come from run_stats.json (robust); fall back to
    # parsing the folder name (chat_{bs}_poc_{n}_mt_{m}) for older runs.
    poc_requests = data.get("poc_requests")
    if poc_requests is None:
        poc_requests = 0
        if folder.startswith("chat_") and "_poc_" in folder:
            parts = folder.split("_")
            if len(parts) >= 4:
                poc_requests = int(parts[3])

    gsm8k = data.get("gsm8k") or {}
    return {
        "model": model,
        "batch_size": data.get("batch_size", 0),
        "poc_requests": poc_requests,
        "poc_max_tokens": data.get("poc_max_tokens", 0),
        "strict_match": gsm8k.get("strict_match", 0.0),
        "flexible_extract": gsm8k.get("flexible_extract", 0.0),
        "elapsed": data.get("elapsed_seconds", 0),
        "median_time": data.get("poc_median_time"),
        "folder": folder,
    }


def generate_table(eval_results_dir: Path, output_file: Path | None = None) -> None:
    """Collect all ``run_stats.json`` files under *eval_results_dir* and write a Markdown table.

    Rows are sorted by (model, batch_size, poc_requests). When *output_file* is
    given the table is written there; otherwise it is printed to stdout.
    """
    rows: list[dict[str, Any]] = []
    for folder in eval_results_dir.iterdir():
        if not folder.is_dir() or not folder.name.startswith("chat_"):
            continue
        stats_file = folder / "run_stats.json"
        if not stats_file.exists():
            continue
        try:
            rows.append(_parse_run_stats(stats_file))
        except Exception as exc:
            print(f"Warning: could not parse {stats_file}: {exc}", file=sys.stderr)

    if not rows:
        print("No results found.", file=sys.stderr)
        return

    rows.sort(key=lambda r: (r["model"], r["batch_size"], r["poc_requests"]))

    header = "| Configuration | Batch Size | PoC Batch | Decode (max_tokens) | Accuracy (Strict/Flexible) | Time gsm8k (s) | Median Time PoC (s) |"
    sep    = "|---------------|------------|-----------|---------------------|----------------------------|----------------|---------------------|"
    lines  = [header, sep]
    for r in rows:
        median_str = f"{r['median_time']:.3f}" if r["median_time"] is not None else ""
        lines.append(
            f"| {r['model']:<13} | {r['batch_size']:<10} | {r['poc_requests']:<9} "
            f"| {r['poc_max_tokens']:<19} "
            f"| {r['strict_match']:.4f} / {r['flexible_extract']:.4f}          "
            f"| {int(r['elapsed']):<14} | {median_str:<19} |"
        )

    table = "\n".join(lines) + "\n"
    if output_file:
        output_file.write_text(table)
        print(f"Table saved to: {output_file}")
    else:
        print(table)


async def _run_eval(args: argparse.Namespace) -> int:
    """Execute one lm-eval run and collect PoC timing data."""
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    poc_requests = 0 if args.disable_poc else args.poc_requests
    poc_max_tokens = 0 if args.disable_poc else args.max_tokens
    # include max_tokens so decode vs prefill runs (same bs/poc) don't collide
    run_name = f"chat_{args.batch_size}_poc_{poc_requests}_mt_{poc_max_tokens}"
    run_output_path = output_path / run_name
    run_output_path.mkdir(parents=True, exist_ok=True)

    server_url = f"http://{args.host}:{args.port}"

    print("=" * 80)
    print(f"Model     : {args.model_name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Server    : {server_url}")
    if args.disable_poc:
        print("PoC       : Disabled")
    else:
        print(f"PoC       : interval={args.poc_interval}s, {args.poc_requests} req/interval")
    print(f"Output    : {run_output_path}")
    print("=" * 80)
    print()

    cmd = [
        "lm-eval",
        "--model", "local-chat-completions",
        "--model_args", f"model={args.model_name},base_url={server_url}/v1/chat/completions,num_concurrent={args.batch_size}",
        "--tasks", args.tasks,
        "--output_path", str(output_path),
        "--log_samples",
        "--apply_chat_template",
    ]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]

    print("Starting lm_eval...")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    start_time = time.time()

    poc_artifacts: list[dict[str, Any]] = []
    poc_times: list[float] = []
    stop_event = asyncio.Event()

    poc_task = None
    if not args.disable_poc:
        poc_task = asyncio.create_task(
            _poc_sender_loop(
                server_url,
                args.model_name,
                args.poc_interval,
                args.poc_requests,
                stop_event,
                poc_artifacts,
                poc_times,
                args.max_tokens,
                args.poc_nonces,
            )
        )

    try:
        return_code = await _stream_process_output(process, run_output_path / "lm_eval.log")
        elapsed = time.time() - start_time

        if poc_task:
            stop_event.set()
            await poc_task

        print()
        print("=" * 80)
        print(f"Completed in {elapsed:.1f}s ({elapsed / 60:.1f}min)")
        print("=" * 80)
        print()

        run_stats: dict[str, Any] = {
            "model_name": args.model_name,
            "batch_size": args.batch_size,
            "server_host": args.host,
            "server_port": args.port,
            "tasks": args.tasks,
            "elapsed_seconds": elapsed,
            "return_code": return_code,
            "poc_enabled": not args.disable_poc,
            "poc_requests": 0 if args.disable_poc else args.poc_requests,
            "poc_nonces": 0 if args.disable_poc else args.poc_nonces,
            "poc_max_tokens": 0 if args.disable_poc else args.max_tokens,
        }

        if return_code == 0:
            gsm8k = _parse_gsm8k_results(output_path, start_time)
            if gsm8k:
                run_stats["gsm8k"] = gsm8k
                print(f"GSM8K  strict:   {gsm8k['strict_match']:.4f}")
                print(f"GSM8K  flexible: {gsm8k['flexible_extract']:.4f}")
                print(f"       file:     {gsm8k['results_file']}")
                print()

        if poc_artifacts:
            run_stats["poc_requests_sent"] = len(poc_artifacts)
            run_stats["poc_requests_successful"] = sum(
                1 for a in poc_artifacts if "error" not in a["result"]
            )
            if poc_times:
                run_stats["poc_median_time"] = float(np.median(poc_times))
                run_stats["poc_mean_time"] = float(np.mean(poc_times))
                run_stats["poc_min_time"] = float(np.min(poc_times))
                run_stats["poc_max_time"] = float(np.max(poc_times))
                print(f"PoC sent:        {run_stats['poc_requests_sent']}")
                print(f"PoC successful:  {run_stats['poc_requests_successful']}")
                print(f"PoC median time: {run_stats['poc_median_time']:.3f}s")
                print()

            artifacts_file = run_output_path / "poc_artifacts.json"
            artifacts_file.write_text(json.dumps(poc_artifacts, indent=2))
            print(f"PoC artifacts saved to: {artifacts_file}")

        stats_file = run_output_path / "run_stats.json"
        stats_file.write_text(json.dumps(run_stats, indent=2))
        print(f"Run stats saved to: {stats_file}")

        table_path = Path(args.table_output) if args.table_output else output_path / "results_table.md"
        generate_table(output_path, table_path)

        return 0 if return_code == 0 else 1

    except KeyboardInterrupt:
        print("\nInterrupted!")
        process.terminate()
        if poc_task:
            stop_event.set()
            await poc_task
        return 130


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run lm-eval with concurrent PoC requests and generate a comparison table."
    )
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument(
        "--host",
        default=None,
        metavar="IP",
        help="Server host / IP (default 127.0.0.1). Requires --port.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port. Provide with --host to use an existing server; omit to auto-launch.",
    )
    parser.add_argument("--output_path", type=str, default="./eval_results")
    parser.add_argument("--tasks", type=str, default="gsm8k")
    parser.add_argument("--poc_interval", type=float, default=1.0)
    parser.add_argument("--poc_requests", type=int, default=1,
                        help="Concurrent PoC requests fired per interval.")
    parser.add_argument("--poc_nonces", type=int, default=32,
                        help="Nonces per PoC request (prod PoC batch = 32).")
    parser.add_argument(
        "--max_tokens", type=int, default=256,
        help="PoC decode steps during the run (256 = decode PoC, the proposal's "
             "purpose; 0 = prefill-only). Requires the server started with --poc-decode.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit gsm8k to first N questions (fast experiments). Omit = full set.",
    )
    parser.add_argument("--disable_poc", action="store_true")
    parser.add_argument(
        "--server-args",
        default="",
        metavar="ARGS",
        help=(
            "Extra args forwarded to 'vllm serve' when auto-launching "
            "(quoted string, e.g. \"--gpu-memory-utilization 0.5 --max-model-len 4096\")"
        ),
    )
    parser.add_argument(
        "--table-output",
        metavar="FILE",
        default=None,
        help="Path for the Markdown comparison table (default: {output_path}/results_table.md)",
    )
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="Skip the eval run; only regenerate the comparison table from existing results.",
    )
    args = parser.parse_args()

    if args.table_only:
        output_path = Path(args.output_path)
        if not output_path.exists():
            print(f"Error: {output_path} does not exist.", file=sys.stderr)
            return 1
        table_path = Path(args.table_output) if args.table_output else output_path / "results_table.md"
        generate_table(output_path, table_path)
        return 0

    if not args.model_name or args.batch_size is None:
        parser.error("--model_name and --batch_size are required unless --table-only is set.")

    if args.host is not None and args.port is None:
        parser.error("--host requires --port.")

    if args.port is not None:
        args.host = args.host or "127.0.0.1"
        return asyncio.run(_run_eval(args))

    from tests.poc._server import PoCTestServer

    extra = shlex.split(args.server_args) if args.server_args else []
    default_args = ["--no-enable-prefix-caching"]
    with PoCTestServer(args.model_name, extra + default_args) as srv:
        args.host = srv.host
        args.port = srv.port
        return asyncio.run(_run_eval(args))


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
import argparse
import asyncio
import subprocess
import time
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import aiohttp
import numpy as np
import os


async def send_poc_request(
    base_url: str,
    model: str,
    block_hash: str,
    nonces: List[int],
    public_key: str = "test_node",
) -> tuple[Optional[Dict[str, Any]], float]:
    url = f"{base_url}/api/v1/pow/generate"
    payload = {
        "block_hash": block_hash,
        "block_height": 100,
        "public_key": public_key,
        "node_id": 0,
        "node_count": 1,
        "nonces": nonces,
        "params": {
            "model": model,
            "seq_len": 256,
            "k_dim": 12,
        },
        "wait": True,
    }
    
    start_time = time.time()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status == 200:
                    result = await resp.json()
                else:
                    result = {"error": f"status {resp.status}"}
    except Exception as e:
        result = {"error": str(e)}
    
    elapsed = time.time() - start_time
    return result, elapsed


async def poc_sender_loop(
    base_url: str,
    model: str,
    interval_seconds: float,
    requests_per_interval: int,
    stop_event: asyncio.Event,
    poc_artifacts: List[Dict[str, Any]],
    poc_times: List[float],
):
    nonce_counter = 0
    request_count = 0
    
    print(f"[PoC] Starting (interval: {interval_seconds}s, requests per interval: {requests_per_interval})\n")
    
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
            break
        except asyncio.TimeoutError:
            pass
        
        tasks = []
        batch_start = request_count + 1
        
        for i in range(requests_per_interval):
            nonces = list(range(nonce_counter, nonce_counter + 1))
            nonce_counter += 1
            request_count += 1
            
            block_hash = f"block_{request_count}"
            tasks.append((request_count, block_hash, nonces))
        
        print(f"[PoC] Sending batch #{batch_start}-{request_count}")
        
        results = await asyncio.gather(*[
            send_poc_request(base_url, model, block_hash, nonces)
            for _, block_hash, nonces in tasks
        ])
        
        for (req_id, block_hash, nonces), (result, elapsed) in zip(tasks, results):
            artifact = {
                "request_id": req_id,
                "timestamp": time.time(),
                "block_hash": block_hash,
                "nonces": nonces,
                "result": result,
                "elapsed_time": elapsed,
            }
            poc_artifacts.append(artifact)
            poc_times.append(elapsed)
        
        success_count = sum(1 for r, _ in results if r and "error" not in r)
        print(f"[PoC] Batch complete: {success_count}/{len(results)} successful\n")
    
    print(f"[PoC] Stopped: {len(poc_artifacts)} requests sent")


async def stream_process_output(process, log_file):
    with open(log_file, 'w', buffering=1) as log_f:
        while True:
            return_code = process.poll()
            
            line = process.stdout.readline()
            if line:
                print(line, end='', flush=True)
                log_f.write(line)
            
            if return_code is not None:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log_f.write(line)
                break
            
            await asyncio.sleep(0.01)
    
    return return_code


def parse_gsm8k_results(output_path: Path, start_time: float) -> Optional[Dict[str, Any]]:
    results_dirs = list(output_path.glob("*/"))
    if not results_dirs:
        return None
    
    results_files = []
    for results_dir in results_dirs:
        for results_file in results_dir.glob("results_*.json"):
            mtime = results_file.stat().st_mtime
            if mtime >= start_time:
                results_files.append(results_file)
    
    if not results_files:
        return None
    
    results_file = max(results_files, key=lambda p: p.stat().st_mtime)
    
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        if "results" in data and "gsm8k" in data["results"]:
            gsm8k = data["results"]["gsm8k"]
            return {
                "strict_match": gsm8k.get("exact_match,strict-match"),
                "flexible_extract": gsm8k.get("exact_match,flexible-extract"),
                "results_file": str(results_file),
            }
    except Exception as e:
        print(f"Error parsing results: {e}")
    
    return None


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output_path", type=str, default="./eval_results")
    parser.add_argument("--tasks", type=str, default="gsm8k")
    parser.add_argument("--poc_interval", type=float, default=1)
    parser.add_argument("--poc_requests", type=int, default=1, help="Number of PoC requests to send per interval")
    parser.add_argument("--disable_poc", action="store_true")
    
    args = parser.parse_args()
    
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    poc_requests = 0 if args.disable_poc else args.poc_requests
    run_name = f"chat_{args.batch_size}_poc_{poc_requests}"
    run_output_path = output_path / run_name
    run_output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Port: {args.port}")
    if args.disable_poc:
        print(f"PoC: Disabled")
    else:
        print(f"PoC: Enabled (interval={args.poc_interval}s, {args.poc_requests} req/interval)")
    print(f"Output: {run_output_path}")
    print("=" * 80)
    print()
    
    base_url = f"http://localhost:{args.port}/v1/chat/completions"
    
    cmd = [
        "lm-eval",
        "--model", "local-chat-completions",
        "--model_args", f"model={args.model_name},base_url={base_url},num_concurrent={args.batch_size}",
        "--tasks", args.tasks,
        "--output_path", str(output_path),
        "--log_samples",
        "--apply_chat_template",
    ]
    
    print("Starting lm_eval...")
    print()
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    start_time = time.time()
    
    poc_artifacts = []
    poc_times = []
    stop_event = asyncio.Event()
    
    if not args.disable_poc:
        poc_base_url = f"http://localhost:{args.port}"
        poc_task = asyncio.create_task(
            poc_sender_loop(poc_base_url, args.model_name, args.poc_interval, args.poc_requests, stop_event, poc_artifacts, poc_times)
        )
    else:
        poc_task = None
    
    log_file = run_output_path / "lm_eval.log"
    
    try:
        return_code = await stream_process_output(process, log_file)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        if poc_task:
            stop_event.set()
            await poc_task
        
        print()
        print("=" * 80)
        print(f"Completed in {elapsed:.1f}s ({elapsed/60:.1f}min)")
        print("=" * 80)
        print()
        
        results = {
            "model_name": args.model_name,
            "batch_size": args.batch_size,
            "port": args.port,
            "tasks": args.tasks,
            "elapsed_seconds": elapsed,
            "return_code": return_code,
            "poc_enabled": not args.disable_poc,
        }
        
        if return_code == 0:
            gsm8k_results = parse_gsm8k_results(output_path, start_time)
            if gsm8k_results:
                results["gsm8k"] = gsm8k_results
                print(f"GSM8K Results:")
                print(f"  Strict match:      {gsm8k_results['strict_match']:.4f}")
                print(f"  Flexible extract:  {gsm8k_results['flexible_extract']:.4f}")
                print(f"  Results file:      {gsm8k_results['results_file']}")
                print()
        
        if poc_artifacts:
            results["poc_requests_sent"] = len(poc_artifacts)
            results["poc_requests_successful"] = sum(
                1 for a in poc_artifacts if "error" not in a["result"]
            )
            
            if poc_times:
                results["poc_median_time"] = float(np.median(poc_times))
                results["poc_mean_time"] = float(np.mean(poc_times))
                results["poc_min_time"] = float(np.min(poc_times))
                results["poc_max_time"] = float(np.max(poc_times))
            
            print(f"PoC Requests:")
            print(f"  Sent:        {results['poc_requests_sent']}")
            print(f"  Successful:  {results['poc_requests_successful']}")
            if poc_times:
                print(f"  Median time: {results['poc_median_time']:.3f}s")
                print(f"  Mean time:   {results['poc_mean_time']:.3f}s")
            print()
            
            artifacts_file = run_output_path / "poc_artifacts.json"
            with open(artifacts_file, 'w') as f:
                json.dump(poc_artifacts, f, indent=2)
            print(f"PoC artifacts saved to: {artifacts_file}")
        
        stats_file = run_output_path / "run_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Run stats saved to: {stats_file}")
        print(f"Log saved to: {log_file}")
        
        return 0 if return_code == 0 else 1
        
    except KeyboardInterrupt:
        print("\n\nInterrupted!")
        process.terminate()
        if poc_task:
            stop_event.set()
            await poc_task
        return 130


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

#!/usr/bin/env python3
"""Throughput — one tool for BOTH PoC nonces and real chat inference.

Pure HTTP client: connect to a running server with --url (the rented box runs only
ML node + vLLM); loop for a duration and report throughput in a common frame so PoC
cost reads directly against real-inference capacity:

  * --mode poc  : 32-nonce /generate batches (decode trajectory) -> nonces/min, steps/s
  * --mode chat : 32 concurrent /v1/chat/completions             -> req/min, tokens/s

req/min (chat) and nonces/min (PoC) are the SAME unit: one decode sequence. Both run
the same concurrency (BATCH) and max_tokens, so the two rows are directly comparable.

  # against a running server (vLLM engine or ML-node proxy):
  python perfomance_nonces.py --mode poc  --url http://HOST:PORT --target vllm  --max-tokens 256
  python perfomance_nonces.py --mode chat --url http://HOST:PORT               --max-tokens 256

  # local dev (auto-boot vLLM, then connect — same client path):
  python perfomance_nonces.py --mode poc --max-tokens 256 [--eager]
"""
import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from poc_validation import (  # noqa: E402
    request_generate, save_run, add_engine_args, deploy_from_args,
)

DEFAULT_MODEL = "RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16"
BATCH = 32  # concurrency = PoC nonces-in-flight = chat concurrent requests, so
            # req/min lines up with nonce/min. Override with --concurrency (batch sweep).


def run_poc(url, target, model, seq_len, max_tokens, duration, warmup):
    """Client-driven continuous 32-nonce batches of /generate (decode trajectory).
    Returns (total_requests=nonces, total_steps, elapsed)."""
    nxt = 0

    def one():
        nonlocal nxt
        request_generate(url, target=target, model=model, nonces=list(range(nxt, nxt + BATCH)),
                         seq_len=seq_len, max_tokens=max_tokens)
        nxt += BATCH

    w_end = time.monotonic() + warmup
    while time.monotonic() < w_end:
        one()
    total, t0, deadline = 0, time.monotonic(), time.monotonic() + duration
    while time.monotonic() < deadline:
        one()
        total += BATCH
    elapsed = time.monotonic() - t0
    return total, total * (max_tokens + 1), elapsed


def _full_trajectories(resp, max_tokens) -> bool:
    """True if EVERY artifact in the response carries a complete k-trajectory
    (max_tokens+1 = prefill k0 + one k per decode step). Guards throughput accounting:
    steps/s credits max_tokens+1 per nonce, so short completions would inflate it.
    max_tokens==0 is prefill-only (no trajectory) — nothing to verify."""
    if max_tokens <= 0:
        return True
    arts = (resp or {}).get("artifacts") or []
    if not arts:
        return False
    return all(len(a.get("k_points_steps") or []) == max_tokens + 1 for a in arts)


def run_poc_pipeline(url, target, model, seq_len, max_tokens, duration, warmup):
    """APPLES-TO-APPLES with run_chat, NO door gap: BATCH worker threads each fire a
    SINGLE-nonce /generate back-to-back, so ~BATCH nonces are always in flight and the
    vLLM scheduler keeps the GPU continuously saturated. The serial run_poc above sends
    one 32-nonce request then WAITS (a gap between batches, mirrors production load);
    this version removes that gap by overlapping requests exactly like run_chat's 32
    continuous single-sequence workers -> fair PoC-vs-inference comparison.
    Returns (total_nonces, total_steps, elapsed)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    c = {"next": 0, "done": 0, "errors": 0, "short": 0}
    lock = threading.Lock()
    state = {"deadline": 0.0}

    def worker():
        while time.monotonic() < state["deadline"]:
            with lock:
                n = c["next"]; c["next"] += 1
            try:
                resp, _ = request_generate(url, target=target, model=model, nonces=[n],
                                           seq_len=seq_len, max_tokens=max_tokens)
            except Exception:
                with lock:
                    c["errors"] += 1   # counted, not silently swallowed (see _print)
                continue  # keep the pipeline full
            # VERIFY THE WORK: throughput is credited as max_tokens+1 steps per nonce, so a
            # nonce that finished SHORT would inflate steps/s (this is exactly how a capped
            # batch once reported an impossible 422 nonce/min). Only count full trajectories.
            if not _full_trajectories(resp, max_tokens):
                with lock:
                    c["short"] += 1
                continue
            with lock:
                c["done"] += 1

    def phase(seconds):
        state["deadline"] = time.monotonic() + seconds
        with ThreadPoolExecutor(max_workers=BATCH) as ex:
            for f in [ex.submit(worker) for _ in range(BATCH)]:
                f.result()

    if warmup:
        phase(warmup)
    with lock:
        c["done"] = c["errors"] = c["short"] = 0
    t0 = time.monotonic()
    phase(duration)
    elapsed = time.monotonic() - t0
    # Loud, not silent: a run that mostly failed (or returned short trajectories) must not
    # report a plausible-looking throughput number.
    if c["errors"] or c["short"]:
        print(f"  [WARNING] {c['errors']} request errors, {c['short']} SHORT trajectories "
              f"(expected {max_tokens + 1} steps) — excluded from throughput")
    return c["done"], c["done"] * (max_tokens + 1), elapsed


def _chat_prompt(prompt_len, idx):
    """Raw prompt of ~prompt_len tokens, UNIQUE per request (leading idx) so
    concurrent workers never share a prefix-cache hit that would fake-cheapen the
    prefill — matching PoC, where every nonce prefills its own vector. Uses raw
    /v1/completions (NOT chat), so there is no chat-template overhead: token count
    == prompt_len, aligned 1:1 with PoC's seq_len prefill (apples-to-apples)."""
    n = max(1, prompt_len - 1)
    return f"{idx} " + "word " * n            # "word" == 1 token -> ~prompt_len tokens


async def _chat_one(client, url, model, max_tokens, idx, prompt_len):
    body = {
        "model": model,
        "prompt": _chat_prompt(prompt_len, idx),   # raw completions: no chat template
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,  # force exactly max_tokens decode steps (fair tok/s)
    }
    r = await client.post(f"{url}/v1/completions", json=body)
    r.raise_for_status()
    u = r.json().get("usage", {})
    return u.get("completion_tokens", 0), u.get("prompt_tokens", 0)


async def _run_chat(url, model, max_tokens, duration, warmup, concurrency, prompt_len):
    import httpx
    counters = {"req": 0, "tok": 0, "idx": 0, "ptok": 0}

    async def worker(client, deadline):
        while time.monotonic() < deadline:
            i = counters["idx"]; counters["idx"] += 1
            try:
                tok, ptok = await _chat_one(client, url, model, max_tokens, i, prompt_len)
            except Exception:
                continue   # transient ReadError under a full pool must not kill the run
            counters["req"] += 1; counters["tok"] += tok; counters["ptok"] = ptok

    # httpx defaults cap the pool at 100 connections, so above --concurrency 100
    # the extra workers silently WAIT for a pooled connection: the engine never
    # sees more than 100 in flight (metrics: running pins at exactly 100) while
    # the tool reports the nominal concurrency. Limits must live ON the transport
    # (a bare transport= would rebuild default limits and override client limits).
    # retries + tolerant warmup absorb the transient ReadErrors a 256+ connection
    # storm produces against a single uvicorn.
    _lim = httpx.Limits(max_connections=concurrency + 8,
                        max_keepalive_connections=concurrency + 8)
    _tr = httpx.AsyncHTTPTransport(retries=5, limits=_lim)
    async with httpx.AsyncClient(timeout=600, transport=_tr) as client:
        await asyncio.gather(*[_chat_one(client, url, model, 8, -1 - k, prompt_len)
                               for k in range(concurrency)],
                             return_exceptions=True)  # warmup (unique idx)
        if warmup:
            wend = time.monotonic() + warmup
            await asyncio.gather(*[worker(client, wend) for _ in range(concurrency)])
        counters["req"] = counters["tok"] = 0
        t0 = time.monotonic(); deadline = t0 + duration
        await asyncio.gather(*[worker(client, deadline) for _ in range(concurrency)])
        elapsed = time.monotonic() - t0
    print(f"  [chat prefill: prompt_tokens={counters['ptok']}]")
    return counters["req"], counters["tok"], elapsed


def run_chat(url, model, max_tokens, duration, warmup, seq_len):
    return asyncio.run(_run_chat(url, model, max_tokens, duration, warmup, BATCH, seq_len))


def _time_poc_once(url, target, model, seq_len, max_tokens, nonce):
    t0 = time.monotonic()
    request_generate(url, target=target, model=model, nonces=[nonce],
                     seq_len=seq_len, max_tokens=max_tokens)
    return time.monotonic() - t0


def _chat_stream(url, model, seq_len, max_tokens, idx):
    """Streamed chat completion -> (ttft_s, total_s). TTFT = wall time to the FIRST
    streamed token = prefill cost (direct). total-ttft over the rest = decode."""
    import httpx
    body = {"model": model, "prompt": _chat_prompt(seq_len, idx), "max_tokens": max_tokens,
            "temperature": 0.0, "ignore_eos": True, "stream": True}
    t0 = time.monotonic(); ttft = None; last = t0
    with httpx.stream("POST", f"{url}/v1/completions", json=body, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data:") or line.strip() == "data: [DONE]":
                continue
            now = time.monotonic()
            if ttft is None:
                ttft = now - t0
            last = now
    return ttft, last - t0


def run_split(mode, url, target, model, seq_len, decode_steps, samples=5):
    """DIRECT prefill/decode split — concurrency 1, wall-clock, NO Little's-law / no
    regression fit. Prefill and decode are measured directly, per mode:
      chat: STREAM one request -> prefill = TTFT (time to first token);
            decode/step = (total - TTFT) / (decode_steps - 1).
      poc:  /generate can't stream, so prefill = a max_tokens=0 (prefill-only) request;
            decode/step = (t[decode_steps] - t[prefill-only]) / decode_steps.
    Median over `samples` (unique nonce/idx each -> no cache reuse)."""
    import statistics
    if mode == "poc":
        def prefill_i(i): return _time_poc_once(url, target, model, seq_len, 0, 100000 + i)
        def full_i(i):    return _time_poc_once(url, target, model, seq_len, decode_steps, 200000 + i)
        _time_poc_once(url, target, model, seq_len, decode_steps, 1)     # warmup
        pf = statistics.median([prefill_i(i) for i in range(samples)])
        fu = statistics.median([full_i(i) for i in range(samples)])
        per_step = (fu - pf) / decode_steps
        return {"mode": mode, "prefill_ms": round(pf * 1000, 1),
                "decode_ms_per_step": round(per_step * 1000, 3),
                "seq_len": seq_len, "decode_steps": decode_steps}
    # chat: streaming TTFT
    _chat_stream(url, model, seq_len, decode_steps, 1)                   # warmup
    ttfts, decs = [], []
    for i in range(samples):
        ttft, total = _chat_stream(url, model, seq_len, decode_steps, 100000 + i)
        ttfts.append(ttft); decs.append((total - ttft) / max(1, decode_steps - 1))
    return {"mode": mode, "prefill_ms": round(statistics.median(ttfts) * 1000, 1),
            "decode_ms_per_step": round(statistics.median(decs) * 1000, 3),
            "seq_len": seq_len, "decode_steps": decode_steps}


def _results(mode, total_req, work, elapsed, max_tokens):
    rpm = total_req / elapsed * 60 if elapsed else 0.0
    wps = work / elapsed if elapsed else 0.0
    res = {"mode": mode, "req_per_min": round(rpm, 1),
           "total_req": total_req, "elapsed_s": round(elapsed, 1), "max_tokens": max_tokens}
    res["steps_per_s" if mode == "poc" else "tokens_per_s"] = round(wps, 1)
    if mode == "poc":  # back-compat keys
        res["nonces_per_s"] = round(total_req / elapsed if elapsed else 0.0, 3)
        res["total_nonces"] = total_req
    return res


def _print(res, prov):
    unit = "nonces" if res["mode"] == "poc" else "requests"
    work = f"steps/s={res['steps_per_s']}" if res["mode"] == "poc" else f"tokens/s={res['tokens_per_s']}"
    print(f"\n=== {res['mode']} throughput ===")
    print(f"req/min = {res['req_per_min']:.0f}   {work}   "
          f"({res['total_req']} {unit} in {res['elapsed_s']}s, "
          f"concurrency={BATCH}, max_tokens={res['max_tokens']})")
    keys = ("vllm_version", "vllm_commit", "gpu", "attention_backend",
            "cudagraph_mode", "dtype", "quantization")
    print("  provenance: " + "  ".join(f"{k}={prov[k]}" for k in keys if k in prov))


def _save_path(stem, mode, multi):
    if not multi:
        return stem
    return stem[:-5] + f".{mode}.json" if stem.endswith(".json") else f"{stem}.{mode}"


def main():
    global BATCH
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["poc", "chat", "both"], default="poc")
    ap.add_argument("--model", required=True)
    add_engine_args(ap)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--warmup", type=float, default=15.0)
    ap.add_argument("--poc-load", choices=["serial", "pipeline"], default="pipeline",
                    help="pipeline=32 continuous single-nonce workers, NO gap, "
                         "apples-to-apples with chat (default). serial=one 32-nonce "
                         "request then wait (production load; has an inter-batch gap).")
    ap.add_argument("--concurrency", type=int, default=BATCH,
                    help="nonces-in-flight = chat concurrent requests (batch sweep knob).")
    ap.add_argument("--split", action="store_true",
                    help="report prefill vs decode/step separately (concurrency-1, "
                         "wall-clock, max_tokens=1 vs --max-tokens). Direct, no throughput fit.")
    ap.add_argument("--split-samples", type=int, default=5)
    ap.add_argument("--save")
    a = ap.parse_args()
    BATCH = a.concurrency                        # so req/min == nonce/min at this batch

    modes = ["poc", "chat"] if a.mode == "both" else [a.mode]
    with deploy_from_args(a, a.model) as (url, srv):  # one server lifetime for all modes
        if a.split:
            print(f"# prefill/decode split (concurrency 1, prefill={a.seq_len} tok, "
                  f"decode={a.max_tokens} steps, median of {a.split_samples})")
            for mode in modes:
                s = run_split(mode, url, a.target, a.model, a.seq_len, a.max_tokens, a.split_samples)
                src = "TTFT stream" if mode == "chat" else "max_tokens=0"
                print(f"{mode:4s}: prefill={s['prefill_ms']:8.1f} ms ({src})   "
                      f"decode={s['decode_ms_per_step']:6.3f} ms/step")
            return
        for mode in modes:
            if mode == "poc":
                poc_fn = run_poc_pipeline if a.poc_load == "pipeline" else run_poc
                total, work, elapsed = poc_fn(url, a.target, a.model, a.seq_len,
                                              a.max_tokens, a.duration, a.warmup)
            else:
                total, work, elapsed = run_chat(url, a.model, a.max_tokens, a.duration, a.warmup, a.seq_len)
            res = _results(mode, total, work, elapsed, a.max_tokens)
            if mode == "poc":
                res["poc_load"] = a.poc_load   # serial (production) vs pipeline (fair)
            _print(res, a.prov)
            if a.save:
                meta = {"model": a.model, "mode": mode, "seq_len": a.seq_len,
                        "max_tokens": a.max_tokens, "batch_size": BATCH,
                        "poc_load": (a.poc_load if mode == "poc" else None), **a.prov}
                path = _save_path(a.save, mode, len(modes) > 1)
                save_run(path, meta, [], results=res)
                print(f"saved -> {path}")


if __name__ == "__main__":
    main()

"""256+256 batch sweep — PoC nonces/min vs chat req/min across concurrency, one graph.

The release-target view: fix the shape at 256 prefill + 256 decode, sweep the batch
(concurrency), and plot PoC vs pure-inference throughput on the SAME axis (req/min ==
nonce/min, one decode sequence). Shows where PoC tracks inference and where it diverges.

APPLES-TO-APPLES: the server MUST have prefix caching OFF (it is the default in
DEFAULT_POC_SERVE_ARGS / perf_gate.sh) — PoC nonces prefill unique seeded vectors and
can never cache, so a cached chat baseline is unfair. Chat here uses raw /v1/completions
with a unique 256-token prompt (perfomance_nonces._chat_prompt).

HOW TO RUN
  # 1) boot an uncached server (prefix caching off is already the PoC default):
  VLLM_POC_MIXED_DECODE=1 .venv/bin/vllm serve <MODEL> --host 127.0.0.1 --port 8000 \
      --poc-decode --no-async-scheduling --no-enable-prefix-caching \
      --gpu-memory-utilization 0.85 --max-model-len 1024

  # 2) sweep + render:
  .venv/bin/python benchmarks/poc/batch_sweep.py --url http://127.0.0.1:8000 --model <MODEL> \
      --concurrencies 1,2,4,8,16,32,64 --seq-len 256 --max-tokens 256 \
      --duration 20 --warmup 6 --out benchmarks/poc/runs/sweep_256_256

Outputs: <out>.json (raw), <out>.csv, and <out>.png if matplotlib is available (else a
printed table). req/min is directly comparable between the two lines.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import perfomance_nonces as pn  # noqa: E402


def _rpm(url, target, model, mode, seq_len, max_tokens, dur, warm, conc):
    pn.BATCH = conc                                   # concurrency for this point
    if mode == "poc":
        total, _work, el = pn.run_poc_pipeline(url, target, model, seq_len, max_tokens, dur, warm)
    else:
        total, _work, el = pn.run_chat(url, model, max_tokens, dur, warm, seq_len)
    return round(total / el * 60, 1) if el else 0.0   # req/min == nonce/min


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="running (uncached!) server base url")
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", default="vllm")
    ap.add_argument("--concurrencies", default="1,2,4,8,16,32,64")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--warmup", type=float, default=6.0)
    ap.add_argument("--out", default="benchmarks/poc/runs/sweep_256_256")
    a = ap.parse_args()

    concs = [int(x) for x in a.concurrencies.split(",")]
    rows = []
    print(f"# 256+256 batch sweep (seq_len={a.seq_len}, max_tokens={a.max_tokens}), uncached")
    print(f"{'batch':>6} {'poc req/min':>12} {'chat req/min':>13} {'poc/chat':>9}")
    for c in concs:
        poc = _rpm(a.url, a.target, a.model, "poc", a.seq_len, a.max_tokens, a.duration, a.warmup, c)
        chat = _rpm(a.url, a.target, a.model, "chat", a.seq_len, a.max_tokens, a.duration, a.warmup, c)
        ratio = round(poc / chat, 3) if chat else 0.0
        rows.append({"concurrency": c, "poc_req_min": poc, "chat_req_min": chat, "poc_over_chat": ratio})
        print(f"{c:>6} {poc:>12} {chat:>13} {ratio:>9}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"model": a.model, "seq_len": a.seq_len, "max_tokens": a.max_tokens,
            "duration": a.duration,
            # NOT verified from the server — this script drives a pre-booted --url server, so
            # prefix caching is whoever typed the boot command's responsibility. Recording a
            # flat "off" here would be a provenance claim we never checked.
            "prefix_caching": "unverified (boot flag; MUST be --no-enable-prefix-caching)",
            "rows": rows}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    out.with_suffix(".csv").write_text(
        "concurrency,poc_req_min,chat_req_min,poc_over_chat\n" +
        "\n".join(f"{r['concurrency']},{r['poc_req_min']},{r['chat_req_min']},{r['poc_over_chat']}"
                  for r in rows) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cs = [r["concurrency"] for r in rows]
        plt.figure(figsize=(7, 4.5))
        plt.plot(cs, [r["poc_req_min"] for r in rows], "o-", label="PoC (nonce/min)")
        plt.plot(cs, [r["chat_req_min"] for r in rows], "s-", label="chat inference (req/min)")
        plt.xscale("log", base=2); plt.xticks(cs, [str(c) for c in cs])
        plt.xlabel("batch (concurrency)"); plt.ylabel("req/min  (one 256+256 sequence)")
        plt.title(f"256+256 throughput vs batch — PoC vs inference (uncached)\n{a.model}")
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
        plt.savefig(out.with_suffix(".png"), dpi=130)
        print(f"\ngraph -> {out.with_suffix('.png')}")
    except Exception as e:
        print(f"\n(matplotlib unavailable: {e}) — data in {out.with_suffix('.json')} / .csv")
    print(f"data  -> {out.with_suffix('.json')}  {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()

"""Generate a random prompt, tokenize it, truncate to 256 tokens, then complete it.

Usage:
    python random_prompt_completion.py --server http://localhost:8000
    python random_prompt_completion.py --server http://localhost:8000 --max-tokens 64 --seq-len 128
"""
import argparse
import json
import random
import string

import requests

WORD_LIST = [
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
    "science", "technology", "neural", "network", "language", "model",
    "inference", "validation", "computation", "distributed", "system",
    "entropy", "gradient", "attention", "transformer", "embedding",
    "matrix", "vector", "probability", "sequence", "token", "decode",
    "encode", "training", "dataset", "batch", "optimizer", "loss",
    "accuracy", "precision", "recall", "benchmark", "hardware", "memory",
    "latency", "throughput", "parallel", "cluster", "node", "server",
    "request", "response", "protocol", "algorithm", "function", "variable",
    "parameter", "hyperparameter", "architecture", "layer", "weight",
    "bias", "activation", "softmax", "logit", "probability", "sample",
]

SENTENCE_TEMPLATES = [
    "The {adj} {noun} {verb} through the {place} while {gerund} {obj}.",
    "In the field of {field}, {noun} plays a crucial role in {gerund} {obj}.",
    "Scientists discovered that {noun} can {verb} more efficiently when {condition}.",
    "The {adj} system processes {obj} by {gerund} each {noun} independently.",
    "Recent advances in {field} have shown that {noun} {verb} beyond expectations.",
    "Consider the following: when {noun} {verb}, the resulting {obj} changes significantly.",
    "Researchers argue that {adj} {noun} should {verb} before {gerund} any {obj}.",
    "The {place} contains many {adj} {noun} that {verb} the {obj} continuously.",
]

ADJECTIVES = ["complex", "distributed", "parallel", "efficient", "robust", "scalable",
               "lightweight", "high-performance", "adaptive", "intelligent", "dynamic"]
NOUNS = ["model", "system", "network", "algorithm", "process", "module", "framework",
          "pipeline", "cluster", "agent", "engine", "component", "dataset", "layer"]
VERBS = ["processes", "generates", "validates", "optimizes", "transforms", "encodes",
          "computes", "evaluates", "scales", "integrates", "synchronizes", "manages"]
GERUNDS = ["processing", "generating", "validating", "optimizing", "transforming",
            "encoding", "computing", "evaluating", "scaling", "integrating"]
OBJECTS = ["tokens", "vectors", "embeddings", "gradients", "parameters", "batches",
            "sequences", "outputs", "inputs", "activations", "weights", "states"]
PLACES = ["cluster", "memory", "network", "pipeline", "system", "environment"]
FIELDS = ["machine learning", "distributed computing", "natural language processing",
           "computer vision", "reinforcement learning", "information theory"]
CONDITIONS = ["properly initialized", "running in parallel", "given sufficient resources",
               "operating at full capacity", "synchronized across nodes"]


def random_sentence() -> str:
    template = random.choice(SENTENCE_TEMPLATES)
    return template.format(
        adj=random.choice(ADJECTIVES),
        noun=random.choice(NOUNS),
        verb=random.choice(VERBS),
        gerund=random.choice(GERUNDS),
        obj=random.choice(OBJECTS),
        place=random.choice(PLACES),
        field=random.choice(FIELDS),
        condition=random.choice(CONDITIONS),
    )


def make_random_prompt(target_chars: int = 4000) -> str:
    sentences = []
    total = 0
    while total < target_chars:
        s = random_sentence()
        sentences.append(s)
        total += len(s) + 1
    return " ".join(sentences)


def get_model(base_url: str, timeout: int = 10) -> str:
    resp = requests.get(f"{base_url}/v1/models", timeout=timeout)
    resp.raise_for_status()
    return resp.json()["data"][0]["id"]


def tokenize(base_url: str, model: str, text: str, timeout: int = 30) -> list[int]:
    resp = requests.post(
        f"{base_url}/tokenize",
        json={"model": model, "prompt": text},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["tokens"]


def complete(
    base_url: str,
    model: str,
    token_ids: list[int],
    max_tokens: int,
    timeout: int = 120,
) -> dict:
    resp = requests.post(
        f"{base_url}/v1/completions",
        json={
            "model": model,
            "prompt": token_ids,
            "max_tokens": max_tokens,
            "temperature": 1.0,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(
        description="Random prompt → tokenize → truncate → complete."
    )
    parser.add_argument("--server", default="http://localhost:8003",
                        help="vllm server base URL")
    parser.add_argument("--seq-len", type=int, default=256,
                        help="Truncate prompt to this many tokens (default: 256)")
    parser.add_argument("--max-tokens", type=int, default=10,
                        help="Max new tokens to generate (default: 32)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print(f"Server : {args.server}")
    print(f"seq_len: {args.seq_len}  max_tokens: {args.max_tokens}\n")

    model = get_model(args.server)
    print(f"Model  : {model}\n")

    prompt = make_random_prompt()
    print(f"Generated prompt ({len(prompt)} chars):")
    print(f"  {prompt[:200]}{'…' if len(prompt) > 200 else ''}\n")

    tokens = tokenize(args.server, model, prompt)
    print(f"Tokenized: {len(tokens)} tokens")

    truncated = tokens[: args.seq_len]
    print(f"Truncated: {len(truncated)} tokens\n")

    print("Sending to /v1/completions …")
    result = complete(args.server, model, truncated, args.max_tokens)

    choice = result["choices"][0]
    completion_text = choice["text"]
    finish_reason = choice["finish_reason"]
    usage = result.get("usage", {})

    print(f"Completion ({finish_reason}):")
    print(f"  {completion_text!r}")
    print(f"\nUsage: prompt_tokens={usage.get('prompt_tokens')}  "
          f"completion_tokens={usage.get('completion_tokens')}  "
          f"total={usage.get('total_tokens')}")


if __name__ == "__main__":
    main()

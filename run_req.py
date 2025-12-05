import asyncio
import aiohttp
import random

URL = "http://localhost:8080/v1/chat/completions"
NUM_REQUESTS = 20
VOCAB = [str(i) for i in range(1000, 1050)]

PROMPTS = [
    "Who won 2020 World Series?",
    "Explain quantum entanglement in one sentence.",
    "What is the capital of France?",
    "Translate 'hello' to Spanish.",
    "What is 2+2?",
    "Write a haiku about snow.",
    "Describe a futuristic city.",
    "Name a famous painting.",
    "What is the largest planet?",
    "Give one programming tip."
]


def generate_random_enforced(max_tokens):
    n = max_tokens  # ровно столько, сколько max_tokens
    tokens = []
    used = set()
    for _ in range(n):
        tok = random.choice(VOCAB)
        while tok in used:
            tok = random.choice(VOCAB)
        used.add(tok)

        # Формируем top_tokens: tok + 2 уникальных случайных, не равных tok
        others = set(VOCAB) - {tok}
        top_two = random.sample(others, 2)
        top = [tok] + top_two

        tokens.append({
            "token": tok,
            "top_tokens": top
        })
    return tokens


def make_random_payload():
    max_t = random.randint(1, 3)
    enforced = generate_random_enforced(max_t)
    prompt = random.choice(PROMPTS)
    return {
        "model": "/mnt/fs/Qwen3-32B-FP8/",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_t,
        "temperature": round(random.uniform(0.1, 1.2), 2),
        "logprobs": True,
        "top_logprobs": 3,
        "enforced_tokens": {"tokens": enforced},
        "_expected": enforced  # для проверки ответа
    }


def validate_response(payload, resp_json):
    try:
        content = resp_json["choices"][0]["logprobs"]["content"]
    except Exception as e:
        return False, f"missing logprobs: {e}"

    expected = payload["_expected"]

    # Проверяем только столько токенов, сколько вернул модель
    if len(content) < len(expected):
        expected = expected[:len(content)]

    for i, enforced_item in enumerate(expected):
        exp_tok = enforced_item["token"]
        exp_top = set(enforced_item["top_tokens"])

        model_tok = content[i]["token"]

        top_logits = content[i]["top_logprobs"]
        if isinstance(top_logits, dict):
            model_top = set(top_logits.keys())
        else:
            model_top = {t["token"] for t in top_logits}

        if model_tok != exp_tok:
            return False, f"token mismatch {model_tok} != {exp_tok}"

        if not model_top.issubset(exp_top):
            return False, f"top-logprobs mismatch {model_top} not subset of {exp_top}"

    return True, "OK"


async def run_one(session, req_id):
    payload = make_random_payload()
    print(f"Request #{req_id} payload:")
    print(payload)
    try:
        async with session.post(URL, json=payload, timeout=20) as resp:
            if resp.status != 200:
                text = await resp.text()
                return req_id, False, f"HTTP {resp.status}: {text}"

            resp_json = await resp.json()
            print(f"Response #{req_id}:")
            print(resp_json)
            ok, msg = validate_response(payload, resp_json)
            return req_id, ok, msg

    except Exception as e:
        return req_id, False, f"Exception: {e}"


async def main():
    print(f"Running {NUM_REQUESTS} parallel diverse tests...\n")

    async with aiohttp.ClientSession() as session:
        tasks = [run_one(session, i) for i in range(NUM_REQUESTS)]
        results = await asyncio.gather(*tasks)

    for rid, ok, msg in results:
        status = "✔ OK" if ok else "❌ FAIL"
        print(f"[{rid:02}] {status} — {msg}")

    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n=== SUMMARY ===\nPassed: {passed}/{NUM_REQUESTS}")
    if passed < NUM_REQUESTS:
        print("\nFailed cases:")
        for rid, ok, msg in results:
            if not ok:
                print(f"  [{rid}] {msg}")


if __name__ == "__main__":
    asyncio.run(main())
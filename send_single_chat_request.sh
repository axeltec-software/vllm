curl http://localhost:8005/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16",
        "messages": [
            {"role": "user", "content": "What is the capital of France? Answer with just the city name."}
        ],
    "max_tokens": 20,
    "temperature": 0.0
    }'

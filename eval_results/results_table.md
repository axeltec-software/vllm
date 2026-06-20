| Configuration | Batch Size | PoC Batch | Decode (max_tokens) | Accuracy (Strict/Flexible) | Time gsm8k (s) | Median Time PoC (s) |
|---------------|------------|-----------|---------------------|----------------------------|----------------|---------------------|
| Qwen2.5-7B-Instruct-quantized.w8a16 | 8          | 0         | 0                   | 0.1300 / 0.6800          | 132            |                     |
| Qwen2.5-7B-Instruct-quantized.w8a16 | 8          | 1         | 0                   | 0.1300 / 0.7200          | 138            | 1.249               |
| Qwen2.5-7B-Instruct-quantized.w8a16 | 8          | 4         | 0                   | 0.1300 / 0.6700          | 149            | 1.400               |
| Qwen2.5-7B-Instruct-quantized.w8a16 | 8          | 8         | 256                 | 0.1300 / 0.6900          | 152            | 15.688              |
| Qwen2.5-7B-Instruct-quantized.w8a16 | 16         | 0         | 0                   | 0.1000 / 0.6900          | 96             |                     |
| Qwen2.5-7B-Instruct-quantized.w8a16 | 16         | 1         | 0                   | 0.1000 / 0.6900          | 99             | 0.963               |
| Qwen2.5-7B-Instruct-quantized.w8a16 | 16         | 1         | 256                 | 0.1200 / 0.6900          | 134            | 27.137              |
| Qwen2.5-7B-Instruct-quantized.w8a16 | 16         | 4         | 0                   | 0.0000 / 0.0000          | 62             | 0.001               |
| Qwen3-0.6B    | 8          | 32        | 256                 | 0.0000 / 0.1800          | 104            | 9.077               |

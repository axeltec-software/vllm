export VLLM_TORCH_PROFILER_DIR=/home/irene/projects/gonka_poc/poc_decode/vllm/torch_profiler_logs 
python -m vllm.entrypoints.openai.api_server \
    --model RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16 \
    --max-model-len 10000 \
    --port 8005 \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --enforce-eager
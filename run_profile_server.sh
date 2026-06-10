export VLLM_TORCH_PROFILER_DIR=/home/irene/projects/gonka_poc/poc_decode/v15-merged/vllm/torch_profiler_logs 
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_POC_CUDAGRAPH=1
python -m vllm.entrypoints.openai.api_server \
    --model RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16 \
    --max-model-len 1024 \
    --port 8005 \
    --gpu-memory-utilization 0.7 \
    --no-async-scheduling \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --poc-decode \
    #--enforce-eager
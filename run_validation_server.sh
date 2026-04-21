#export VLLM_CONFIGURE_LOGGING=1
#export VLLM_LOGGING_LEVEL=DEBUG
#export VLLM_LOGGING_CONFIG_PATH=logging_config.json
#export VLLM_BATCH_INVARIANT=0
export VLLM_ATTENTION_BACKEND="FLASHINFER"
export VLLM_USE_V1="1"
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-0.6B \
    --max-model-len 10000 \
    --port 8010 \
    --gpu-memory-utilization 0.3 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --poc-decode 
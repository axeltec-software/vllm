export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_LOGGING_CONFIG_PATH=logging_config.json
#export VLLM_BATCH_INVARIANT=0
# Qwen/Qwen3-0.6B Qwen/Qwen2-1.5B-Instruct Qwen/Qwen3-0.6B-FP8
export VLLM_ATTENTION_BACKEND="FLASHINFER"
export VLLM_USE_V1="1"
rm -rf vllm.log
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-0.6B-FP8 \
    --max-model-len 10000 \
    --port 8005 \
    --gpu-memory-utilization 0.3 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --poc-decode
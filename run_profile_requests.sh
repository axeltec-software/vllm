export VLLM_TORCH_PROFILER_DIR=/home/irene/projects/gonka_poc/poc_decode/v15-merged/vllm/torch_profiler_logs 
python random_prompt_completion.py --server http://localhost:8005 --max-tokens 64 --seq-len 128
python random_prompt_completion.py --server http://localhost:8005 --max-tokens 64 --seq-len 128
#python send_poc_request.py --url http://localhost:8005 --nonces 0
#python send_poc_request.py --url http://localhost:8005 --nonces 0
curl -X POST http://localhost:8005/start_profile
 python random_prompt_completion.py --server http://localhost:8005 --max-tokens 5 --seq-len 32
 #python send_poc_request.py --url http://localhost:8005 --nonces 0 --max-tokens 5
curl -X POST http://localhost:8005/stop_profile
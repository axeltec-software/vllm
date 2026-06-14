 python send_poc_request.py --url http://localhost:8005 --nonces 0 --max-tokens 50
 python send_poc_request.py --url http://localhost:8005 --nonces 0 1 2 3 4 5 6 7 --block-hash 0xkv_integrity_1 
 python random_prompt_completion.py --server http://localhost:8005
 python send_poc_request.py --url http://localhost:8005 --nonces 0 1 2 3 4 5 6 7 --block-hash 0xkv_integrity_1 
 
 python random_prompt_completion.py --server http://localhost:8005
 python send_poc_request.py --url http://localhost:8005 --nonces 0 --max-tokens 50
 python send_poc_request.py --url http://localhost:8005 --nonces 1 --max-tokens 50

 python send_poc_request.py --url http://localhost:8005 --nonces 0 --max-tokens 50
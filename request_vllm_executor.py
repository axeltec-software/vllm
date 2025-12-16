import json
import requests
import os

url = "http://localhost:9112/v1/chat/completions"  # Replace with your actual API endpoint
payload = {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "messages": [
              {"role": "system", "content": "You are a helpful guide."},
              {"role": "user", "content": "Describe in details what species live in Africa, describe each of them, make a long essay."}
            ],
            "max_tokens": 2000,
            "temperature": 0.99,
            "logprobs": True
          }
#{"role": "user", "content": "What is the capital of France?"}

try:
    file_to_delete = '/home/imizus/projects/vllm_gonka_tests/vllm_validator/vllm/output_tokens_with_ids.jsonl'
    
    if os.path.exists(file_to_delete):
        try:
            os.remove(file_to_delete)
            print(f"File '{file_to_delete}' deleted successfully.")
        except PermissionError:
            print(f"Permission denied to delete the file '{file_to_delete}'.")
        except Exception as e:
            print(f"An error occurred while deleting the file: {e}")
    else:
        print(f"File '{file_to_delete}' does not exist.")


    response = requests.post(url, json=payload)
    if response.status_code // 100 != 2:
        print(f"Error: Received status code {response.status_code}")
        print(f"Response: {response.text}")
        exit()

    # response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

    # resp_json = response.json()
    # with open('response.json', 'w') as f:
    #     json.dump(resp_json, f, indent=4)
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
    exit()

print("Request was successful!")

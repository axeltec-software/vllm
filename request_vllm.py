import json
import requests
import os

url = "http://localhost:9111/v1/chat/completions"  # Replace with your actual API endpoint
payload = {
            "model": "RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16",
            "messages": [
              {"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": "What is the capital of France?"}
            ],
            "max_tokens": 3000,
            "temperature": 0.99,
            "logprobs": True
          }

try:
    file_to_delete = '/home/imizus/projects/vllm/vllm_sampling_ranks.txt'
    
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

    #response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

    #resp_json = response.json()
    # with open('response.json', 'w') as f:
    #     json.dump(resp_json, f, indent=4)
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
    exit()

print("Request was successful!")

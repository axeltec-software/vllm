
import argparse
import requests
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm
from typing import List
from datasets import load_dataset
from transformers import AutoTokenizer
import json
import os
import pickle

files_to_delete = ['/home/imizus/projects/vllm/vllm_top_p_probs_max.txt', '/home/imizus/projects/vllm/vllm_top_p_probs_mean.txt']
model_name = "RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16"

if not os.path.exists('./datasets'):
    os.makedirs('datasets')

path = "datasets/case_summ_tokenized.pickle"

def choose_tokenized_prompt(prompt_tokens: int, num_prompts: int) -> List[str]:
    dataset = load_dataset(
                "ChicagoHAI/CaseSumm", split="train", trust_remote_code=True
            )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    system_prompt = "## TASK: Make a summary of the following text:\n\n ## TEXT: "
    prompts = [system_prompt + doc for doc in dataset["opinion"]]
    
    if os.path.exists(path):
        print("Load tokenized prompts...")
        with open(path, "rb") as f:
            tokenized_prompts = pickle.load(f)
    else:
        print("Starting prompt tokenization...")
        tokenized_prompts = [tokenizer.tokenize(text) for text in prompts]
        print("Ended prompt tokenization...")
        with open(path, "wb") as f:
            pickle.dump(tokenized_prompts, f)

    allowed_prompts = []
    for i, tokens in enumerate(tokenized_prompts):
        if len(tokens) >= prompt_tokens:
            prompt = tokenizer.convert_tokens_to_string(tokens[:prompt_tokens])
            allowed_prompts.append(prompt)
    return random.choices(allowed_prompts, k=num_prompts)

def choose_long_prompt() -> str:
    long_prompt = """
    *"Analyze how Shakespeare’s narrative techniques and themes persist in 21st-century films, TV, and literature. Address:

    Adaptations vs. inspirations (e.g., The Lion King as Hamlet vs. Westworld’s thematic echoes).

    The use of soliloquies in modern media (e.g., House of Cards’ breaking the fourth wall).

    How tropes like ‘the tragic hero’ (Breaking Bad) or ‘mistaken identity’ (She’s the Man) evolve.

    Whether contemporary audiences recognize these influences without explicit references.
    Support your analysis with 5+ examples from post-2010 works. Argue if Shakespeare’s legacy is timeless or fading in the digital age."*
    """
    return long_prompt

def choose_random_prompt(num_prompts: int) -> List[str]:
    prompts = [
    """Compare the feasibility and scalability of three proposed climate change solutions: direct air capture (DAC), ocean iron fertilization, and next-gen nuclear reactors. Analyze their energy requirements, costs (per ton of CO₂ mitigated), and potential unintended consequences. Use real-world examples like Climeworks’ DAC plants or China’s HTR-PM reactor. Conclude with a policy recommendation for governments prioritizing one technology.""",
    """Explain how Shor’s algorithm threatens RSA and ECC encryption. Detail post-quantum cryptography alternatives (lattice-based, hash-based), their trade-offs (key size, speed), and adoption timelines. Mention NIST’s PQC standardization process. Should enterprises start migrating now, or is this premature? Argue with evidence.""",
    """A self-driving car must choose between hitting a pedestrian crossing illegally or swerving into a wall, risking the passenger’s life. Discuss how utilitarianism, deontology, and liability law (e.g., EU’s 2025 AV regulations) would resolve this. Should such decisions be standardized? Reference MIT’s Moral Machine experiment.""",
    """Analyze the dual-use potential of CRISPR-Cas9 and mRNA vaccine tech. Case studies: pandemic preparedness vs. bioengineered pathogens. Propose safeguards (e.g., gene-drive kill switches, AI-driven DNA synthesis screening) without stifling innovation.""",
    """Predict how AI tools (Copilot, ChatGPT, robotics) will reshape jobs in healthcare, law, and creative industries by 2035. Which roles are most resilient? Discuss UBI vs. reskilling as policy responses, citing Denmark’s flexicurity model.""",
    """Compare axion searches (ADMX) vs. WIMP detectors (XENONnT). Why has neither succeeded? Evaluate emerging theories (primordial black holes, modified gravity) and the role of JWST in indirect detection.""",
    """Assess non-financial blockchain applications: supply chain (IBM Food Trust), voting (Voatz), and IP management. Highlight scalability limits (TPS vs. energy use) and regulatory hurdles (GDPR compliance).""",
    """Contrast Global Workspace Theory with Integrated Information Theory. How do fMRI studies of vegetative patients support or challenge these models? Discuss implications for AI consciousness claims.""",
    """How might Europe differ if the Library of Alexandria had survived? Trace plausible impacts on the Renaissance, scientific method, and Islamic Golden Age knowledge transfer.""",
    """Should Stability AI or artists own rights to images trained on copyrighted works? Analyze the US Copyright Office’s 2023 ruling on ‘Zarya of the Dawn’ and EU’s proposed AI Act provisions.""",
    """Re-evaluate the Great Filter hypothesis in light of JWST’s exoplanet data. Are technosignatures (e.g., Dyson spheres) more likely to be detected via infrared or radio? Critique the ‘zoo hypothesis’.""",
    """How can ‘nudge theory’ combat vaccine hesitancy or climate inaction? Contrast the UK’s Nudge Unit with mandates. When do nudges become manipulative?""",
    """Break down the viability of SpaceX’s 1-million-person Mars city. Focus on radiation shielding, in-situ resource utilization (water extraction), and psychological risks of isolation.""",
    """Evaluate techniques to reduce factual errors in LLMs: retrieval-augmented generation (RAG), chain-of-thought prompting, and fine-tuning on curated datasets. Quantify trade-offs (latency, cost).""",
    """Reverse-engineer the Antikythera mechanism or Roman concrete. How could these technologies transform modern engineering if rediscovered earlier? Cite recent MIT/UC Berkeley research."""
]
    return random.choices(prompts, k=num_prompts)

def create_dummy_prompt(length: int) -> str:
    """Generate a prompt string of a given length."""
    tokens = [
        "pizza",
        "cheese",
        "the",
        "circle",
        ".",
        "12",
        "5",
        "by",
        "(",
        ")",
        ":",
        "Hello",
        "!",
    ]
    return " ".join(random.choices(tokens, k=length))

def send_chat_completion(url: str, model_name: str, prompt: str, max_tokens: int):
    """Send a chat completion request to the specified URL."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    try:
        response = requests.post(f"{url}/v1/chat/completions", headers=headers, data=json.dumps(payload))
        print("Status code:", response.status_code)
        print("Response:", response.json())
    except Exception as e:
        print(e)

def create_prompts(batch_size: int, prompt_type: str, prompt_len: int) -> List[str]:
    if prompt_type == "dummy":
        prompts = [create_dummy_prompt(prompt_len) for _ in range(batch_size)]
    elif prompt_type == "random":
        prompts = choose_random_prompt(batch_size)
    elif prompt_type == "long":
        prompts = [choose_long_prompt() for _ in range(batch_size)]
    elif prompt_type == "summary":
        prompts = choose_tokenized_prompt(prompt_len, batch_size)
    else:
        raise ValueError(f"Unsuported prompt type: {prompt_type}")
    return prompts

def main():
    parser = argparse.ArgumentParser(description="Send chat completion request.")
    parser.add_argument("--prompt-len", type=int, default=100, help="Length of the prompt to generate")
    parser.add_argument("--url", type=str, default="http://0.0.0.0:8005", help="Base URL of the API endpoint")
    parser.add_argument("--model-name", type=str, default="dummy_model", help="Model name to use")
    parser.add_argument("--max-tokens", type=int, default=10, help="Maximum tokens for completion")
    parser.add_argument("--batch-size", type=str, default="1", help="Number of parallel requests to send")
    parser.add_argument("--prompt-type", type=str, choices=["long", "random", "dummy", "summary"], default="dummy")
    args = parser.parse_args()

    parse_batch_size = args.batch_size.split(":")
    if len(parse_batch_size) > 3:
        raise ValueError(f"Unsupported batch_size format: {args.batch_size}")

    try:
        if len(parse_batch_size) == 1:
            batch_size_list = [int(parse_batch_size[0])]
        elif len(parse_batch_size) == 2:
            batch_size_list = list(range(int(parse_batch_size[0]), int(parse_batch_size[1]) + 1))
        else:
            start = int(parse_batch_size[0])
            stop = int(parse_batch_size[1]) + 1
            step = int(parse_batch_size[2])
            batch_size_list = list(range(start, stop, step))
    except Exception as e:
        raise ValueError(e)
    
    for file_to_delete in files_to_delete:
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


    for bs in tqdm(batch_size_list):
        prompts = create_prompts(bs, args.prompt_type, args.prompt_len)

        with ThreadPoolExecutor(max_workers=bs) as executor:
            futures = [
                executor.submit(send_chat_completion, args.url, args.model_name, prompt, args.max_tokens)
                for prompt in prompts
            ]
            for future in as_completed(futures):
                pass  # All output is handled in send_chat_completion
        # time.sleep(5)


if __name__ == "__main__":
    main()


#python3 send_request.py --url http://localhost:9111 --prompt-len 100 --max-tokens 100 --batch-size 1 --model-name /mnt/fs/Qwen3-32B-FP8 --prompt-type random
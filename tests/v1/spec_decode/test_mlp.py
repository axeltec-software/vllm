import os
os.environ["VLLM_USE_V1"] = "1"

import pytest

from pathlib import Path
from vllm.entrypoints.llm import LLM, SamplingParams


@pytest.mark.parametrize(
    "spec_model", [
        "drafters/Llama3.1_8b_instruct/mlp_pointwise_v1.1/",
    ])
def test_llm_with_speculator(spec_model: str):
    absolute_path = Path.cwd() / Path(spec_model)
    if not absolute_path.is_dir():
        raise ValueError(
            "Download mlp drafters first!\n"
            "First, get creds by:\n"
            "export ACCESS_KEY_ID=$(kubectl get secret -n build builder-s3-secret -o json | jq -r '.data.accessKeyID' | base64 -d)\n"
            "SECRET_ACCESS_KEY=$(kubectl get secret -n build builder-s3-secret -o json | jq -r '.data.secretAccessKey' | base64 -d)\n"
            "ENDPOINT=$(kubectl get secret -n build builder-s3-secret -o json | jq -r '.data.endpoint' | base64 -d)\n"
            "REGION=$(kubectl get secret -n build builder-s3-secret -o json | jq -r '.data.region' | base64 -d)\n"
            "Then dowload the models: \n"
            "aws s3 cp s3://studio-model-storage/drafters/Llama3.1_8b_instruct/mlp_based_v1.1/ drafters/Llama3.1_8b_instruct/mlp_based_v1.1/ --recursive\n"
        )
    llm = LLM(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        max_model_len=1024,
        speculative_config={
            "model": spec_model,
            "num_speculative_tokens": 3,
            "method": "mlp_speculator",
        },
        tensor_parallel_size=2,
        seed=0,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=200,
    )

    outputs = llm.generate(
        prompts=[
            "Hi! How are you doing?",
            "What's new?",
            "I'm a man in a hat ",
        ],
        use_tqdm=True,
        sampling_params=sampling_params,
    )

    assert outputs[0].outputs[0].text

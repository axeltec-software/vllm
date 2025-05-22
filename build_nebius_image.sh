#!/bin/bash

# Check if .aws_credentials file exists
if [ ! -f .aws_credentials ]; then
    echo "Error: .aws_credentials file is missing."
    echo "Please create .aws_credentials file with the following content:"
    echo "export AWS_ACCESS_KEY_ID=\"...\""
    echo "export AWS_SECRET_ACCESS_KEY=\"...\""
    echo "Environment values themselves can be fetched via \"kubectl --context ai-studio-testing -n build get secret builder-s3-secret -o yaml\""
    exit 1
fi

# Clean current directory from all development caches to make build clean
py3clean .

# Get current git branch and short commit hash
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
GIT_COMMIT=$(git rev-parse --short HEAD)
IMAGE_TAG="${GIT_BRANCH}.${GIT_COMMIT}"

# --build-arg max_jobs=16 --build-arg nvcc_threads=4
# Because 16 * 4 = 64 - num of threads, shouldn't be more than num of vCPUs
# torch_cuda_arch_list="8.9;9.0"
# Because of CUDA/GPU compatiblity matrix, see https://global.discourse-cdn.com/nvidia/original/4X/a/1/1/a115050c920924bd4cb211962a34de3c5e8500c2.png
DOCKER_BUILDKIT=1 docker build . \
  --file docker/Dockerfile \
  --target vllm-openai \
  --tag vllm/vllm-openai:${IMAGE_TAG} \
  --build-arg max_jobs=16 \
  --build-arg nvcc_threads=4 \
  --build-arg torch_cuda_arch_list="8.9;9.0" \
  --build-arg USE_SCCACHE=1 \
  --build-arg SCCACHE_BUCKET_NAME=vllm-build-cache \
  --build-arg SCCACHE_REGION_NAME=eu-north1 \
  --build-arg SCCACHE_S3_USE_SSL=true \
  --build-arg SCCACHE_ENDPOINT=https://storage.eu-north1.nebius.cloud \
  --secret id=aws_credentials,src=.aws_credentials
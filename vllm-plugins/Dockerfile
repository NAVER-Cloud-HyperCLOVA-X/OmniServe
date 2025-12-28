# vllm docker: Please see https://hub.docker.com/r/vllm/vllm-openai/tags
ARG VLLM_VERSION="v0.8.3"
FROM vllm/vllm-openai:${VLLM_VERSION} AS vllm-base

ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
WORKDIR /workspace

COPY . /workspace/HyperCLOVAX
RUN pip install -e /workspace/HyperCLOVAX/

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]

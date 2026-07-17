# Custom RunPod serverless worker for openbmb/MiniCPM-V-4_5.
#
# MiniCPM-V 4.5 (version tag "4,5") is only supported by vLLM >= 0.10.2 (merged
# to vLLM main on 2025-08-27). On older vLLM you get:
#   ValueError: MiniCPMV only supports versions 2.0, 2.5, 2.6, 4.0. Got version: (4, 5)
# This is a DIFFERENT pin from the Nanonets worker (which must stay on v0.9.2) —
# the two models have incompatible vLLM requirements, so they get separate images.
#
# Bases on the official vLLM OpenAI image so the vLLM version is guaranteed, then
# adds the same small RunPod handler that proxies jobs to the in-container vLLM
# OpenAI server on 127.0.0.1:8000.
FROM vllm/vllm-openai:v0.10.2

RUN pip install --no-cache-dir runpod requests
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('openbmb/MiniCPM-V-4_5')"

# Model + remote code are already baked into the image above. Without these,
# vLLM's --trust-remote-code path still calls out to the Hub on every worker
# boot to check for updated remote-code files; if that call hangs (HF
# unreachable/slow from the host), the worker sits in "initializing" forever
# with no crash and no unhealthy signal. Offline mode forces the local cache.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

WORKDIR /app
COPY handler.py /app/handler.py

# Override the base image's vLLM entrypoint; we launch vLLM ourselves from handler.py.
ENTRYPOINT []
CMD ["python3", "/app/handler.py"]

FROM vllm/vllm-openai:latest

RUN python3 -c "\
import importlib.metadata, subprocess, sys; \
from packaging.version import Version; \
current = Version(importlib.metadata.version('vllm')); \
required = Version('0.9.0'); \
print(f'Installed vLLM: {current}'); \
(subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', '--no-cache-dir', 'vllm', '--extra-index-url', 'https://wheels.vllm.ai/nightly']) \
    if current < required else print('Version sufficient, skipping upgrade.'))"

ENV HF_HOME=./.cache/huggingface

EXPOSE 8000
ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--model", "Qwen/Qwen3.5-9B", "--port", "8000", \
    "--host", "0.0.0.0", \
    "--tensor-parallel-size","2", \
    "--max-model-len", "32768", \
    "--reasoning-parser", "qwen3",\
    "--served-model-name", "qwen3.5-9b"]
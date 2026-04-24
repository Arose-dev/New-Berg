# Base image: RunPod PyTorch with CUDA 12.4
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /workspace

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget && \
    rm -rf /var/lib/apt/lists/*

# Install Qwen deps first (separate layer so it caches between code changes)
COPY requirements_qwen.txt .
RUN pip install --no-cache-dir -r requirements_qwen.txt

# flash-attn speeds up attention on Ampere+ GPUs; safe to skip if build fails
RUN pip install --no-cache-dir flash-attn --no-build-isolation || \
    echo "flash-attn not installed — continuing without it"

# Copy project code
COPY . .

# Cache HuggingFace downloads to the network volume mount point so they
# survive across pod restarts. Override with -e HF_HOME=... if needed.
ENV HF_HOME=/workspace/hf_cache
ENV TRANSFORMERS_CACHE=/workspace/hf_cache

# Default run: LEGO-Lite benchmark with Qwen3-VL-30B-A3B-Instruct.
# Override any flag via `docker run ... python run_lego_qwen.py --full`, etc.
CMD ["python", "run_lego_qwen.py"]

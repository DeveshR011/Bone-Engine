# =============================================================================
# Vectorless Semantic Search System — Dockerfile (CPU)
# =============================================================================
# Build:   docker build -t vectorless-search .
# Run:     docker run --rm -it vectorless-search python scripts/index_documents.py
#
# For GPU support, see the commented section at the bottom of this file,
# or refer to README.md § GPU Support.
# =============================================================================

FROM python:3.10-slim AS base

# ---- System dependencies ----------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        git \
        curl \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- Working directory -------------------------------------------------------
WORKDIR /app

# ---- Python dependencies (cached layer) -------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- Copy project source -----------------------------------------------------
COPY src/ src/
COPY scripts/ scripts/
COPY config.yaml config.yaml
COPY config.docker.yaml config.docker.yaml

# ---- Create data & results directories ---------------------------------------
RUN mkdir -p data results .cache/huggingface

# ---- Environment -------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

# ---- Expose API port (if running FastAPI) ------------------------------------
EXPOSE 8000

# ---- Default entrypoint ------------------------------------------------------
# Override with: docker run ... python scripts/<script>.py --config config.docker.yaml
CMD ["python", "scripts/index_documents.py", "--config", "config.docker.yaml"]


# =============================================================================
# GPU VARIANT
# =============================================================================
# To build a GPU-enabled image, replace the FROM line above with:
#
#   FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS base
#
#   # Install Python 3.10
#   RUN apt-get update && apt-get install -y --no-install-recommends \
#           software-properties-common \
#       && add-apt-repository ppa:deadsnakes/ppa \
#       && apt-get update && apt-get install -y --no-install-recommends \
#           python3.10 python3.10-venv python3.10-dev python3-pip \
#           build-essential gcc g++ git curl libpq-dev \
#       && ln -sf /usr/bin/python3.10 /usr/bin/python \
#       && ln -sf /usr/bin/pip3 /usr/bin/pip \
#       && rm -rf /var/lib/apt/lists/*
#
# Then install PyTorch with CUDA support before the rest of requirements:
#
#   RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121
#
# Build:
#   docker build -f Dockerfile.gpu -t vectorless-search:gpu .
#
# Run with GPU access:
#   docker run --gpus all --rm -it vectorless-search:gpu \
#       python scripts/benchmark_gpu_vs_cpu.py --config config.docker.yaml
#
# Prerequisites:
#   - NVIDIA GPU driver installed on the host
#   - NVIDIA Container Toolkit:
#       https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
# =============================================================================

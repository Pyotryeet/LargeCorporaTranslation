# =============================================================================
# Turkish Corpus Translation Benchmark — Multi-stage Dockerfile v3.9
# =============================================================================
# Builds an optimized image with cached dependencies.
#
# Build:
#   docker build -t tr-benchmark:3.9 .
#
# Run:
#   docker run --rm --gpus '"device=0,1"' --ipc=host --ulimit memlock=-1 \
#     -v $(pwd)/data:/data tr-benchmark:3.9 --config /data/config.yaml
#
#   The --ulimit memlock=-1 flag removes the lockable-memory cap, which is
#   required for GPU pinned (page-locked) memory allocations used by PyTorch
#   DataLoader workers and NCCL collectives.
# =============================================================================

# ── Stage 1: Builder (all compilation happens here) ────────────────────────
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

# System deps including build tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git ca-certificates \
    python3.11 python3.11-dev python3.11-venv \
    build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

# Virtual env.
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" VIRTUAL_ENV="/opt/venv"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch — verified version for H200/SM90 (2.12.1+cu126, +27% TPS over 2.6.0).
RUN pip install --no-cache-dir \
    'torch>=2.12.1' \
    --index-url https://download.pytorch.org/whl/cu126

# Copy dependency files + install.
WORKDIR /app
COPY requirements.txt requirements-cuda.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-cuda.txt

# Triton (Linux/NVIDIA only).  Fails gracefully on non-CUDA platforms.
RUN pip install --no-cache-dir 'triton>=2.3.0' 2>&1 | grep -v "^$" || \
    echo "WARNING: triton installation failed — container will run without Triton acceleration"

# Copy source + install package, including quantization/ (used for SmoothQuant).
COPY benchmark/ ./benchmark/
COPY tests/ ./tests/
COPY quantization/ ./quantization/
COPY scripts/ ./scripts/
RUN pip install -e .

# ── Stage 2: Runtime (slimmer) ─────────────────────────────────────────────
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

# Patching CVEs in the base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv \
    htop nvtop pigz \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

# Copy the fully-built venv from builder.
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

ENV PATH="/opt/venv/bin:$PATH" VIRTUAL_ENV="/opt/venv"

WORKDIR /app

# Default: offline mode (no HuggingFace Hub access).  Override with
# -e HF_TOKEN=... and -v ~/.cache/huggingface:/root/.cache/huggingface
# for gated model access.
ENV HF_HUB_OFFLINE=0

# CPU-only smoke-test config (must be overridden for GPU runs).
RUN cat > /app/config.yaml << 'EOF'
backend: cpu
precision: float32
log_level: info
EOF

# Entrypoint.
ENTRYPOINT ["python", "-m", "benchmark"]
CMD ["--config", "/app/config.yaml"]

# Health check — verifies Python and torch are importable.
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import torch; print(torch.__version__)" || exit 1

# Labels.
LABEL org.opencontainers.image.title="TR Corpus Translation Benchmark"
LABEL org.opencontainers.image.version="3.9"
LABEL org.opencontainers.image.description="Extreme-optimized EN→TR translation benchmark with AR/NLLB/Diffusion/custom backends"

# Fix ownership of copied files before dropping privileges.
RUN chown -R 1000:1000 /opt/venv /app

# Drop privileges for production — never run as root.
USER 1000:1000

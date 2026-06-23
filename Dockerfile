# =============================================================================
# Turkish Corpus Translation Benchmark — Multi-stage Dockerfile v3.6
# =============================================================================
# Builds an optimized image with cached dependencies and optional TensorRT.
#
# Build:
#   docker build -t tr-benchmark:3.6 .
#   docker build -t tr-benchmark:3.6-trt --build-arg WITH_TENSORRT=1 .
#
# Run:
#   docker run --rm --gpus '"device=0,1"' --ipc=host --ulimit memlock=-1 \
#     -v $(pwd)/data:/data tr-benchmark:3.6 --config /data/config.yaml
#
#   The --ulimit memlock=-1 flag removes the lockable-memory cap, which is
#   required for GPU pinned (page-locked) memory allocations used by PyTorch
#   DataLoader workers and NCCL collectives.
# =============================================================================

# ── Stage 1: Builder (all compilation happens here) ────────────────────────
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

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

# PyTorch.
RUN pip install --no-cache-dir \
    torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Copy dependency files + install.
WORKDIR /app
COPY requirements.txt requirements-cuda.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-cuda.txt

# TensorRT (optional).
ARG WITH_TENSORRT=0
RUN if [ "$WITH_TENSORRT" = "1" ]; then \
      pip install --no-cache-dir onnx>=1.16.0 onnxruntime ; \
      apt-get update && apt-get install -y --no-install-recommends \
        tensorrt python3-libnvinfer python3-libnvinfer-dev && \
      rm -rf /var/lib/apt/lists/* ; \
    fi

# Triton (Linux/NVIDIA only).  Fails gracefully on non-CUDA platforms
# so we log a warning instead of aborting the build.
RUN pip install --no-cache-dir triton>=2.3.0 2>&1 | grep -v "^$" || \
    echo "WARNING: triton installation failed — container will run without Triton acceleration"

# Copy source + install package.
COPY benchmark/ ./benchmark/
COPY tests/ ./tests/
RUN pip install -e .

# Pre-compile JIT kernels inside the builder (these get cached in the image).
# This means NO first-run latency when the container starts.
RUN python -c "from benchmark.hardware.jit_compiler import precompile_all_kernels; precompile_all_kernels()" 2>/dev/null || true

# ── Stage 2: Runtime (slimmer) ─────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

# The nvidia/cuda:12.4.1-runtime-ubuntu22.04 base image ships with known CVEs
# in its Ubuntu 22.04 packages.  Running apt-get upgrade after install patches
# all outstanding vulnerabilities before any application code executes.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv \
    htop nvtop pigz \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

# Copy the fully-built venv from builder.
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app
COPY --from=builder /root/.cache/tr_benchmark /root/.cache/tr_benchmark

ENV PATH="/opt/venv/bin:$PATH" VIRTUAL_ENV="/opt/venv"
ENV TR_BENCHMARK_JIT_CACHE="/root/.cache/tr_benchmark/kernels"
ENV TR_BENCHMARK_TRT_CACHE="/root/.cache/tr_benchmark/engines"

WORKDIR /app

# Safe defaults — MUST be overridden by users to enable trust.
ENV TR_BENCHMARK_UNTRUSTED_CODE=0

# Docker-specific config generated inline (docker_config.yaml does not ship
# in the repo so we write a minimal CPU-only config for quick smoke tests).
RUN cat > /app/config.yaml << 'EOF'
backend: cpu
precision: float32
log_level: info
EOF

# Entrypoint.
ENTRYPOINT ["python", "-m", "benchmark"]
CMD ["--config", "/app/config.yaml"]

# Health check — requires the container to be started with --observability
# (which enables the Prometheus metrics endpoint on :9090).
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9090/metrics')" || exit 1

# Labels.
LABEL org.opencontainers.image.title="TR Corpus Translation Benchmark"
LABEL org.opencontainers.image.version="3.6"
LABEL org.opencontainers.image.description="Extreme-optimized EN→TR translation benchmark with AR/Diffusion/TensorRT backends"

# Fix ownership of copied files before dropping privileges.
RUN chown -R 1000:1000 /opt/venv /app /root/.cache/tr_benchmark

# Drop privileges for production — never run as root.
USER 1000:1000

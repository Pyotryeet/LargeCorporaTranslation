#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Launch llama.cpp server with 1M context on 2×H200
#
# Usage:
#   ./scripts/launch_llama_server.sh [MODEL_KEY] [PORT]
#
#   MODEL_KEY defaults to GLM-5.2-UD-IQ2_M
#   PORT defaults to 8080
#
# Multi-GPU is ON by default (--split-mode row, tensor-split 0.5,0.5).
# Set LLAMA_SINGLE_GPU=1 to force single-GPU (offloads KV to RAM).
#
# Examples:
#   ./scripts/launch_llama_server.sh                           # auto 2×GPU
#   ./scripts/launch_llama_server.sh Qwen3-235B-A22B-IQ2_XXS 9090
#   LLAMA_SINGLE_GPU=1 ./scripts/launch_llama_server.sh        # single-GPU fallback
# ──────────────────────────────────────────────────────────────
set -euo pipefail

MODEL_KEY="${1:-UD-IQ2_M}"
PORT="${2:-8080}"
HOST="${3:-0.0.0.0}"

LLAMA_MODEL_ROOT="${LLAMA_MODEL_ROOT:-${HOME}/models/GLM-5.2-GGUF}"
MODEL_DIR="${LLAMA_MODEL_ROOT}/${MODEL_KEY}"
# Try the standard filename pattern first, then fall back to glob
MODEL_FILE="${MODEL_DIR}/GLM-5.2-${MODEL_KEY}-00001-of-00006.gguf"

if [ ! -f "${MODEL_FILE}" ]; then
    # Pattern didn't match — the key might already include the model prefix.
    # Try matching any .gguf shard in the directory.
    FIRST_SHARD=$(ls -1 "${MODEL_DIR}"/*.gguf 2>/dev/null | head -1)
    if [ -n "${FIRST_SHARD}" ]; then
        MODEL_FILE="${FIRST_SHARD}"
    fi
fi

if [ ! -f "${MODEL_FILE}" ]; then
    echo "❌ Model not found: ${MODEL_FILE}"
    echo "   Available models under ${HOME}/models/GLM-5.2-GGUF/:"
    ls -1 "${HOME}/models/GLM-5.2-GGUF/" 2>/dev/null || echo "   (none)"
    exit 1
fi

# ── GPU detection ────────────────────────────────────────────
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || wc -l || echo 0)
CTX_SIZE=1048576
GPU_LAYERS=99
BATCH_SIZE=2048
UBATCH_SIZE=512

SINGLE_GPU="${LLAMA_SINGLE_GPU:-0}"

echo "═══ llama.cpp server ═══"
echo "  model:       ${MODEL_FILE}"
echo "  host:        ${HOST}:${PORT}"
echo "  ctx size:    1,048,576 (1M)"
echo "  gpu layers:  99"
echo "  flash attn:  on"
echo "  kv cache:    q4_0 (GPU resident)"

if [ "${SINGLE_GPU}" = "1" ]; then
    echo "  mode:        SINGLE-GPU (KV → RAM, no tensor split)"
    echo "═══════════════════════════"
    echo ""
    exec ~/llama.cpp/llama-server \
        --model "${MODEL_FILE}" \
        --host "${HOST}" \
        --port "${PORT}" \
        --n-gpu-layers "${GPU_LAYERS}" \
        --ctx-size "${CTX_SIZE}" \
        --cache-type-k q4_0 \
        --cache-type-v q4_0 \
        --flash-attn on \
        --no-kv-offload \
        --cache-ram 400000 \
        --temp 1.0 \
        --top-p 0.95 \
        --min-p 0.01
else
    echo "  mode:        MULTI-GPU (row split, 0.5/0.5 tensor)"
    echo "  batch:       ${BATCH_SIZE} / ubatch ${UBATCH_SIZE}"
    echo "═══════════════════════════"
    echo ""
    exec ~/llama.cpp/llama-server \
        --model "${MODEL_FILE}" \
        --host "${HOST}" \
        --port "${PORT}" \
        --n-gpu-layers "${GPU_LAYERS}" \
        --ctx-size "${CTX_SIZE}" \
        --cache-type-k q4_0 \
        --cache-type-v q4_0 \
        --flash-attn on \
        --split-mode row \
        --tensor-split 0.5,0.5 \
        --batch-size "${BATCH_SIZE}" \
        --ubatch-size "${UBATCH_SIZE}" \
        --temp 1.0 \
        --top-p 0.95 \
        --min-p 0.01
fi

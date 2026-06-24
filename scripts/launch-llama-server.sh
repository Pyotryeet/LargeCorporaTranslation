#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Launch llama.cpp server with 1M context on H200
#
# Usage:
#   ./scripts/launch-llama-server.sh [MODEL_KEY] [PORT]
#
#   MODEL_KEY defaults to GLM-5.2-UD-IQ2_M
#   PORT defaults to 8080
#
# Examples:
#   ./scripts/launch-llama-server.sh                    # defaults
#   ./scripts/launch-llama-server.sh Qwen3-235B-A22B-IQ2_XXS 9090
# ──────────────────────────────────────────────────────────────
set -euo pipefail

MODEL_KEY="${1:-GLM-5.2-UD-IQ2_M}"
PORT="${2:-8080}"
HOST="${3:-0.0.0.0}"

MODEL_DIR="${HOME}/models/GLM-5.2-GGUF/${MODEL_KEY}"
MODEL_FILE="${MODEL_DIR}/GLM-5.2-${MODEL_KEY}-00001-of-00006.gguf"

if [ ! -f "${MODEL_FILE}" ]; then
    echo "❌ Model not found: ${MODEL_FILE}"
    echo "   Available models under ${HOME}/models/GLM-5.2-GGUF/:"
    ls -1 "${HOME}/models/GLM-5.2-GGUF/" 2>/dev/null || echo "   (none)"
    exit 1
fi

echo "═══ llama.cpp server ═══"
echo "  model:      ${MODEL_FILE}"
echo "  host:       ${HOST}:${PORT}"
echo "  ctx size:   1,048,576 (1M)"
echo "  gpu layers: 99"
echo "  flash attn: on"
echo "  kv cache:   q4_0 (not offloaded)"
echo "  cache ram:  400 GB"
echo "  thinking:   enabled (chat-template native)"
echo "═══════════════════════════"
echo ""

exec ~/llama.cpp/llama-server \
    --model "${MODEL_FILE}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --n-gpu-layers 99 \
    --ctx-size 1048576 \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --flash-attn on \
    --no-kv-offload \
    --cache-ram 400000 \
    --temp 1.0 \
    --top-p 0.95 \
    --min-p 0.01

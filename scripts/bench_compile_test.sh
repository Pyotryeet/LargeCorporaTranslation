#!/bin/bash
# Quick compile test on single GPU
set -e
cd ~/LargeCorporaTranslation
source .venv/bin/activate
export NVIDIA_TF32_OVERRIDE=1

echo "=== Compile test: PyTorch 2.12.1 ==="
CUDA_VISIBLE_DEVICES=1 TR_SKIP_FP8=1 \
timeout 180 python3 -m benchmark \
  --model translategemma-4b-bf16 --dry-run --batch-size 32 \
  2>&1 | grep -v 'FutureWarning\|pynvml\|config.json\|special_tokens\|chat_template\|You are\|Downloading\|will be\|HTTP Request\|WARNING: Running\|shards\|torch_dtype' | grep -E 'torch.compile|warmup complete|Starting|batches=|tps=|Complete|Estimated|BERTScore|tokens'

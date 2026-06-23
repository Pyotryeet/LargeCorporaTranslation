#!/usr/bin/env bash
# =============================================================================
# Sequential 5-Minute Benchmark — 5 New Models on MPS
# =============================================================================
# Usage: bash benchmarks/run_new_models.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

R='\033[0;31m'; G='\033[0;32m'; C='\033[0;36m'; B='\033[1m'; N='\033[0m'
banner(){ echo -e "\n${C}══════════════════════════════════════════════════${N}"; echo -e "${C}  $*${N}"; echo -e "${C}══════════════════════════════════════════════════${N}\n"; }
ok(){ echo -e "  ${G}✓${N} $*"; }
fail(){ echo -e "  ${R}✗${N} $*"; }

OUTPUT_DIR="data/output/benchmark_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
RESULTS_FILE="$OUTPUT_DIR/results.jsonl"
: > "$RESULTS_FILE"

# ── Model list: label → model_path ──────────────────────────────────────────
declare -A MODELS=(
    ["DiffusionGemma 26B-A4B"]="google/diffusiongemma-26B-A4B-it"
    ["Gemma 4 E2B QAT"]="google/gemma-4-E2B-it-qat-mobile-ct"
    ["Gemma 4 E4B QAT"]="google/gemma-4-E4B-it-qat-mobile-ct"
    ["Gemma 4 E2B Q4_0"]="google/gemma-4-E2B-it-qat-mobile-transformers"
    ["Gemma 4 E4B Q4_0"]="google/gemma-4-E4B-it-qat-mobile-transformers"
)

MODEL_ORDER=(
    "DiffusionGemma 26B-A4B"
    "Gemma 4 E2B QAT"
    "Gemma 4 E4B QAT"
    "Gemma 4 E2B Q4_0"
    "Gemma 4 E4B Q4_0"
)

TOTAL=${#MODEL_ORDER[@]}
CURRENT=0
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-.venv}"

# Activate venv
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
fi

banner "SEQUENTIAL 5-MINUTE BENCHMARK — ${TOTAL} Models on MPS"

# ── Quick system check ──────────────────────────────────────────────────────
$PYTHON_BIN -c "
import torch, psutil
print(f'  PyTorch: {torch.__version__}')
print(f'  MPS:     {torch.backends.mps.is_available()}')
print(f'  RAM:     {psutil.virtual_memory().total/(1024**3):.0f} GB')
"

# ── Run each model ──────────────────────────────────────────────────────────
for MODEL_LABEL in "${MODEL_ORDER[@]}"; do
    CURRENT=$((CURRENT + 1))
    MODEL_PATH="${MODELS[$MODEL_LABEL]}"
    MODEL_OUTDIR="$OUTPUT_DIR/run_${CURRENT}"
    mkdir -p "$MODEL_OUTDIR"

    banner "[$CURRENT/$TOTAL] $MODEL_LABEL"

    # Determine if diffusion
    if [[ "$MODEL_PATH" == *diffusiongemma* ]]; then
        BACKEND_ARG="--diffusion"
    else
        BACKEND_ARG=""
    fi

    # Write runtime config
    CONFIG_FILE="$MODEL_OUTDIR/runtime_config.yaml"
    cat > "$CONFIG_FILE" << YAMLEND
backend: "mps"
model:
  model_path: "$MODEL_PATH"
  tokenizer_path: ""
  max_input_tokens: 256
  max_new_tokens: 128
  temperature: 0.0
  do_sample: false
  num_beams: 1
  dtype: "bfloat16"
  tensor_parallel_size: 1
  use_flash_attention: false
  backend_type: "auto"
  diffusion_steps: 128
  guidance_scale: 2.0
  noise_schedule: "linear"
runtime:
  target_duration_seconds: 300
  checkpoint_interval_seconds: 120
  heartbeat_interval_seconds: 20
  seed: 42
data:
  input_paths: ["./data/input/fineweb_en_sample.jsonl.gz"]
  output_dir: "$MODEL_OUTDIR"
  reference_set_path: "./data/references/golden_en_tr.jsonl"
  prefetch_workers: 2
  shuffle: false
  min_chunk_tokens: 10
  max_garbage_ratio: 0.95
  chunk_overlap_tokens: 50
extrapolation:
  total_clearnet_non_tr_tokens: 6230000000000
YAMLEND

    echo -e "  ${C}Config:${N} $CONFIG_FILE"
    echo -e "  ${C}Model:${N}  $MODEL_PATH"

    RUN_START=$(date +%s)

    # Suppress noisy warnings
    export TORCH_LOGS="-dynamo"
    export PYTHONWARNINGS="ignore::FutureWarning"

    set +e
    $PYTHON_BIN -m benchmark \
        --config "$CONFIG_FILE" \
        --quick --translate-only --mps-safe \
        2>&1 | tee "$MODEL_OUTDIR/benchmark.log"
    EXIT_CODE=$?
    set -e

    RUN_END=$(date +%s)
    RUN_DURATION=$((RUN_END - RUN_START))

    # Extract throughput
    MEAN_TPS="N/A"
    REPORT_JSON="$MODEL_OUTDIR/report/benchmark_report.json"
    if [ -f "$REPORT_JSON" ]; then
        MEAN_TPS=$($PYTHON_BIN -c "
import json, sys
try:
    with open('$REPORT_JSON') as f:
        r = json.load(f)
    tps = r.get('metrics',{}).get('batch',{}).get('mean_tps', 'N/A')
    print(tps)
except: print('N/A')
" 2>/dev/null || echo "N/A")
    fi

    # Handle missing report (model may not have downloaded, etc.)
    if [ "$MEAN_TPS" = "N/A" ] || [ "$MEAN_TPS" = "None" ]; then
        MEAN_TPS="N/A"
    fi

    if [ $EXIT_CODE -eq 0 ]; then
        ok "Completed in ${RUN_DURATION}s | Mean TPS: ${MEAN_TPS}"
    else
        fail "Exit code $EXIT_CODE | Duration: ${RUN_DURATION}s | TPS: ${MEAN_TPS}"
    fi

    # Save result
    echo "{\"model\":\"$MODEL_LABEL\",\"path\":\"$MODEL_PATH\",\"exit\":$EXIT_CODE,\"duration_s\":$RUN_DURATION,\"mean_tps\":\"$MEAN_TPS\"}" >> "$RESULTS_FILE"

    # Aggressively free memory between runs
    $PYTHON_BIN -c "
import gc, torch
gc.collect()
if hasattr(torch.mps, 'empty_cache'):
    torch.mps.empty_cache()
print('MPS cache cleared')
" 2>/dev/null || true

    echo ""
    sleep 5
done

# ── Final Summary ───────────────────────────────────────────────────────────
banner "BENCHMARK SUMMARY — All Results"

$PYTHON_BIN << PYEOF
import json, sys

results = []
with open("$RESULTS_FILE") as f:
    for line in f:
        line = line.strip()
        if line:
            results.append(json.loads(line))

print(f"  Total runs: {len(results)}")
print()
header = f"  {'Model':<30s} {'Status':>7s} {'Time':>7s} {'TPS':>10s}"
print(header)
print(f"  {'-'*len(header)}")
for r in results:
    status = "OK" if r['exit'] == 0 else f"ERR({r['exit']})"
    tps = r.get('mean_tps', 'N/A')
    if isinstance(tps, (int, float)):
        tps_str = f"{tps:>8.0f}"
    else:
        tps_str = f"{tps:>8s}"
    print(f"  {r['model']:<30s} {status:>7s} {str(r['duration_s'])+'s':>7s} {tps_str} tok/s")

print()
print(f"  Full logs: $OUTPUT_DIR/")
PYEOF

echo ""
banner "Done"
echo "  Results: $OUTPUT_DIR/"
echo "  Summary: $RESULTS_FILE"

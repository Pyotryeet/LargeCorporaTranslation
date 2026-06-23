#!/bin/bash
# TR Corpus Translation Benchmark — Complete Model Comparison
# All models sourced from official Google and Facebook HuggingFace repos
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

OUT_DIR="data/output"
mkdir -p "$OUT_DIR"

TOTAL_START=$(date +%s)

echo "======================================================================"
echo "  TR Corpus Translation Benchmark — Complete Model Comparison"
echo "  Platform: MPS (Apple Silicon)"
echo "  Duration per model: 120s translation + BERTScore quality eval"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"

# Phase 1: NLLB family (best translators, fastest paths)
PYTHON_MODELS=(
    "nllb_600m"
    "nllb_1.3b"
    "nllb_3.3b"
    "smollm2"
    "translategemma"
)

# Phase 2: Google QAT CT/mobile models (Python backend — may fail, loads BF16)
PYTHON_QAT_MODELS=(
    "gemma_e4b_qat_mobile_ct"
    "gemma_e2b_qat_mobile_ct"
    "gemma_e4b_qat_mobile_transformers"
    "gemma_e2b_qat_mobile_transformers"
    "gemma_e4b_qat_w4a16_ct"
    "gemma_e2b_qat_w4a16_ct"
)

# Phase 3: llama.cpp GGUF models (Google QAT + DiffusionGemma)
# Only include models whose GGUF files exist or can be downloaded
LLAMA_MODELS=()

# Google QAT GGUF — check if files exist (or will be auto-downloaded)
GOOGLE_QAT_DIR="$HOME/Documents/ComputerScience/Projects/llama/models/google_qat"
if [ -f "$GOOGLE_QAT_DIR/gemma-4-E2B_q4_0-it.gguf" ] || \
   python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('google/gemma-4-E2B-it-qat-q4_0-gguf', 'gemma-4-E2B_q4_0-it.gguf', local_dir='$GOOGLE_QAT_DIR', resume_download=True)" 2>/dev/null; then
    LLAMA_MODELS+=("gemma_e2b_qat_q4_0_gguf")
fi

if [ -f "$GOOGLE_QAT_DIR/gemma-4-E4B_q4_0-it.gguf" ] || \
   python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('google/gemma-4-E4B-it-qat-q4_0-gguf', 'gemma-4-E4B_q4_0-it.gguf', local_dir='$GOOGLE_QAT_DIR', resume_download=True)" 2>/dev/null; then
    LLAMA_MODELS+=("gemma_e4b_qat_q4_0_gguf")
fi

if [ -f "$GOOGLE_QAT_DIR/gemma-4-26B_q4_0-it.gguf" ] || \
   python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('google/gemma-4-26B-A4B-it-qat-q4_0-gguf', 'gemma-4-26B_q4_0-it.gguf', local_dir='$GOOGLE_QAT_DIR', resume_download=True)" 2>/dev/null; then
    LLAMA_MODELS+=("gemma_26b_a4b_qat_q4_0_gguf")
fi

# DiffusionGemma
if [ -f "$HOME/Documents/ComputerScience/Projects/llama/models/diffusiongemma-26B-A4B-it-Q8_0.gguf" ]; then
    LLAMA_MODELS+=("diffusiongemma")
fi

echo "  Python models (NLLB+AR): ${#PYTHON_MODELS[@]}"
echo "  Python QAT models:       ${#PYTHON_QAT_MODELS[@]}"
echo "  llama.cpp GGUF models:   ${#LLAMA_MODELS[@]}"
echo "  Total:                   $((${#PYTHON_MODELS[@]} + ${#PYTHON_QAT_MODELS[@]} + ${#LLAMA_MODELS[@]}))"

# ── Run Python models (NLLB + autoregressive) ──
for i in "${!PYTHON_MODELS[@]}"; do
    model="${PYTHON_MODELS[$i]}"
    idx=$((i + 1))
    echo ""
    echo "===== [Python $idx/${#PYTHON_MODELS[@]}] $model ====="
    python -u scripts/run_one_model.py "$model" 2>&1 | tee "/tmp/bm_${model}.log" || true
    # Memory cleanup between runs
    python3 -c "import gc; gc.collect(); import torch; torch.mps.empty_cache()" 2>/dev/null || true
    sleep 2
done

# ── Run Python QAT models (CT/mobile — expected to fail, but we try) ──
for i in "${!PYTHON_QAT_MODELS[@]}"; do
    model="${PYTHON_QAT_MODELS[$i]}"
    idx=$((i + 1))
    echo ""
    echo "===== [Python QAT $idx/${#PYTHON_QAT_MODELS[@]}] $model ====="
    python -u scripts/run_one_model.py "$model" 2>&1 | tee "/tmp/bm_${model}.log" || true
    python3 -c "import gc; gc.collect(); import torch; torch.mps.empty_cache()" 2>/dev/null || true
    sleep 2
done

# ── Run llama.cpp GGUF models ──
for i in "${!LLAMA_MODELS[@]}"; do
    model="${LLAMA_MODELS[$i]}"
    idx=$((i + 1))
    echo ""
    echo "===== [llama.cpp $idx/${#LLAMA_MODELS[@]}] $model ====="
    python -u scripts/run_one_model.py "$model" 2>&1 | tee "/tmp/bm_${model}.log" || true
    python3 -c "import gc; gc.collect(); import torch; torch.mps.empty_cache()" 2>/dev/null || true
    sleep 2
done

TOTAL_END=$(date +%s)
ELAPSED=$((TOTAL_END - TOTAL_START))

# ═════════════════════════════════════════════════════════════════════════════
# Assemble final report
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo "======================================================================"
echo "  ASSEMBLING FINAL REPORT"
echo "  Total time: $((ELAPSED / 60)) min $((ELAPSED % 60)) sec"
echo "======================================================================"

python3 << PYEOF
import json, glob, os
from datetime import datetime, timezone

out_dir = "data/output"
results = []

for f in sorted(glob.glob(f"{out_dir}/result_*.json")):
    with open(f) as fh:
        r = json.load(fh)
    results.append(r)

if not results:
    print("⚠ No result files found!")
    exit(1)

# Results table
hdr = f"\n{'MODEL':<42} {'TPS':>8} {'Lat(ms)':>9} {'B':>4}  {'BERTScore':>10} {'Load(s)':>8}  Status"
sep = "=" * len(hdr)
print(sep)
print(hdr)
print(sep)

ok = 0; fail = 0
for r in results:
    n = r.get("model", "?")[:41]
    if r.get("error"):
        err = r['error'][:40]
        print(f"{n:<42} {'—':>8} {'—':>9} {'—':>4}  {'—':>10} {'—':>8}  ✗ {err}")
        fail += 1
    else:
        tps = r.get("mean_tps", 0)
        lat = r.get("mean_latency_ms", 0)
        bs_b = r.get("batch_size", "?")
        bert = r.get("bertscore") or 0
        load = r.get("load_seconds", 0)
        print(f"{n:<42} {tps:>8.1f} {lat:>9.0f} {str(bs_b):>4}  {bert:>10.4f} {load:>8.1f}  ✓")
        ok += 1
print(sep)
print(f"  ✓ {ok} succeeded  ✗ {fail} failed")

# Rankings
ok_results = [r for r in results if not r.get("error")]
if ok_results:
    by_tps = sorted(ok_results, key=lambda r: r.get("mean_tps", 0), reverse=True)
    by_bert = sorted([r for r in ok_results if r.get("bertscore")],
                    key=lambda r: r.get("bertscore", 0), reverse=True)

    if by_tps:
        print(f"\n═══ TPS RANKING (tokens/sec) ═══")
        for j, r in enumerate(by_tps, 1):
            tps = r.get("mean_tps", 0)
            bert = r.get("bertscore") or "—"
            bert_str = f"{bert:.4f}" if isinstance(bert, (int, float)) else str(bert)
            extra = " ★ BEST" if j == 1 else ""
            print(f"  {j:2d}. {r['model']:<42} {tps:>8.0f} tok/s  BERTScore={bert_str}{extra}")

    if by_bert:
        print(f"\n═══ QUALITY RANKING (BERTScore) ═══")
        for j, r in enumerate(by_bert, 1):
            tps = r.get("mean_tps", 0)
            bert = r.get("bertscore", 0)
            extra = " ★ BEST" if j == 1 else ""
            print(f"  {j:2d}. {r['model']:<42} BERTScore={bert:.4f}  TPS={tps:.0f}{extra}")

    # Speed-Quality Pareto
    print(f"\n═══ PARETO FRONTIER (trade-off analysis) ═══")
    for r in ok_results:
        tps = r.get("mean_tps", 0)
        bert = r.get("bertscore") or 0
        # TPS-per-quality-point
        if bert > 0.5:
            efficiency = tps / bert
            print(f"  {r['model']:<42} {tps:>8.0f} tok/s @ {bert:.4f} BERTScore  →  {efficiency:.0f} tok/s per quality point")

# Write final report
report = {
    "title": "TR Corpus Translation — Complete Model Comparison",
    "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "platform": "MPS (Apple Silicon)",
    "pytorch": os.popen("python3 -c 'import torch; print(torch.__version__)' 2>/dev/null").read().strip(),
    "config": {"run_duration_per_model_s": 120, "quality_max_references": 32},
    "models_tested": len(results),
    "models_succeeded": ok,
    "models_failed": fail,
    "results": results,
}
with open(f"{out_dir}/model_comparison.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"\n✓ Full report: {out_dir}/model_comparison.json")
PYEOF

echo ""
echo "======================================================================"
echo "  BENCHMARK COMPLETE"
echo "======================================================================"

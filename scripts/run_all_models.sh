#!/bin/bash
# TR Corpus Translation Benchmark — Complete Model Comparison
# Works on: CUDA (H200), MPS (Apple Silicon), CPU
set -euo pipefail

cd "$(dirname "$0")/.."

# Detect platform
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    PLATFORM="cuda"
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "Platform: CUDA — $GPU_NAME"
elif python3 -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
    PLATFORM="mps"
    echo "Platform: MPS (Apple Silicon)"
else
    PLATFORM="cpu"
    echo "Platform: CPU"
fi

OUT_DIR="data/output"
mkdir -p "$OUT_DIR"
TOTAL_START=$(date +%s)

echo "======================================================================"
echo "  TR Corpus Translation Benchmark — Complete Model Comparison"
echo "  Platform: $PLATFORM"
echo "  Duration per model: 120s translation + BERTScore quality eval"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"

# ── Model list — user-specified execution order ──
# Phase 1: SmolLM2 (crash-prone — FP8 + fused kernel + compile, test first)
PYTHON_MODELS=(
    "smollm2"
)

# Phase 2: NLLB family (Meta/Facebook — proven EN→TR, fastest translators)
PYTHON_MODELS+=(
    "nllb_600m"
    "nllb_1.3b"
    "nllb_3.3b"
)

# Phase 3: MADLAD-400 (Google T5-based, 450-language MT)
PYTHON_MODELS+=(
    "madlad_3b"
    "madlad_10b"
)

# Phase 4: TranslateGemma-4B (Google autoregressive translator)
PYTHON_MODELS+=(
    "translategemma"
)

TOTAL=${#PYTHON_MODELS[@]}
echo "  Total models: $TOTAL"
echo "    SmolLM2:          1"
echo "    NLLB (Meta):      3"
echo "    MADLAD-400:       2"
echo "    TranslateGemma:   1"

# ── Run all models ──
for i in "${!PYTHON_MODELS[@]}"; do
    model="${PYTHON_MODELS[$i]}"
    idx=$((i + 1))
    echo ""
    echo "===== [$idx/$TOTAL] $model ====="
    python -u scripts/benchmark_single.py "$model" 2>&1 | tee "/tmp/bm_${model}.log" || true
    python3 -c "import gc; gc.collect(); import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true
    sleep 2
done

TOTAL_END=$(date +%s)
ELAPSED=$((TOTAL_END - TOTAL_START))

# ═════════════════════════════════════════════════════════════════════════════
# Final report
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo "======================================================================"
echo "  BENCHMARK COMPLETE — $((ELAPSED / 60)) min $((ELAPSED % 60)) sec"
echo "======================================================================"

python3 << PYEOF
import json, glob, os
from datetime import datetime, timezone

results = []
for f in sorted(glob.glob("data/output/result_*.json")):
    with open(f) as fh:
        r = json.load(fh)
    results.append(r)

if not results:
    print("⚠ No results found!")
    exit(1)

ok = [r for r in results if not r.get("error")]
fail = [r for r in results if r.get("error")]

# Table
hdr = f"\n{'MODEL':<42} {'TPS':>9} {'Lat(ms)':>10} {'Batch':>6}  {'BERTScore':>10} {'Load(s)':>8}  Status"
sep = "=" * len(hdr)
print(sep)
print(hdr)
print(sep)

for r in sorted(results, key=lambda r: r.get("mean_tps", 0) or 0, reverse=True):
    n = r.get("model", "?")[:41]
    if r.get("error"):
        print(f"{n:<42} {'—':>9} {'—':>10} {'—':>6}  {'—':>10} {'—':>8}  ✗ {r['error'][:40]}")
    else:
        tps = r.get("mean_tps", 0)
        lat = r.get("mean_latency_ms", 0)
        bs = r.get("batch_size", "?")
        bert = r.get("bertscore")
        bstr = f"{bert:.4f}" if bert is not None else "—"
        load = r.get("load_seconds", 0)
        print(f"{n:<42} {tps:>9.1f} {lat:>10.0f} {str(bs):>6}  {bstr:>10} {load:>8.1f}  ✓")

print(sep)
print(f"  ✓ {len(ok)} succeeded  ✗ {len(fail)} failed  Total: {len(results)}")

# Rankings
if ok:
    by_tps = sorted(ok, key=lambda r: r.get("mean_tps", 0), reverse=True)
    by_bert = sorted([r for r in ok if r.get("bertscore")], key=lambda r: r.get("bertscore", 0), reverse=True)

    print(f"\n═══ TPS RANKING ═══")
    for j, r in enumerate(by_tps, 1):
        tps = r.get("mean_tps", 0)
        bert = r.get("bertscore")
        bstr = f" BERT={bert:.4f}" if bert else ""
        flag = " ★ FASTEST" if j == 1 else ""
        print(f"  {j:2d}. {r['model']:<42} {tps:>9.0f} tok/s{bstr}{flag}")

    if by_bert:
        print(f"\n═══ QUALITY RANKING (BERTScore) ═══")
        for j, r in enumerate(by_bert, 1):
            print(f"  {j:2d}. {r['model']:<42} {r.get('bertscore',0):.4f} — TPS={r.get('mean_tps',0):.0f}")

    # Pareto
    print(f"\n═══ EFFICIENCY (TPS per quality point) ═══")
    pareto = sorted([r for r in ok if r.get("bertscore", 0) > 0.6],
                    key=lambda r: r.get("mean_tps",0) / max(r.get("bertscore",0.001),0.001), reverse=True)
    for j, r in enumerate(pareto, 1):
        eff = r.get("mean_tps",0) / max(r.get("bertscore",0.001),0.001)
        print(f"  {j:2d}. {r['model']:<42} {eff:>8.0f} eff  ({r.get('mean_tps',0):.0f} TPS / {r.get('bertscore',0):.4f})")

# Write final JSON
report = {
    "title": "TR Corpus Translation — Model Comparison",
    "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "platform": "$PLATFORM",
    "results": results,
}
with open("data/output/model_comparison.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"\n✓ Full report: data/output/model_comparison.json")
PYEOF

echo ""
echo "Done."

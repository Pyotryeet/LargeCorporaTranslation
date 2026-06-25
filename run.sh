#!/usr/bin/env bash
# =============================================================================
#  run.sh — Unified benchmark launcher for ALL platforms & modes
# =============================================================================
#  Auto-detects macOS MPS vs Linux CUDA vs CPU.  One script, one interface.
#
#  Usage (common):
#    ./run.sh                            # Full benchmark (auto-detect platform)
#    ./run.sh --dry-run                  # 60-second smoke test
#    ./run.sh --quick                    # 5-minute evaluation
#    ./run.sh --model 12B               # Select model size (4B|12B|27B|E2B|E4B|E2B-Q4_0|E4B-Q4_0|26B-A4B)
#    ./run.sh --multi-gpu               # Run 1 model copy per GPU (data-parallel)
#    ./run.sh --diffusion               # Use diffusion model (LLaDA 8B)
#    ./run.sh --tensorrt                # TensorRT acceleration (CUDA only)
#    ./run.sh --precompile              # Pre-compile all kernels, then exit
#    ./run.sh --observability           # Enable Prometheus dashboard on :9090
#    ./run.sh --batch-size 128          # Force batch size
#    ./run.sh --duration 3600           # 1-hour run
#    ./run.sh --resume output/dir/      # Resume from checkpoint
#
#  Quick combinations:
#    ./run.sh --quick --tensorrt        # Fast eval with TRT
#    ./run.sh --diffusion --quick       # Test diffusion model
#    ./run.sh --full --observability    # Production with live dashboard
#    ./run.sh --no-compile --multi-gpu  # 2× throughput on dual GPU
# =============================================================================

# ═══════════════════════════════════════════════════════════════════════════
# FORK BOMB GUARD — must be at the top of the file, before any logic.
# The auto-shard path calls "$0" to spawn child processes.  If a child
# re-enters the auto-shard path, it would recursively spawn indefinitely.
# TR_SHARD=N is set on children to prevent re-entry.
# TR_DEPTH=N tracks nesting and kills at depth > 1.
# ═══════════════════════════════════════════════════════════════════════════
if [ -n "${TR_DEPTH:-}" ] && [ "${TR_DEPTH:-0}" -ge 2 ]; then
    echo "FATAL: recursive invocation detected (TR_DEPTH=$TR_DEPTH). Aborting." >&2
    exit 1
fi
_TR_DEPTH=$(( ${TR_DEPTH:-0} + 1 ))
export TR_DEPTH=$_TR_DEPTH
# ── End fork bomb guard ───────────────────────────────────────────────────

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colors ─────────────────────────────────────────────────────────────────
R='\033[0;31m'; G='\033[0;32m'; Y='\033[1;33m'; C='\033[0;36m'; B='\033[1m'; N='\033[0m'
banner(){ echo -e "\n${C}══════════════════════════════════════════════════${N}"; echo -e "${C}  $*${N}"; echo -e "${C}══════════════════════════════════════════════════${N}\n"; }
ok(){ echo -e "  ${G}✓${N} $*"; }
warn(){ echo -e "  ${Y}⚠${N} $*"; }
fail(){ echo -e "  ${R}✗${N} $*"; exit 1; }
info(){ echo -e "  ${C}→${N} $*"; }

# ═══════════════════════════════════════════════════════════════════════════
# Parse flags
# ═══════════════════════════════════════════════════════════════════════════

# Run mode.
MODE="full"                    # full | quick | dry-run | benchmark-only | translate-only | warmup-only | precompile
MODEL_SIZE="4B"               # 4B | E2B | E4B | E2B-Q4_0 | E4B-Q4_0 | 26B-A4B | ministral-3b
BACKEND="auto"                 # auto | autoregressive | diffusion | tensorrt
DURATION=""                    # Override seconds
BATCH_SIZE=""                  # Override batch size
RESUME_DIR=""                  # Resume path
CONFIG_FILE=""                 # Explicit config file (overrides auto-gen)
OBSERVABILITY=false            # Enable Prometheus
FORCE_RECOMPILE=false          # Force JIT + TRT recompilation
SPECULATIVE=false              # Enable speculative decoding
SPEC_MODE="self"               # self | draft_model
SPEC_TOKENS=3                  # K speculative tokens
SPEC_DRAFT_LAYERS=0            # 0 = auto
PYTHON_BIN="${PYTHON_BIN:-python3}"  # override with e.g. PYTHON_BIN=python3.12 if needed
VENV_DIR="${VENV_DIR:-.venv}"
EXTRA_ARGS=()
# ── H200 compatibility flags (from run_h200.sh) ──
FORCE_CHECKPOINT=""            # Custom checkpoint interval
DISABLE_FP8=false              # Skip FP8 entirely (TR_SKIP_FP8=1)
SINGLE_GPU=false               # Force single-GPU
TRT_PRECISION="fp16"           # fp16 | fp8 | int8
TRT_CACHE_DIR=""               # TensorRT engine cache dir
TRT_CALIBRATION=""             # Calibration data file for INT8
TRT_ENABLED=false              # Track TensorRT flag (--tensorrt or --use-tensorrt)
DATA_SHARD=""                  # Data shard index for multi-GPU runs (0-based)
MULTI_GPU=false                # Opt-in: run 1 model copy per GPU with data shards

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full)              MODE="full"; shift ;;
        --quick)             MODE="quick"; shift ;;
        --dry-run)           MODE="dry-run"; shift ;;
        --benchmark-only)    MODE="benchmark-only"; shift ;;
        --translate-only)    MODE="translate-only"; shift ;;
        --warmup-only)       MODE="warmup-only"; shift ;;
        --precompile)        MODE="precompile"; shift ;;
        --model)             MODEL_SIZE="$2"; shift 2 ;;
        --diffusion)         BACKEND="diffusion"; shift ;;
        --tensorrt)          BACKEND="tensorrt"; shift ;;
        --nllb)              BACKEND="encoder_decoder"; shift ;;
        --ar)                BACKEND="autoregressive"; shift ;;
        --nllb-src-lang)     NLLB_SRC_LANG="$2"; shift 2 ;;
        --nllb-tgt-lang)     NLLB_TGT_LANG="$2"; shift 2 ;;
        --speculative)       SPECULATIVE=true; shift ;;
        --spec-mode)         SPEC_MODE="$2"; shift 2 ;;
        --spec-tokens)       SPEC_TOKENS="$2"; shift 2 ;;
        --spec-draft-layers) SPEC_DRAFT_LAYERS="$2"; shift 2 ;;
        --duration)          DURATION="$2"; shift 2 ;;
        --batch-size)        BATCH_SIZE="$2"; shift 2 ;;
        --resume)            RESUME_DIR="$2"; shift 2 ;;
        --shard)             DATA_SHARD="$2"; shift 2 ;;
        --config)            CONFIG_FILE="$2"; shift 2 ;;
        --observability)     OBSERVABILITY=true; shift ;;
        --force-recompile)   FORCE_RECOMPILE=true; shift ;;
        --output)            FORCE_OUTPUT="$2"; shift 2 ;;
        --data)              FORCE_DATA="$2"; shift 2 ;;
        --refs)              FORCE_REFS="$2"; shift 2 ;;
        --seed)              FORCE_SEED="$2"; shift 2 ;;
        --cost)              FORCE_COST="$2"; shift 2 ;;
        --tokens)            FORCE_TOKENS="$2"; shift 2 ;;
        --python)            PYTHON_BIN="$2"; shift 2 ;;
        --venv)              VENV_DIR="$2"; shift 2 ;;
        --no-compile)        EXTRA_ARGS+=("--no-compile"); shift ;;
        --safe-mode)         EXTRA_ARGS+=("--safe-mode"); shift ;;
        --mps-safe)          EXTRA_ARGS+=("--mps-safe"); shift ;;
        # ── H200 compatibility flags ──
        --warmup)            MODE="warmup-only"; shift ;;
        --checkpoint)        FORCE_CHECKPOINT="$2"; shift 2 ;;
        --no-fp8)            DISABLE_FP8=true; shift ;;
        --single-gpu)        SINGLE_GPU=true; shift ;;
        --no-sudo)           info "run.sh does not require sudo — ignoring --no-sudo"; shift ;;
        --use-tensorrt)      BACKEND="tensorrt"; TRT_ENABLED=true; shift ;;
        --multi-gpu)         MULTI_GPU=true; shift ;;
        --trt-precision)     TRT_PRECISION="$2"; shift 2 ;;
        --trt-calibration)   TRT_CALIBRATION="$2"; shift 2 ;;
        --trt-cache)         TRT_CACHE_DIR="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: ./run.sh [OPTIONS]"
            echo ""
            echo "MODES:"
            echo "  (default)      Full 2-hour benchmark"
            echo "  --quick        5-minute evaluation"
            echo "  --dry-run      60-second smoke test"
            echo "  --precompile   Pre-compile all kernels + TRT engines, then exit"
            echo "  --warmup-only  Load + warmup model, then exit"
            echo "  --benchmark-only   Quality evaluation only"
            echo "  --translate-only   Translation only (skip quality)"
            echo ""
            echo "MODEL:"
            echo "  --model 4B|E2B|E4B|E2B-Q4_0|E4B-Q4_0|26B-A4B|ministral-3b   Model size (default: 4B)"
            echo "  --diffusion           Use diffusion model (LLaDA 8B)"
            echo "  --tensorrt            TensorRT acceleration (CUDA only)"
            echo "  --nllb                NLLB encoder-decoder translation model"
            echo "  --nllb-src-lang CODE  NLLB source language (default: eng_Latn)"
            echo "  --nllb-tgt-lang CODE  NLLB target language (default: tur_Latn)"
            echo "  --use-tensorrt        Alias for --tensorrt"
            echo "  --ar                  Force autoregressive (default)"
            echo "  --speculative         Enable speculative decoding (self-speculative)"
            echo "  --spec-mode MODE      'self' or 'draft_model' (default: self)"
            echo "  --spec-tokens N       Speculative tokens K (default: 3)"
            echo "  --spec-draft-layers N Early layers for draft (0=auto)"
            echo ""
            echo "OPTIONS:"
            echo "  --duration N      Run duration in seconds"
            echo "  --batch-size N    Force batch size"
            echo "  --observability   Enable Prometheus dashboard on :9090"
            echo "  --force-recompile Force JIT + TRT recompilation"
            echo "  --config FILE     Use explicit config file"
            echo "  --resume DIR      Resume from checkpoint directory"
            echo "  --output DIR      Output directory"
            echo "  --data GLOB       Input data glob"
            echo "  --refs FILE       Reference set path"
            echo "  --seed N          Random seed"
            echo "  --cost N          GPU cost per hour (USD)"
            echo "  --tokens N        Total token count for extrapolation"
            echo "  --checkpoint N    Checkpoint interval in seconds"
            echo "  --no-compile      Disable torch.compile"
            echo "  --safe-mode       Run with safety sandbox (no eval, restricted paths)"
            echo "  --mps-safe        On Apple Silicon: skip batch tuning & shuffle"
            echo "                    to avoid IOAccelerator bloat (enabled by default)"
            echo "  --warmup          Alias for --warmup-only"
            echo "  --single-gpu      Force single-GPU mode"
            echo "  --no-fp8          Disable FP8 precision (use BF16)"
            echo "  --trt-precision   fp16|fp8|int8  TensorRT precision (default: fp16)"
            echo "  --trt-cache DIR   TensorRT engine cache directory"
            echo "  --trt-calibration FILE  Calibration data for INT8"
            echo "  --no-sudo         No-op (run.sh does not require sudo)"
            echo "  --python PATH     Python binary"
            echo "  --venv DIR        Virtual env directory"
            exit 0
            ;;
        *) fail "Unknown option: $1 (try --help)" ;;
    esac
done

# ═══════════════════════════════════════════════════════════════════════════
# Platform detection + auto-config
# ═══════════════════════════════════════════════════════════════════════════

UNAME_S="$(uname -s)"
UNAME_M="$(uname -m)"

if [ "$UNAME_S" = "Darwin" ] && [ "$UNAME_M" = "arm64" ]; then
    PLATFORM="mps"
    DEFAULT_MODEL_PATH="google/translategemma-4b-it"
    DEFAULT_DTYPE="bfloat16"
    DEFAULT_TP=1
    DEFAULT_DURATION="${DURATION:-3600}"
    DEFAULT_INPUT="data/input/fineweb_en_sample.jsonl.gz"
    DEFAULT_OUTPUT="data/output"
    DEFAULT_REFS="data/references/golden_en_tr.jsonl"
    PLATFORM_LABEL="macOS Apple Silicon (MPS)"
    if [ "$BACKEND" = "tensorrt" ]; then
        warn "TensorRT is CUDA-only — using extreme-optimized AR backend instead"
        BACKEND="autoregressive"
    fi
elif [ "$UNAME_S" = "Linux" ] && command -v nvidia-smi &>/dev/null; then
    PLATFORM="cuda"
    NG=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "GPU")
    DEFAULT_MODEL_PATH="/data/model/translategemma-12b-fp8"
    DEFAULT_DTYPE="auto"
    DEFAULT_TP=0
    DEFAULT_DURATION="${DURATION:-7200}"
    DEFAULT_INPUT="/data/input/clearnet_en_sample_*.jsonl.gz"
    DEFAULT_OUTPUT="/data/output"
    DEFAULT_REFS="/data/references/golden_en_tr.jsonl"
    PLATFORM_LABEL="Linux $NG (CUDA)"

    # cudaMallocAsync is incompatible with torch.compile's cudagraph_trees
    # in PyTorch 2.6 (RuntimeError: cudaMallocAsync does not yet support
    # checkPoolLiveAllocations).  Use the default PyTorch allocator instead.
    # export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-backend:cudaMallocAsync}"
else
    PLATFORM="cpu"
    DEFAULT_MODEL_PATH="google/translategemma-4b-it"
    DEFAULT_DTYPE="float32"
    DEFAULT_TP=0
    DEFAULT_DURATION="${DURATION:-300}"
    DEFAULT_INPUT="data/input/fineweb_en_sample.jsonl.gz"
    DEFAULT_OUTPUT="data/output"
    DEFAULT_REFS="data/references/golden_en_tr.jsonl"
    PLATFORM_LABEL="CPU"
    if [ "$BACKEND" = "tensorrt" ]; then
        warn "TensorRT requires CUDA — using AR backend"
        BACKEND="autoregressive"
    fi
fi

# Override model path for diffusion.
if [ "$BACKEND" = "diffusion" ]; then
    DEFAULT_MODEL_PATH="GSAI-ML/LLaDA-8B-Instruct"
    info "Diffusion mode: $DEFAULT_MODEL_PATH"
fi

# Override model path for NLLB encoder-decoder.
NLLB_SRC_LANG="${NLLB_SRC_LANG:-eng_Latn}"
NLLB_TGT_LANG="${NLLB_TGT_LANG:-tur_Latn}"
if [ "$BACKEND" = "encoder_decoder" ]; then
    DEFAULT_MODEL_PATH="facebook/nllb-200-distilled-600M"
    info "NLLB mode: $DEFAULT_MODEL_PATH (${NLLB_SRC_LANG} → ${NLLB_TGT_LANG})"
fi

# Apply overrides.
MODEL_PATH="${FORCE_MODEL_PATH:-$DEFAULT_MODEL_PATH}"
INPUT="${FORCE_DATA:-$DEFAULT_INPUT}"
OUTPUT="${FORCE_OUTPUT:-$DEFAULT_OUTPUT}"
REFS="${FORCE_REFS:-$DEFAULT_REFS}"
SEED="${FORCE_SEED:-42}"
COST="${FORCE_COST:-}"
TOKENS="${FORCE_TOKENS:-6230000000000}"
CHECKPOINT="${FORCE_CHECKPOINT:-300}"

# Fall back to ./output if the default output dir's parent doesn't exist
# (e.g. /data is not mounted on this machine).
if [ ! -d "$(dirname "$OUTPUT")" ] && [ "$OUTPUT" != "./output" ]; then
    warn "Output parent $(dirname "$OUTPUT") does not exist — using ./output"
    OUTPUT="./output"
fi

# Fall back data paths: if /data doesn't exist, use project-relative paths.
if [ ! -d "$(dirname "$INPUT" 2>/dev/null)" ] && [ "$(dirname "$INPUT")" != "./data/input" ]; then
    INPUT="./data/input/*.jsonl.gz"
fi
if [ ! -f "$REFS" ] && [ "$REFS" != "./data/references/golden_en_tr.jsonl" ]; then
    REFS="./data/references/golden_en_tr.jsonl"
fi

# Multi-GPU data sharding: --shard 0 and --shard 1 each get half the data.
if [ -n "$DATA_SHARD" ]; then
    SHARD_FILE="./data/input/shards/shard_${DATA_SHARD}.jsonl.gz"
    if [ -f "$SHARD_FILE" ]; then
        INPUT="$SHARD_FILE"
        info "Data shard $DATA_SHARD: $INPUT"
    else
        warn "Shard file not found: $SHARD_FILE — using full dataset"
    fi
fi

# Apply model size.
case "$MODEL_SIZE" in
    4B)  MODEL_PATH="google/translategemma-4b-it" ;;
    # v3.4: Gemma 4 QAT models.
    E2B) MODEL_PATH="google/gemma-4-E2B-it-qat-mobile-ct" ;;
    E4B) MODEL_PATH="google/gemma-4-E4B-it-qat-mobile-ct" ;;
    E2B-Q4_0) MODEL_PATH="google/gemma-4-E2B-it-qat-mobile-transformers" ;;
    E4B-Q4_0) MODEL_PATH="google/gemma-4-E4B-it-qat-mobile-transformers" ;;
    # v3.4: DiffusionGemma 26B-A4B.
    26B-A4B|DiffusionGemma) MODEL_PATH="google/diffusiongemma-26B-A4B-it"; BACKEND="diffusion" ;;
    # v3.6: Ministral 3B.
    ministral-3b) MODEL_PATH="mistralai/Ministral-3B-Instruct" ;;
    *)   MODEL_PATH="$MODEL_SIZE" ;;  # literal path
esac

# Fall back model path: if the default doesn't exist locally
# (e.g. /data/model not mounted), use HuggingFace Hub.
if [ ! -d "$MODEL_PATH" ] && [ ! -f "$MODEL_PATH" ] && [ "$MODEL_PATH" != "${MODEL_PATH#/}" ]; then
    case "$MODEL_SIZE" in
        4B)  MODEL_PATH="google/translategemma-4b-it" ;;
        E2B) MODEL_PATH="google/gemma-4-E2B-it-qat-mobile-ct" ;;
        E4B) MODEL_PATH="google/gemma-4-E4B-it-qat-mobile-ct" ;;
        E2B-Q4_0) MODEL_PATH="google/gemma-4-E2B-it-qat-mobile-transformers" ;;
        E4B-Q4_0) MODEL_PATH="google/gemma-4-E4B-it-qat-mobile-transformers" ;;
        26B-A4B|DiffusionGemma) MODEL_PATH="google/diffusiongemma-26B-A4B-it" ;;
        ministral-3b) MODEL_PATH="mistralai/Ministral-3B-Instruct" ;;
        *)   MODEL_PATH="google/translategemma-4b-it" ;;
    esac
    info "Local model not found, using HF Hub: $MODEL_PATH"
fi

# ── H200 compatibility overrides ──
# --no-fp8: force BF16 even on CUDA
if [ "$DISABLE_FP8" = true ]; then
    DEFAULT_DTYPE="bfloat16"
    info "FP8 disabled — using BF16 (--no-fp8)"
fi

# --single-gpu: force single-GPU
if [ "$SINGLE_GPU" = true ]; then
    DEFAULT_TP=1
    info "Single-GPU mode forced (--single-gpu)"
fi

# ── Default batch size ──────────────────────────────────────────────────
# CUDA/H200: default 512 (safe for 4B-12B models on 140 GB GPU).
# MPS: auto-tune (unified memory is tight).
if [ -z "$BATCH_SIZE" ] || [ "$BATCH_SIZE" = "0" ]; then
    if [ "$PLATFORM" = "cuda" ]; then
        BATCH_SIZE=512
    fi
fi

# ── Multi-GPU data-parallel mode (STRICTLY OPT-IN via --multi-gpu) ────
# Only activates when the user explicitly passes --multi-gpu, there are
# >=2 GPUs, and the model fits on a single GPU (4B or 12B).
# Protected against fork-bomb recursion by TR_DEPTH (top of file) and
# TR_SHARD (set per child).
_NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
_AUTO_SHARD=false
if [ "$MULTI_GPU" = true ] && [ "$PLATFORM" = "cuda" ] && [ "$_NUM_GPUS" -ge 2 ]; then
    if [ "$SINGLE_GPU" = true ]; then
        warn "--multi-gpu and --single-gpu conflict — ignoring --multi-gpu"
    else
        case "$MODEL_SIZE" in
            4B|12B) _AUTO_SHARD=true ;;
            *) warn "--multi-gpu only supports 4B/12B (model fits on one GPU) — running single-GPU" ;;
        esac
    fi
fi

if [ "$_AUTO_SHARD" = true ]; then
    # Pre-split data if shards don't already exist.
    _SHARD_DIR="./data/input/shards"
    _SHARD_0="$_SHARD_DIR/shard_0.jsonl.gz"
    _SHARD_1="$_SHARD_DIR/shard_1.jsonl.gz"
    if [ ! -f "$_SHARD_0" ] || [ ! -f "$_SHARD_1" ]; then
        warn "Splitting data into $_NUM_GPUS shards for multi-GPU parallelism..."
        mkdir -p "$_SHARD_DIR"
        # SECURITY: INPUT is passed via argv to avoid shell injection from --data flag.
        # Never interpolate user-controlled strings into python -c blocks.
        _SPLIT_SCRIPT=$(mktemp /tmp/tr_data_split.XXXXXX.py)
        cat > "$_SPLIT_SCRIPT" << 'PYSPLIT'
import json, gzip, glob, os, sys

input_pattern = sys.argv[1]
num_gpus = int(sys.argv[2])
shard_dir = sys.argv[3]

# Find the input file (resolve glob)
patterns = [input_pattern] if not input_pattern.startswith('./data/input/shards') else ['data/input/fineweb_en_sample.jsonl.gz', 'data/input/*.jsonl.gz']
docs = []
for pat in patterns:
    for f in sorted(glob.glob(pat)):
        opener = gzip.open if f.endswith('.gz') else open
        with opener(f, 'rt', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try: docs.append(json.loads(line))
                    except: continue
        if docs:
            break

n = len(docs)
if n < num_gpus * 2:
    print(f'Not enough docs ({n}) to split across {num_gpus} GPUs — using full dataset per GPU')
    sys.exit(0)

chunk = n // num_gpus
for gpu in range(num_gpus):
    start = gpu * chunk
    end = start + chunk if gpu < num_gpus - 1 else n
    subset = docs[start:end]
    path = os.path.join(shard_dir, f'shard_{gpu}.jsonl.gz')
    with gzip.open(path, 'wt', encoding='utf-8') as fh:
        for doc in subset:
            fh.write(json.dumps(doc, ensure_ascii=False) + '\n')
    print(f'shard_{gpu}: {len(subset)} docs ({os.path.getsize(path)/1024**2:.1f} MB)')
PYSPLIT
        $PYTHON_BIN "$_SPLIT_SCRIPT" "$INPUT" "$_NUM_GPUS" "$_SHARD_DIR" 2>&1 || warn "Data split failed — falling back to single-GPU mode"
        rm -f "$_SPLIT_SCRIPT"
    fi

    # Check shards actually exist after split attempt.
    if [ -f "$_SHARD_0" ] && [ -f "$_SHARD_1" ]; then
        ok "Multi-GPU data-parallel mode: $_NUM_GPUS GPUs, $_NUM_GPUS model copies"
        info "Each GPU processes a disjoint data shard — 0% wasted work"

        # Reconstruct original CLI args, replacing --output if present.
        _ORIG_ARGS=()
        _HAS_OUTPUT=false
        for _a in "$@"; do
            if [ "$_HAS_OUTPUT" = true ]; then
                _HAS_OUTPUT=false
                continue
            fi
            case "$_a" in
                --output) _HAS_OUTPUT=true ;;
                --multi-gpu) ;;  # Don't forward to children — they'd recurse
                *) _ORIG_ARGS+=("$_a") ;;
            esac
        done

        _PIDS=()
        _OUTPUTS=()
        for _gpu in $(seq 0 $((_NUM_GPUS - 1))); do
            _OUT="./output/gpu$_gpu"
            CUDA_VISIBLE_DEVICES="$_gpu" TR_SHARD="$_gpu" "$0" --shard "$_gpu" --output "$_OUT" "${_ORIG_ARGS[@]}" &
            _PIDS+=($!)
            _OUTPUTS+=("$_OUT")
        done

        _FAILED=0
        for _i in "${!_PIDS[@]}"; do
            wait "${_PIDS[$_i]}" || _FAILED=$((_FAILED + 1))
        done

        echo ""
        if [ "$_FAILED" -eq 0 ]; then
            ok "All $_NUM_GPUS GPU processes completed successfully"
            for _out in "${_OUTPUTS[@]}"; do
                _latest=$(ls -td "$_out"/20* 2>/dev/null | head -1)
                if [ -n "$_latest" ] && [ -f "$_latest/report/benchmark_report.json" ]; then
                    echo -e "  ${C}── $(basename "$_out") ──${N}"
                    $PYTHON_BIN benchmark/utils/print_summary.py "$_latest/report/benchmark_report.json" 2>/dev/null || true
                fi
            done
        else
            warn "$_FAILED of $_NUM_GPUS GPU processes failed — check logs in output/gpu*/"
        fi
        exit "$_FAILED"
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════
# Pre-compile mode
# ═══════════════════════════════════════════════════════════════════════════
if [ "$MODE" = "precompile" ]; then
    banner "Pre-compiling JIT Kernels + TRT Engines"
    source "${VENV_DIR}/bin/activate" 2>/dev/null || true

    if [ "$FORCE_RECOMPILE" = true ]; then
        export TR_BENCHMARK_FORCE_RECOMPILE=1
        info "Force recompilation enabled"
    fi

    $PYTHON_BIN -c "
from benchmark.hardware.jit_compiler import precompile_all_kernels, get_jit_compiler
from benchmark.hardware.trt_builder import TRTEngineBuilder
import torch

n_jit = precompile_all_kernels()
print(f'JIT Kernels compiled: {n_jit}')

compiler = get_jit_compiler()
print(f'Cache: {compiler.cache_stats()}')
" 2>&1 || warn "JIT pre-compilation had warnings (expected if nvcc not available)"

    if [ "$PLATFORM" = "cuda" ] && $PYTHON_BIN -c "import tensorrt" 2>/dev/null; then
        info "TensorRT engine pre-build requires a model download on first run."
        info "The engine will be built automatically during the first benchmark."
    fi

    banner "Pre-compilation Complete"
    echo "  Run ./run.sh --full to start benchmarking."
    exit 0
fi

# ═══════════════════════════════════════════════════════════════════════════
# Write runtime config
# ═══════════════════════════════════════════════════════════════════════════

if [ -z "$CONFIG_FILE" ]; then
    CONFIG_FILE="$OUTPUT/runtime_config_$(date +%Y%m%d_%H%M%S).yaml"
    mkdir -p "$OUTPUT"

    TRT_BOOL="false"
    if [ "$BACKEND" = "tensorrt" ]; then TRT_BOOL="true"; fi

    DIFF_STEPS=256
    DIFF_GUIDANCE=1.0
    if [ "$BACKEND" = "diffusion" ]; then
        DIFF_STEPS="${DIFF_STEPS:-256}"
        DIFF_GUIDANCE="${DIFF_GUIDANCE:-1.0}"
    fi

    cat > "$CONFIG_FILE" << YAML
# Auto-generated by run.sh — $(date)
backend: "auto"

model:
  model_path: "$MODEL_PATH"
  tokenizer_path: ""
  max_input_tokens: 512
  max_new_tokens: 512
  temperature: 0.0
  do_sample: false
  num_beams: 1
  dtype: "$DEFAULT_DTYPE"
  tensor_parallel_size: $DEFAULT_TP
  use_flash_attention: true
  backend_type: "$BACKEND"
  plugin_name: ""
  plugin_config: {}
  use_tensorrt: $TRT_BOOL
  tensorrt_precision: "${TRT_PRECISION:-fp16}"
  tensorrt_max_batch: 32
  tensorrt_cache_dir: "${TRT_CACHE_DIR:-}"
  tensorrt_calibration_file: "${TRT_CALIBRATION:-}"
  # v3.4: Speculative decoding
  use_speculative: ${SPECULATIVE,,}   # bash lower-casing: true/false
  speculative_mode: "$SPEC_MODE"
  speculative_num_tokens: $SPEC_TOKENS
  speculative_draft_model: ""
  speculative_num_draft_layers: $SPEC_DRAFT_LAYERS
  # v3.6: NLLB encoder-decoder
  nllb_source_lang: "$NLLB_SRC_LANG"
  nllb_target_lang: "$NLLB_TGT_LANG"
  diffusion_steps: $DIFF_STEPS
  guidance_scale: $DIFF_GUIDANCE
  noise_schedule: "cosine"
  target_length_multiplier: 2.0

runtime:
  target_duration_seconds: $DEFAULT_DURATION
  checkpoint_interval_seconds: $CHECKPOINT
  heartbeat_interval_seconds: 10
  metrics_sample_rate_hz: 1
  seed: $SEED

data:
  input_paths:
    - "$INPUT"
  output_dir: "$OUTPUT"
  reference_set_path: "$REFS"
  shard_size_mb: 100
  prefetch_workers: 4
  shuffle: true
  min_chunk_tokens: 10
  max_garbage_ratio: 0.95
  chunk_overlap_tokens: 50

extrapolation:
  total_clearnet_non_tr_tokens: $TOKENS
  gpu_cost_per_hour_usd: $( [ -n "$COST" ] && echo "$COST" || echo "null" )
  # ^ null means "no cost estimation" — cost projection is skipped in the report.
  #   Pass --cost <dollars_per_hour> to enable extrapolated cost estimates.
YAML
    ok "Config written: $CONFIG_FILE"
else
    ok "Using config: $CONFIG_FILE"
fi

# ═══════════════════════════════════════════════════════════════════════════
# Build CLI args
# ═══════════════════════════════════════════════════════════════════════════
CLI_ARGS=("--config" "$CONFIG_FILE")

# Mode.
case "$MODE" in
    full)            RUN_LABEL="Full ($DEFAULT_DURATION s)" ;;
    quick)           CLI_ARGS+=("--quick"); RUN_LABEL="Quick (300 s)" ;;
    dry-run)         CLI_ARGS+=("--dry-run"); RUN_LABEL="Dry-run (60 s)" ;;
    benchmark-only)  CLI_ARGS+=("--benchmark-only"); RUN_LABEL="Quality only" ;;
    translate-only)  CLI_ARGS+=("--translate-only"); RUN_LABEL="Translation only" ;;
    warmup-only)     CLI_ARGS+=("--warmup-only"); RUN_LABEL="Warmup only" ;;
esac

# Overrides.
[ -n "$RESUME_DIR" ] && CLI_ARGS+=("--resume" "$RESUME_DIR")
[ -n "$BATCH_SIZE" ] && [ "$BATCH_SIZE" != "0" ] && CLI_ARGS+=("--batch-size" "$BATCH_SIZE")
[ -n "$DURATION" ] && CLI_ARGS+=("--duration" "$DURATION")

# Observability flag — pass through to the benchmark process.
[ "$OBSERVABILITY" = true ] && CLI_ARGS+=("--observability")

# Speculative decoding flags.
[ "$SPECULATIVE" = true ] && CLI_ARGS+=("--speculative" "--spec-mode" "$SPEC_MODE" "--spec-tokens" "$SPEC_TOKENS" "--spec-draft-layers" "$SPEC_DRAFT_LAYERS")

# Only expand EXTRA_ARGS if it has elements (avoids empty arg with set -u).
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    CLI_ARGS+=("${EXTRA_ARGS[@]}")
fi

# ═══════════════════════════════════════════════════════════════════════════
# Pre-flight summary
# ═══════════════════════════════════════════════════════════════════════════
banner "TR Benchmark v3.6 — $PLATFORM_LABEL"
echo -e "  ${B}Mode:${N}      $RUN_LABEL"
# Derive a human-readable model label from MODEL_PATH for the banner.
_model_label="$MODEL_PATH"
case "$MODEL_PATH" in
    *4b*|*4B*) _model_label="TranslateGemma 4B" ;;
    *12b*|*12B*) _model_label="TranslateGemma 12B" ;;
    *27b*|*27B*) _model_label="TranslateGemma 27B" ;;
    *translategemma-4b*) _model_label="TranslateGemma 4B" ;;
    *translategemma-12b*) _model_label="TranslateGemma 12B" ;;
    *translategemma-27b*) _model_label="TranslateGemma 27B" ;;
    *SmolLM*|*smollm*) _model_label="SmolLM2 1.7B" ;;
    *LLaDA*|*llada*) _model_label="LLaDA 8B" ;;
    # v3.4: Gemma 4 QAT models.
    *gemma-4-E2B*qat*|*gemma-4-e2b*qat*) _model_label="Gemma 4 E2B QAT" ;;
    *gemma-4-E4B*qat*|*gemma-4-e4b*qat*) _model_label="Gemma 4 E4B QAT" ;;
    *gemma-4*2b*|*gemma-4*2B*) _model_label="Gemma 4 E2B" ;;
    *gemma-4*4b*|*gemma-4*4B*) _model_label="Gemma 4 E4B" ;;
    # v3.4: DiffusionGemma.
    *diffusiongemma*|*DiffusionGemma*) _model_label="DiffusionGemma 26B-A4B" ;;
esac

echo -e "  ${B}Model:${N}     ${_model_label}"
echo -e "  ${B}Backend:${N}   $BACKEND"
echo -e "  ${B}Platform:${N}  $PLATFORM ($(uname -m))"
echo -e "  ${B}Precision:${N} BF16 + TF32 + SDPA (FP8 not active on pip venvs; see docs/FP8_TE_CUDA_ISSUES.md)"
echo -e "  ${B}Config:${N}    $CONFIG_FILE"
echo ""

# Quick system check.
$PYTHON_BIN -c "
import torch, sys, psutil
gpu = 'none'
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'  GPU:    {gpu} ({mem:.0f} GB) × {torch.cuda.device_count()}')
elif torch.backends.mps.is_available():
    print(f'  GPU:    Apple Silicon (MPS) — {psutil.virtual_memory().total/(1024**3):.0f} GB unified')
else:
    print(f'  GPU:    none (CPU) — {psutil.cpu_count()} cores')
print(f'  PyTorch: {torch.__version__}')
" 2>/dev/null || true

if [ "$FORCE_RECOMPILE" = true ]; then
    export TR_BENCHMARK_FORCE_RECOMPILE=1
    warn "Force recompilation enabled — engines will be rebuilt"
fi

# ═══════════════════════════════════════════════════════════════════════════
# Activate venv + launch
# ═══════════════════════════════════════════════════════════════════════════
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
    # Validate venv activation: verify $VIRTUAL_ENV is set and matches VENV_DIR.
    _expected_venv="$(cd "$SCRIPT_DIR" && pwd)/${VENV_DIR}"
    if [ "${VIRTUAL_ENV:-}" != "$_expected_venv" ]; then
        fail "Virtualenv activation failed: VIRTUAL_ENV=$VIRTUAL_ENV expected $_expected_venv"
    fi
    if ! python -c "import sys; sys.exit(0 if sys.prefix == '$_expected_venv' else 1)" 2>/dev/null; then
        fail "Virtualenv activation mismatch: python prefix does not point to $_expected_venv"
    fi
fi

if [ "$OBSERVABILITY" = true ]; then
    info "Starting Prometheus metrics exporter (background)..."
    $PYTHON_BIN -c "
from benchmark.observability.server import start_dashboard_server
srv = start_dashboard_server(port=9090)
print('Dashboard: http://localhost:9090/')
print('Metrics:   http://localhost:9090/metrics')
import time; time.sleep(2)
" &
    OBS_PID=$!
    # Clean up on SIGINT/SIGTERM as well — not just EXIT.
    # Without this, Ctrl-C leaves the Prometheus exporter orphaned.
    _cleanup_obs() { kill "$OBS_PID" 2>/dev/null || true; }
    trap _cleanup_obs EXIT SIGINT SIGTERM
    ok "Observability: http://localhost:9090/"
fi

info "Launching benchmark..."
echo ""

# Suppress torch._dynamo recompile spam + pynvml deprecation.
export TORCH_LOGS="-dynamo"
export PYTHONWARNINGS="ignore::FutureWarning"

$PYTHON_BIN -m benchmark "${CLI_ARGS[@]}"
EXIT_CODE=$?

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
echo ""
banner "Benchmark Complete"
if [ "$EXIT_CODE" -eq 0 ]; then
    echo -e "  ${G}✓${N} Exit code: 0"
    if [ -d "$OUTPUT" ]; then
        LATEST=$(ls -td "$OUTPUT"/20* 2>/dev/null | head -1)
        if [ -n "$LATEST" ]; then
            echo -e "  ${G}✓${N} Results: $LATEST"
            if [ -f "$LATEST/report/benchmark_report.json" ]; then
                $PYTHON_BIN benchmark/utils/print_summary.py "$LATEST/report/benchmark_report.json" 2>/dev/null || true
            fi
        fi
    fi
else
    echo -e "  ${R}✗${N} Exit code: $EXIT_CODE"
    if [ -d "$OUTPUT" ]; then
        LATEST=$(ls -td "$OUTPUT"/20* 2>/dev/null | head -1)
        [ -n "$LATEST" ] && echo "  Logs: $LATEST/benchmark.log"
    fi
fi
echo ""

exit $EXIT_CODE

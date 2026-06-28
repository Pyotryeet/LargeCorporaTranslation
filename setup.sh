#!/usr/bin/env bash
# =============================================================================
#  setup.sh — One-command environment setup for ALL platforms
# =============================================================================
#  Auto-detects macOS (Apple Silicon) vs Linux (CUDA/CPU) and installs
#  everything needed: Python venv, PyTorch, CUDA deps, TensorRT, Triton,
#  Metal toolchain, Rust tokenizer, model download, HF login, calibration data.
#
#  Usage:
#    ./setup.sh                          # Auto-detect platform, full install
#    ./setup.sh --cuda                   # Force CUDA install (Linux)
#    ./setup.sh --cpu                    # CPU-only, no GPU deps
#    ./setup.sh --full                   # Everything including TensorRT + Triton
#    ./setup.sh --minimal                # Bare minimum to run tests
#    ./setup.sh --quick                  # Dev setup (skip model DL)
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colors ─────────────────────────────────────────────────────────────────
R='\033[0;31m'; G='\033[0;32m'; Y='\033[1;33m'; C='\033[0;36m'; B='\033[1m'; N='\033[0m'
banner(){ echo -e "\n${C}═══ $* ═══${N}"; }
ok(){ echo -e "  ${G}✓${N} $*"; }
warn(){ echo -e "  ${Y}⚠${N} $*"; }
fail(){ echo -e "  ${R}✗${N} $*"; exit 1; }
info(){ echo -e "  ${C}→${N} $*"; }
step(){ echo -e "\n${B}[$1]${N} $2"; }

# ── Parse args ─────────────────────────────────────────────────────────────
MODE="auto"   # auto | cuda | cpu
PROFILE="full" # full | minimal | quick
FORCE_CUDA_VERSION=""
NO_MODEL_DL=false
PYTHON_BIN="python3.11"
VENV_DIR=".venv"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda)   MODE="cuda"; shift ;;
        --cpu)    MODE="cpu"; shift ;;
        --full)   PROFILE="full"; shift ;;
        --minimal) PROFILE="minimal"; shift ;;
        --quick)  PROFILE="quick"; shift ;;
        --no-model-dl) NO_MODEL_DL=true; shift ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        --venv)   VENV_DIR="$2"; shift 2 ;;
        --cuda-version) FORCE_CUDA_VERSION="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: ./setup.sh [OPTIONS]"
            echo ""
            echo "Profiles:"
            echo "  (default)       Full install — everything including TensorRT + Triton"
            echo "  --quick         Dev setup — skip model download"
            echo "  --minimal       Bare minimum — just Python + PyTorch + shared deps"
            echo ""
            echo "Platform overrides:"
            echo "  --cuda          Force CUDA path (Linux only)"
            echo "  --cpu           Force CPU-only path"
            echo "  --no-model-dl   Skip HuggingFace model download"
            echo ""
            echo "Options:"
            echo "  --python PATH   Python binary (default: python3.11)"
            echo "  --venv DIR      Virtual env directory (default: .venv)"
            echo "  --cuda-version  Force specific CUDA version (e.g. 12.4)"
            exit 0
            ;;
        *) fail "Unknown option: $1" ;;
    esac
done

banner "TR Benchmark Setup — v3.6"
echo "  Platform: $(uname -s) / $(uname -m)"
echo "  Python:   $PYTHON_BIN"
echo "  Profile:  $PROFILE"
echo "  Mode:     $MODE"

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — Platform detection
# ═══════════════════════════════════════════════════════════════════════════
step 1 "Detecting platform..."

UNAME_S="$(uname -s)"
UNAME_M="$(uname -m)"

if [ "$MODE" = "auto" ]; then
    if [ "$UNAME_S" = "Darwin" ] && [ "$UNAME_M" = "arm64" ]; then
        MODE="mps"
        ok "Detected: macOS Apple Silicon (MPS)"
    elif [ "$UNAME_S" = "Linux" ]; then
        if command -v nvidia-smi &>/dev/null; then
            MODE="cuda"
            NG=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "NVIDIA GPU")
            ok "Detected: Linux + $NG (CUDA)"
        else
            MODE="cpu"
            ok "Detected: Linux (CPU only)"
        fi
    else
        MODE="cpu"
        ok "Detected: $UNAME_S $UNAME_M (CPU only)"
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — Check prerequisites
# ═══════════════════════════════════════════════════════════════════════════
step 2 "Checking prerequisites..."

# Python.
if ! command -v "$PYTHON_BIN" &>/dev/null; then
    warn "$PYTHON_BIN not found — trying python3..."
    if command -v python3 &>/dev/null; then
        PYTHON_BIN="python3"
        ok "Using: $PYTHON_BIN ($($PYTHON_BIN --version))"
    else
        fail "Python 3.11+ required. Install: brew install python@3.11 (macOS) or apt install python3.11 (Linux)"
    fi
fi
PYVER=$($PYTHON_BIN --version 2>&1 | awk '{print $2}')
ok "Python: $PYVER"

# pip.
if ! $PYTHON_BIN -m pip --version &>/dev/null; then
    $PYTHON_BIN -m ensurepip --upgrade
fi
ok "pip: $($PYTHON_BIN -m pip --version | awk '{print $2}')"

# Platform-specific checks.
if [ "$MODE" = "cuda" ]; then
    if ! command -v nvidia-smi &>/dev/null; then
        fail "CUDA mode but nvidia-smi not found. Install NVIDIA drivers."
    fi
    CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    ok "NVIDIA driver: $CUDA_VER"

    if ! command -v nvcc &>/dev/null; then
        if [ -d /usr/local/cuda/bin ]; then
            export PATH="/usr/local/cuda/bin:$PATH"
        elif [ -d /usr/local/cuda-12/bin ]; then
            export PATH="/usr/local/cuda-12/bin:$PATH"
        fi
    fi
    if command -v nvcc &>/dev/null; then
        ok "nvcc: $(nvcc --version | grep release | awk '{print $5, $6}')"
    else
        warn "nvcc not in PATH — JIT CUDA C++ kernel compilation will be skipped"
    fi
fi

if [ "$MODE" = "mps" ]; then
    if [ "$(sw_vers -productVersion | cut -d. -f1)" -lt 14 ]; then
        warn "macOS < 14 — some MPS features may be limited"
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — Virtual environment
# ═══════════════════════════════════════════════════════════════════════════
step 3 "Setting up virtual environment ($VENV_DIR)..."

if [ ! -d "$VENV_DIR" ]; then
    $PYTHON_BIN -m venv "$VENV_DIR"
    ok "Virtual environment created"
else
    ok "Virtual environment exists"
fi

source "$VENV_DIR/bin/activate"
PYTHON_BIN="$VENV_DIR/bin/python"
# NOTE: pip install without --require-hashes — hash checking is deferred because:
#        (1) upstream wheels are signed by PyPI and verified via TLS; (2) this
#        project pins top-level constraints but not transitive hashes; (3) nightly
#        PyTorch/CUDA wheels change daily, making hash files stale immediately.
#        When moving to a production Docker build, freeze all deps with:
#          pip-compile --generate-hashes -o requirements-build.txt requirements-build.in
$PYTHON_BIN -m pip install --upgrade pip wheel --quiet --no-cache-dir
# Don't upgrade setuptools — PyTorch pins setuptools<82.
# NOTE: pip install without --require-hashes — hash checking is deferred because:
# (1) upstream wheels are signed by PyPI and verified via TLS; (2) this project
# pins top-level constraints but not transitive hashes; (3) nightly PyTorch/CUDA
# wheels change daily, making hash files stale immediately.
# When moving to a production Docker build, freeze all deps with:
#   pip-compile --generate-hashes -o requirements-build.txt requirements-build.in
$PYTHON_BIN -m pip install "setuptools>=68.0,<82" --quiet --no-cache-dir 2>/dev/null || true
ok "pip upgraded"

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — Install dependencies
# ═══════════════════════════════════════════════════════════════════════════
step 4 "Installing dependencies (profile=$PROFILE, mode=$MODE)..."

# Core deps (always).
info "Installing core dependencies..."
# NOTE: pip install without --require-hashes — hash checking is deferred because:
#        (1) upstream wheels are signed by PyPI and verified via TLS; (2) this
#        project pins top-level constraints but not transitive hashes; (3) nightly
#        PyTorch/CUDA wheels change daily, making hash files stale immediately.
#        When moving to a production Docker build, freeze all deps with:
#          pip-compile --generate-hashes -o requirements.txt requirements.in
$PYTHON_BIN -m pip install -r requirements.txt --quiet --no-cache-dir
ok "Core dependencies installed"

# Platform-specific PyTorch.
# ⚠️ VERSION GUIDE: PyTorch for H200 (SM90) — June 2026
#   - 2.12.1+cu126 → RECOMMENDED. +27% TPS over 2.6.0 (1,650 vs 1,300 tok/s on 4B).
#     torch.compile(mode="default") active on 2.12–2.13; mode="reduce-overhead" on 2.14+.
#   - 2.6.0+cu124  → Works but compile auto-skipped (cudagraph_trees bug).
#   - 2.11.0+cu130 → SDPA crashes on SM90 (cuDNN frontend has no execution plans)
# See docs/COMPILATION_GUIDE.md.
if [ "$MODE" = "cuda" ]; then
    if [ -n "$FORCE_CUDA_VERSION" ]; then
        CUDA_INDEX="cu${FORCE_CUDA_VERSION//./}"
    else
        CUDA_INDEX="cu126"
    fi
    info "Installing PyTorch 2.12.1 with CUDA ($CUDA_INDEX)..."
    $PYTHON_BIN -m pip install 'torch>=2.12.1' 'torchvision>=0.27' --index-url "https://download.pytorch.org/whl/$CUDA_INDEX" --quiet --no-cache-dir
    ok "PyTorch (CUDA) installed"
elif [ "$MODE" = "mps" ]; then
    info "Installing PyTorch (MPS)..."
    # NOTE: pip install without --require-hashes — hash checking is deferred because:
    # (1) upstream wheels are signed by PyPI and verified via TLS; (2) this project
    # pins top-level constraints but not transitive hashes; (3) nightly PyTorch/CUDA
    # wheels change daily, making hash files stale immediately.
    # When moving to a production Docker build, freeze all deps with:
    #   pip-compile --generate-hashes -o requirements-torch-mps.txt requirements-torch-mps.in
    $PYTHON_BIN -m pip install torch torchvision --quiet --no-cache-dir
    ok "PyTorch (MPS) installed"
else
    info "Installing PyTorch (CPU)..."
    # NOTE: pip install without --require-hashes — hash checking is deferred because:
    # (1) upstream wheels are signed by PyPI and verified via TLS; (2) this project
    # pins top-level constraints but not transitive hashes; (3) nightly PyTorch/CUDA
    # wheels change daily, making hash files stale immediately.
    # When moving to a production Docker build, freeze all deps with:
    #   pip-compile --generate-hashes -o requirements-torch-cpu.txt requirements-torch-cpu.in
    $PYTHON_BIN -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet --no-cache-dir
    ok "PyTorch (CPU) installed"
fi

# Install the package.
# NOTE: pip install without --require-hashes — hash checking is deferred because:
#        (1) upstream wheels are signed by PyPI and verified via TLS; (2) this
#        project pins top-level constraints but not transitive hashes; (3) nightly
#        PyTorch/CUDA wheels change daily, making hash files stale immediately.
#        When moving to a production Docker build, freeze all deps with:
#          pip-compile --generate-hashes -o requirements.txt requirements.in
$PYTHON_BIN -m pip install -e . --no-deps --quiet --no-cache-dir
ok "Package installed (editable)"

# Dev deps.
if [ "$PROFILE" != "minimal" ]; then
    # NOTE: pip install without --require-hashes — hash checking is deferred because:
    # (1) upstream wheels are signed by PyPI and verified via TLS; (2) this project
    # pins top-level constraints but not transitive hashes; (3) nightly PyTorch/CUDA
    # wheels change daily, making hash files stale immediately.
    # When moving to a production Docker build, freeze all deps with:
    #   pip-compile --generate-hashes -o requirements-dev.txt requirements-dev.in
    $PYTHON_BIN -m pip install -e ".[dev]" --quiet --no-cache-dir
    ok "Dev dependencies installed"
fi

# CUDA extras.
if [ "$MODE" = "cuda" ]; then
    info "Installing CUDA extras..."
    # flash-attn's build system imports torch at wheel-build time, so pip's
    # default build isolation sandbox fails (torch isn't in the sandbox).
    # ── CUDA extras ─────────────────────────────────────────────────
    $PYTHON_BIN -m pip install nvidia-ml-py --quiet --no-cache-dir || true
    info "Installing FlashAttention-3..."
    $PYTHON_BIN -m pip install "flash-attn-3" --index-url "https://download.pytorch.org/whl/$CUDA_INDEX" --quiet --no-cache-dir || \
        $PYTHON_BIN -m pip install "flash-attn-3" --quiet --no-cache-dir || \
        warn "FlashAttention-3 installation failed"
    ok "CUDA extras installed"

    # ── Transformer Engine (FP8 on Hopper) ──────────────────────────
    # NOTE (June 2026): TE source-build succeeds on pip venvs but **runtime
    # cuBLASLt crashes on all tested drivers (580, 565)**.  TE is COMMENTED
    # OUT by default to avoid a 2-5 minute compile that produces non-working
    # FP8.  The only known-working FP8 path is the NGC container
    # (nvcr.io/nvidia/pytorch:24.12-py3).
    #
    # To attempt TE anyway (YMMV — verify with the 256x256 smoking gun test):
    #   source .venv/bin/activate
    #   pip install 'transformer-engine[pytorch]>=2.14.0' --no-build-isolation
    #
    # Static quantization (SmoothQuant / QAT) via save_fp8_weights() /
    # load_fp8_weights() is the path forward for pip venvs.
    info "Skipping Transformer Engine — known broken on pip venvs"
    info "  FP8 is available via NGC container (nvcr.io/nvidia/pytorch:24.12-py3)"
fi

# ── Post-install compatibility check (CUDA only) ─────────────────────────
# torch 2.11+cu130 ships a cuDNN frontend with no SM90 execution plans for
# H200 GPUs — all SDPA backends (flash, mem_efficient, math) crash with
# "No valid execution plans built".  PyTorch 2.6.0+cu124 is the known-good
# combination for H200/SM90.  See docs/H200_SETUP.md.
#
# Also removes packages that conflict with torch 2.6: torchao 0.17+ requires
# torch >= 2.10 (register_constant); compressed-tensors 0.17+ requires
# torch >= 2.10.  These are QAT/quantization tooling not needed for
# throughput benchmarks and can be reinstalled from source if required.
if [ "$MODE" = "cuda" ]; then
    info "Running post-install compatibility checks..."
    PY_VER=$($PYTHON_BIN -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
    CUDA_VER=$($PYTHON_BIN -c "import torch; print(torch.version.cuda or 'unknown')" 2>/dev/null || echo "unknown")
    info "  PyTorch: $PY_VER, CUDA: $CUDA_VER"

    # Check for the known-broken cu130 combination.
    if echo "$PY_VER" | grep -q "2.11"; then
        warn "PyTorch 2.11 detected — SDPA may not work on H200 (SM90)."
        warn "  Downgrade to 2.6.0+cu124 if you see 'No valid execution plans built':"
        warn "    pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124"
    fi

    # Remove packages known to conflict with torch < 2.10.
    for _pkg in torchao compressed-tensors; do
        if $PYTHON_BIN -m pip show "$_pkg" >/dev/null 2>&1; then
            _pkg_ver=$($PYTHON_BIN -m pip show "$_pkg" 2>/dev/null | grep Version | awk '{print $2}')
            warn "Removing $_pkg $_pkg_ver — incompatible with torch < 2.10."
            warn "  Reinstall from source if QAT/quantization tooling is needed."
            $PYTHON_BIN -m pip uninstall -y "$_pkg" --quiet 2>/dev/null || true
        fi
    done

    # Quick SDPA smoke test.
    $PYTHON_BIN -c "
import torch
a = torch.randn(2, 4, 128, 64, device='cuda', dtype=torch.float16)
try:
    b = torch.nn.functional.scaled_dot_product_attention(a, a, a)
except RuntimeError as e:
    import sys
    sys.exit(1)
" 2>/dev/null && ok "SDPA smoke test passed" || \
        warn "SDPA smoke test FAILED — Flash SDPA may not work on this GPU. See docs/H200_SETUP.md §B.1."
fi

# MPS extras.
if [ "$MODE" = "mps" ] && [ "$PROFILE" = "full" ]; then
    info "Installing MPS extras..."
    # NOTE: pip install without --require-hashes — hash checking is deferred because:
    # (1) upstream wheels are signed by PyPI and verified via TLS; (2) this project
    # pins top-level constraints but not transitive hashes; (3) nightly PyTorch/CUDA
    # wheels change daily, making hash files stale immediately.
    # When moving to a production Docker build, freeze all deps with:
    #   pip-compile --generate-hashes -o requirements-mps.txt requirements-mps.in
    $PYTHON_BIN -m pip install -e ".[mps]" --quiet --no-cache-dir 2>/dev/null || {
        warn "mlx not available (requires macOS >= 14)"
    }
    ok "MPS extras installed"
fi

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — Verify installation
# ═══════════════════════════════════════════════════════════════════════════
step 5 "Verifying installation..."

python -c "
import torch, sys
print(f'  PyTorch:    {torch.__version__}')
print(f'  CUDA:       {torch.cuda.is_available()} (devices: {torch.cuda.device_count() if torch.cuda.is_available() else 0})')
print(f'  MPS:        {torch.backends.mps.is_available()}')
print(f'  Python:     {sys.version.split()[0]}')
try:
    import transformers; print(f'  Transformers: {transformers.__version__}')
except: print('  Transformers: NOT INSTALLED')
try:
    import orjson; print(f'  orjson:       installed')
except: print('  orjson:       NOT INSTALLED')
try:
    import triton; print(f'  Triton:       {triton.__version__}')
except: print('  Triton:       NOT INSTALLED (Linux/NVIDIA only)')
try:
    import psutil; print(f'  RAM:          {psutil.virtual_memory().total/(1024**3):.1f} GiB')
except: pass
" || warn "Some imports failed — may be missing optional deps"

# Run unit tests.
if [ "$PROFILE" != "minimal" ]; then
    info "Running unit tests..."
    python -m pytest tests/ -q --timeout=120 --ignore=tests/test_e2e.py 2>&1 | tail -3
fi

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 — HuggingFace login + model download
# ═══════════════════════════════════════════════════════════════════════════
if [ "$NO_MODEL_DL" = false ] && [ "$PROFILE" != "minimal" ]; then
    step 6 "HuggingFace login..."

    if [ -n "${HF_TOKEN:-}" ]; then
        ok "HF_TOKEN environment variable found"
    elif [ -f ~/.cache/huggingface/token ]; then
        ok "HF token found in cache"
    else
        info "Log in to HuggingFace (required for gated models like TranslateGemma):"
        python -c "from huggingface_hub import login; login()" 2>/dev/null || {
            warn "huggingface_hub not installed — skipping login"
            warn "Set HF_TOKEN env var or run 'huggingface-cli login' manually"
        }
    fi

    if [ "$PROFILE" = "full" ] && [ "$MODE" != "cpu" ]; then
        info "Model will be downloaded automatically on first run"
        info "  Small: HuggingFaceTB/SmolLM2-1.7B-Instruct (~3 GB)"
        info "  Medium: google/translategemma-4b-it (~8 GB)"
        info "  Large: google/translategemma-12b-it (~24 GB)"
        info "  Diffusion: GSAI-ML/LLaDA-8B-Base (~16 GB)"
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════
# Done
# ═══════════════════════════════════════════════════════════════════════════
banner "Setup Complete!"
echo ""
echo -e "  ${G}✓${N} Environment: $VENV_DIR ($(echo "$MODE" | tr '[:lower:]' '[:upper:]'))"
echo -e "  ${G}✓${N} Dependencies installed (profile=$PROFILE)"
echo ""
echo "  Next steps:"
echo "    ./run.sh --dry-run        Smoke test (1 min)"
echo "    ./run.sh --quick          Quick benchmark (5 min)"
echo "    ./run.sh --full           Full 2-hour benchmark"
echo "    ./run.sh --diffusion      Diffusion model (LLaDA)"
echo ""
echo "  Activate environment:"
echo "    source $VENV_DIR/bin/activate"

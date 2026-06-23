"""Central constants — single source of truth for all magic numbers.

Import from here instead of hardcoding values.  When a model or hardware
changes, update these constants in one place.
"""

# ── Model architecture defaults (TranslateGemma 4B) ──
# These align with the default model in schema.py: google/translategemma-4b-it
# head_dim=256 and vocab_size=262_144 are shared across all Gemma 3/4 family sizes.
# Reference for 12B Gemma 3 variant (not used as default):
#   num_layers=48, hidden_size=3840, num_kv_heads=8
DEFAULT_NUM_LAYERS = 36
DEFAULT_NUM_KV_HEADS = 4
DEFAULT_HEAD_DIM = 256
DEFAULT_HIDDEN_SIZE = 2560
DEFAULT_VOCAB_SIZE = 262_144  # shared across all Gemma variants

# ── Model presets ─────────────────────────────────────────────────────────
# FIXME: Architecture defaults now live in benchmark.config.model_presets.MODEL_PRESETS.
# Use model_presets.resolve_architecture_defaults() or get_preset_by_name()
# instead of hardcoding architecture values.  MODEL_ARCHITECTURES is retained
# only for test_coverage_gaps.py sanity checks.
MODEL_ARCHITECTURES = {
    "4B":  {"num_layers": 36, "num_kv_heads": 4,  "head_dim": 256},
    "E2B": {"num_layers": 26, "num_kv_heads": 4,  "head_dim": 256},
    "E4B": {"num_layers": 34, "num_kv_heads": 8,  "head_dim": 256},
    "26B-A4B": {"num_layers": 48, "num_kv_heads": 8, "head_dim": 256},
}

# ── PagedAttention ──
PAGED_BLOCK_SIZE = 16
PAGED_NUM_BLOCKS_LARGE_GPU = 1024   # >80 GB VRAM
PAGED_NUM_BLOCKS_SMALL_GPU = 512    # ≤80 GB VRAM
PAGED_LARGE_GPU_THRESHOLD_GB = 80

# ── CUDA Graph ──
CUDA_GRAPH_DEFAULT_BATCH_SIZE = 4

# ── Memory budget ──
GPU_MEMORY_BUDGET_FRACTION = 0.95
GPU_MEMORY_RESERVE_BYTES = 4 * 1024**3  # 4 GiB headroom

# ── Pipeline timeouts (seconds) ──
LOADER_JOIN_TIMEOUT = 30
WORKER_JOIN_TIMEOUT = 10
SENTINEL_PUT_TIMEOUT = 1.0
BATCH_COLLECT_TIMEOUT = 5.0
TOKENISER_GET_TIMEOUT = 1.0

# ── Tokenizer ──
DEFAULT_MAX_SEQ_LEN = 2048
DEFAULT_TRUNCATION_LENGTH = 2048

# ── Warmup ──
WARMUP_SHORT_BATCHES = 5
WARMUP_LONG_BATCHES = 5
WARMUP_SHORT_TOKENS = 20
WARMUP_LONG_TOKENS = 128

# ── Quality ──
QUALITY_BATCH_SIZE = 32
QUALITY_BLEU_TARGET = 25
QUALITY_CHRF_TARGET = 54
QUALITY_COMET_TARGET = 0.72
QUALITY_BERTSORE_TARGET = 0.55
COMET_BATCH_SIZE = 8
# ── Tokenizer ──
# Use "intl" tokenizer instead of "13a" because "13a" strips all non-ASCII
# characters, destroying Turkish-specific characters (s-cedilla, g-breve,
# u/o-umlaut, c-cedilla, dotted/dotless I).  "intl" preserves these characters
# and is the recommended sacrebleu tokenizer for multilingual text.
SACREBLEU_TOKENIZER = "intl"
# Maximum new tokens for quality benchmark translations.  Reference sentences
# average ~42 characters (~10-20 tokens), so 128 is generous.  The translation
# prompt template adds ~100 tokens of instruction overhead, leaving ample room
# for even long reference translations.  Using model.max_new_tokens (default
# 512) would waste 20-50× more GPU time per sentence.
QUALITY_MAX_NEW_TOKENS = 128
# Gemma models emit <end_of_turn> (token 106) to signal completion of their
# response turn.  Without including 106 in the EOS set, generate() continues
# until max_new_tokens is exhausted, producing spurious repetition.
END_OF_TURN_TOKEN_ID = 106

# ── Metrics ──
POWERMETRICS_CACHE_TTL = 5.0
POWERMETRICS_TIMEOUT = 3
DEFAULT_SAMPLE_RATE_HZ = 1
METRICS_FLUSH_INTERVAL = 10
BATCH_FLUSH_INTERVAL = 50
MAX_METRICS_BUFFER_SIZE = 10_000   # drop oldest samples if buffer exceeds this after flush failure

# ── Shuffle load ──
SHUFFLE_LOAD_TIMEOUT = 15
MAX_IN_MEMORY_DOCS = 10_000_000

# ── External shuffle ──
# Byte budget for the in-memory shuffle buffer (2 GiB).  When the estimated
# total uncompressed text size exceeds this, the loader switches from
# in-memory Fisher-Yates to a disk-backed external sort.
SHUFFLE_MEMORY_BUDGET_BYTES = 2 * 1024**3  # 2 GiB
# Maximum number of open run files during the k-way merge.  If the number
# of sorted-run files exceeds this, the loader performs intermediate
# multi-pass merges to stay within OS file-descriptor limits.
SHUFFLE_MAX_OPEN_RUNS = 256
# Memory-overhead multiplier for estimating Python string object cost.
# A Python str has ~49 bytes base overhead plus 1 byte per char (ASCII).
# The multiplier 2.0 conservatively accounts for tuple overhead, the text
# copy, and the string object itself.
SHUFFLE_BYTES_PER_CHAR_OVERHEAD = 2.0

# ── Diffusion ──
DEFAULT_DIFFUSION_STEPS = 256
DEFAULT_NOISE_SCHEDULE = "cosine"
DEFAULT_GUIDANCE_SCALE = 1.0
DEFAULT_TARGET_LENGTH_MULTIPLIER = 2.0

# ── Checkpoint ──
DEFAULT_CHECKPOINT_INTERVAL = 300
CHECKPOINT_ROTATION = 3

# ── Thread pool ──
METRICS_PARALLEL_WORKERS = 3

# ── Registry ──
DIFFUSION_KEYWORDS = [
    "diffusion", "ddpm", "mdlm", "diffuseq",
    "diffullm", "sedd", "d3pm", "plaid",
    "diffusion-lm", "diffusionbert", "llada", "e2d2",
    "bd3lm", "bd3-lm", "block-diffusion", "block_diffusion",
    "diffusiongemma",
]

# ── QAT model detection keywords ──────────────────────────────────────────
# Keywords for detecting QAT models (checked against model_path by the
# autoregressive backend).  Model paths are resolved through MODEL_PRESETS
# in model_presets.py.

QAT_MODEL_KEYWORDS: tuple[str, str, str] = (
    "qat", "qat-mobile", "q4_0",
)

# DiffusionGemma model path and default steps.
DIFFUSION_GEMMA_MODEL_PATH = "google/diffusiongemma-26B-A4B-it"
# DiffusionGemma uses fewer steps than LLaDA — 128 is the recommended default.
DIFFUSION_GEMMA_DEFAULT_STEPS = 128
# DiffusionGemma noise schedule — "linear" or "cosine".
DIFFUSION_GEMMA_NOISE_SCHEDULE = "linear"

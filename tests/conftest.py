import json
import os
import warnings
import pytest
from pathlib import Path

TESTS_DIR = Path(__file__).parent
FIXTURE_DIR = TESTS_DIR / "fixtures"

# ---------------------------------------------------------------------------
# Strict fixtures mode — set TR_STRICT_FIXTURES=1 to fail tests when real
# data is unavailable, instead of silently falling back to auto-generated
# synthetic garbage that produces green CI with zero signal.
# ---------------------------------------------------------------------------
_STRICT_FIXTURES = os.environ.get("TR_STRICT_FIXTURES", "").lower() in (
    "1", "true", "yes", "on",
)

# ---------------------------------------------------------------------------
# Paths to REAL project data -- these are the primary sources for tests.
# ---------------------------------------------------------------------------
_PROJECT_DATA_DIR = TESTS_DIR.parent / "data"
_REAL_FINEWEB_GZ = _PROJECT_DATA_DIR / "input" / "fineweb_en_sample.jsonl.gz"
_REAL_GOLDEN_REFERENCES = _PROJECT_DATA_DIR / "references" / "golden_en_tr.jsonl"

# Fallback test fixtures in case real data is absent.
_FIXTURE_INPUT_JSONL = FIXTURE_DIR / "sample_input.jsonl"
_FIXTURE_INPUT_GZ = FIXTURE_DIR / "sample_input.jsonl.gz"
_FIXTURE_GOLDEN_JSONL = FIXTURE_DIR / "golden_en_tr.jsonl"

# WARNING: These auto-generation routines exist ONLY as a last-resort escape
# hatch so CI does not hard-crash when *both* real data and pre-generated
# fixtures are missing.  Synthetically-generated data produces meaningless
# test results and MUST NOT be treated as a valid signal of correctness.
_LAST_RESORT_NUM_DOCS = 100


def _fail_or_warn_on_fallback(fixture_name: str, real_path: Path, fixture_path: Path):
    """Fail in strict mode, warn otherwise when falling back to auto-generated data.

    In strict mode (TR_STRICT_FIXTURES=1), tests fail immediately when real data
    is unavailable.  This prevents silently passing tests on synthetic data that
    produces meaningless green CI results.
    """
    msg = (
        f"{fixture_name} has no real data available. "
        f"Real data not found at {real_path}. "
        f"Fixture not found at {fixture_path}. "
    )
    if _STRICT_FIXTURES:
        pytest.fail(
            msg + "TR_STRICT_FIXTURES=1 is set — refusing to generate synthetic data."
        )
    warnings.warn(
        msg + "Auto-generating synthetic data. Results are MEANINGLESS.",
        UserWarning,
        stacklevel=3,
    )


def _auto_generate_jsonl(path, num_docs):
    """LAST RESORT: auto-generate a synthetic JSONL fixture.

    WARNING: Tests backed by auto-generated data are NOT meaningful.
    Delete the generated file and provide real data before trusting results.
    """
    warnings.warn(
        f"Auto-generating synthetic JSONL fixture at {path!s}. "
        f"Tests backed by auto-generated data are NOT meaningful. "
        f"Provide real data at data/input/ or tests/fixtures/.",
        UserWarning,
        stacklevel=2,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(num_docs):
            f.write(
                json.dumps(
                    {
                        "text": (
                            f"This is auto-generated English text number {i} "
                            f"for translation benchmarking purposes. "
                            f"THIS DATA IS NOT REAL."
                        )
                    }
                )
                + "\n"
            )
    return str(path)


def _auto_generate_jsonl_gz(path, num_docs):
    """LAST RESORT: auto-generate a synthetic gzipped JSONL fixture.

    WARNING: Tests backed by auto-generated data are NOT meaningful.
    Delete the generated file and provide real data before trusting results.
    """
    warnings.warn(
        f"Auto-generating synthetic gzipped JSONL fixture at {path!s}. "
        f"Tests backed by auto-generated data are NOT meaningful. "
        f"Provide real data at data/input/ or tests/fixtures/.",
        UserWarning,
        stacklevel=2,
    )
    import gzip

    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(str(path), "wt", encoding="utf-8") as f:
        for i in range(num_docs):
            f.write(
                json.dumps(
                    {
                        "text": (
                            f"This is auto-generated compressed English text "
                            f"number {i} for testing. THIS DATA IS NOT REAL."
                        )
                    }
                )
                + "\n"
            )
    return str(path)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def real_tokenizer():
    """Load the real google/translategemma-4b-it tokenizer via HuggingFace.

    Session-scoped so multiple test files can reuse the same loaded instance.

    Skips the entire test session if HuggingFace is unreachable or the
    model is not available.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        pytest.skip(f"transformers not installed: {e}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "google/translategemma-4b-it",
            trust_remote_code=False,
        )
    except Exception as e:
        pytest.skip(f"HF tokenizer unavailable: {e}")

    return tokenizer


@pytest.fixture
def fixture_dir():
    return FIXTURE_DIR


@pytest.fixture
def sample_jsonl_path():
    """Return a path to a real English JSONL file.

    Resolution order:
      1. Real data:  data/input/fineweb_en_sample.jsonl.gz
         (The Loader handles .gz extensions transparently.)
      2. Test fixture:  tests/fixtures/sample_input.jsonl
         (Pre-generated snapshot for offline / CI use.)
      3. LAST RESORT: auto-generate a synthetic file.
         WARNING: results from this path are MEANINGLESS.
    """
    # 1 -- real data
    if _REAL_FINEWEB_GZ.exists():
        return str(_REAL_FINEWEB_GZ)

    # 2 -- pre-generated test fixture
    if _FIXTURE_INPUT_JSONL.exists():
        return str(_FIXTURE_INPUT_JSONL)

    # 3 -- LAST RESORT (generates meaningless data)
    _fail_or_warn_on_fallback(
        "sample_jsonl_path", _REAL_FINEWEB_GZ, _FIXTURE_INPUT_JSONL,
    )
    return _auto_generate_jsonl(_FIXTURE_INPUT_JSONL, _LAST_RESORT_NUM_DOCS)


@pytest.fixture
def sample_jsonl_gz_path():
    """Return a path to a real gzipped English JSONL file.

    Resolution order:
      1. Real data:  data/input/fineweb_en_sample.jsonl.gz
      2. Test fixture:  tests/fixtures/sample_input.jsonl.gz
         (Pre-generated snapshot for offline / CI use.)
      3. LAST RESORT: auto-generate a synthetic file.
         WARNING: results from this path are MEANINGLESS.
    """
    # 1 -- real data
    if _REAL_FINEWEB_GZ.exists():
        return str(_REAL_FINEWEB_GZ)

    # 2 -- pre-generated test fixture
    if _FIXTURE_INPUT_GZ.exists():
        return str(_FIXTURE_INPUT_GZ)

    # 3 -- LAST RESORT (generates meaningless data)
    _fail_or_warn_on_fallback(
        "sample_jsonl_gz_path", _REAL_FINEWEB_GZ, _FIXTURE_INPUT_GZ,
    )
    return _auto_generate_jsonl_gz(_FIXTURE_INPUT_GZ, _LAST_RESORT_NUM_DOCS)


@pytest.fixture
def golden_references_path():
    """Return a path to real golden English->Turkish reference translations.

    Resolution order:
      1. Real data:  data/references/golden_en_tr.jsonl
         (Human-verified Turkish translations.)
      2. Test fixture:  tests/fixtures/golden_en_tr.jsonl
         (A small subset of real references for offline testing.)
      3. If neither exists, the test is SKIPPED.
         Auto-generating fake Turkish text is UNACCEPTABLE
         for an academic-grade benchmark.
    """
    # 1 -- real data
    if _REAL_GOLDEN_REFERENCES.exists():
        return str(_REAL_GOLDEN_REFERENCES)

    # 2 -- pre-generated test fixture
    if _FIXTURE_GOLDEN_JSONL.exists():
        return str(_FIXTURE_GOLDEN_JSONL)

    # 3 -- NO auto-generation. Skip.
    pytest.skip(
        "No golden references file found.  Provide real references at "
        f"{_REAL_GOLDEN_REFERENCES} or {_FIXTURE_GOLDEN_JSONL}."
    )


@pytest.fixture
def mock_config_dict():
    return {
        "backend": "auto",
        "model": {
            "model_path": "google/translategemma-4b-it",
            "tokenizer_path": "",
            "max_input_tokens": 512,
            "max_new_tokens": 512,
            "temperature": 0.0,
            "do_sample": False,
            "num_beams": 1,
            "dtype": "auto",
            "tensor_parallel_size": 0,
            "use_flash_attention": True,
        },
        "runtime": {
            "target_duration_seconds": 60,
            "checkpoint_interval_seconds": 30,
            "heartbeat_interval_seconds": 10,
            "metrics_sample_rate_hz": 1,
            "seed": 42,
        },
        "data": {
            "input_paths": [str(FIXTURE_DIR / "sample_input.jsonl")],
            "output_dir": str(FIXTURE_DIR / "output"),
            "reference_set_path": str(FIXTURE_DIR / "golden_en_tr.jsonl"),
            "shard_size_mb": 100,
            "prefetch_workers": 2,
            "shuffle": False,
            "min_chunk_tokens": 10,
            "max_garbage_ratio": 0.95,
            "chunk_overlap_tokens": 50,
        },
        "extrapolation": {
            "total_clearnet_non_tr_tokens": 6_230_000_000_000,
            "gpu_cost_per_hour_usd": None,
        },
    }

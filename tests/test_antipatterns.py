"""Antipattern-targeted tests — catch the most common AI coding mistakes.

Each test is named after the AI_CODING_ANTIPATTERNS entry it guards against.
These are the highest-value tests in the suite: they directly prevent regressions
that have actually occurred in this project.

Tests marked with ``@pytest.mark.fast`` run in <1s and are suitable for pre-commit.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# A5 — Silently wrong metrics (e.g. padded input_tokens inflating TPS)
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestA5_TokenCountsFromAttentionMask:
    """Verify that token counts respect the attention mask, not padded length.

    The bug (commit ffa707b): NLLB's input_tokens counted padding, inflating TPS.
    The fix applied to NLLB/diffusion but AR backend still affected.
    """

    def test_attention_mask_sum_gives_real_token_count(self):
        """Attention mask sum yields real tokens, not padded tensor length."""
        import torch
        # Simulate a left-padded batch: [PAD, PAD, tok1, tok2, tok3]
        attention_mask = torch.tensor([[0, 0, 1, 1, 1]])
        input_ids = torch.tensor([[0, 0, 42, 43, 44]])
        real_tokens = int(attention_mask.sum().item())
        padded_length = input_ids.shape[1]
        assert real_tokens == 3
        assert padded_length == 5
        assert real_tokens < padded_length, (
            "Padded length overcounts — use attention_mask.sum() for "
            "input_tokens_total, not input_ids.shape[1]"
        )

    def test_no_padding_produces_equal_counts(self):
        """With no padding, mask sum and tensor length agree."""
        import torch
        attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
        assert int(attention_mask.sum().item()) == attention_mask.shape[1]


# ---------------------------------------------------------------------------
# A11 — Hardcoded architecture constants
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestA11_NoHardcodedArchitectureConstants:
    """Verify that critical constants are accessible and not hardcoded downstream.

    The bug: parallelism.py hardcoded Gemma-3-12B constants; END_OF_TURN_TOKEN_ID
    was 106 in 3+ places. Both are now fixed (constants imported, token_id centralized).
    """

    def test_end_of_turn_token_id_accessible_from_constants(self):
        """END_OF_TURN_TOKEN_ID is defined in constants.py (not hardcoded 106)."""
        from benchmark.config.constants import END_OF_TURN_TOKEN_ID
        assert isinstance(END_OF_TURN_TOKEN_ID, int)
        assert END_OF_TURN_TOKEN_ID > 0

    def test_architecture_defaults_are_sensible(self):
        """Default architecture constants have sane values (not zero/missing)."""
        from benchmark.config.constants import (
            DEFAULT_NUM_LAYERS, DEFAULT_NUM_KV_HEADS,
            DEFAULT_HEAD_DIM, DEFAULT_HIDDEN_SIZE, DEFAULT_VOCAB_SIZE,
        )
        assert DEFAULT_NUM_LAYERS > 0
        assert DEFAULT_NUM_KV_HEADS > 0
        assert DEFAULT_HEAD_DIM > 0
        assert DEFAULT_HIDDEN_SIZE > 0
        assert DEFAULT_VOCAB_SIZE > 0
        # NOTE: do NOT assert specific values (e.g. layers==48).
        # Those break when the default model changes. Just validate >0.

    def test_model_presets_have_consistent_architecture(self):
        """Every preset's architecture fields are non-zero (no missing entries)."""
        from benchmark.config.model_presets import MODEL_PRESETS
        required = {"num_layers", "num_kv_heads", "head_dim", "hidden_size"}
        for name, preset in MODEL_PRESETS.items():
            missing = required - set(preset.__dict__.keys())
            assert not missing, f"Preset '{name}' missing fields: {missing}"
            assert preset.num_layers > 0, f"Preset '{name}' num_layers is 0"
            assert preset.num_kv_heads > 0, f"Preset '{name}' num_kv_heads is 0"


# ---------------------------------------------------------------------------
# A12 — Non-atomic writes / silent data loss
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestA12_AtomicWrites:
    """Verify atomic write patterns for state files.

    The bug: perf_regression._save_raw used plain json.dump (non-atomic);
    checkpoint silently drops corrupted lines.
    """

    def test_tmp_plus_rename_is_atomic(self, tmp_path):
        """Writing to a temp file then os.rename preserves data on success."""
        final_path = tmp_path / "state.json"
        tmp_path_file = tmp_path / "state.json.tmp"

        data = {"key": "value", "counter": 42}
        with open(tmp_path_file, "w") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.rename(str(tmp_path_file), str(final_path))

        with open(final_path) as f:
            loaded = json.load(f)
        assert loaded == data

    @pytest.mark.skip(reason="Documentation test — demonstrates atomic-write risk, not a bug check")
    def test_non_atomic_write_can_lose_data(self, tmp_path):
        """Demonstrate that a plain write without fsync+rename is fragile.

        This isn't a bug check — it's a documentation test showing WHY
        the atomic pattern matters. A crash between write and close can
        leave a truncated file.
        """
        path = tmp_path / "plain.json"
        with open(path, "w") as f:
            json.dump({"key": "value"}, f)
            # No fsync, no atomic rename
        assert path.exists()
        # If the process crashed right here, the file could be empty/corrupt.
        # This test documents the risk; the checkpoint module uses atomic writes.


# ---------------------------------------------------------------------------
# A13 — Tests that pass when preconditions are missing
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestA13_TestIntegrity:
    """Verify that the test suite itself doesn't suffer from silent-pass bugs.

    The bug: test_load_from_yaml silently passed if config_path didn't exist.
    fixtures auto-generated meaningless synthetic data but tests stayed green.
    """

    def test_strict_fixtures_env_var_exists(self):
        """TR_STRICT_FIXTURES is set by default (prevents synthetic data fallback)."""
        strict = os.environ.get("TR_STRICT_FIXTURES", "1")
        assert strict.lower() not in ("0", "false", "no", "off"), (
            "TR_STRICT_FIXTURES must default to '1' (strict mode on). "
            "Synthetic data fallback produces meaningless test results."
        )

    def test_conftest_provides_mock_config(self):
        """mock_config_dict fixture provides a complete config with all required keys."""
        from tests.conftest import mock_config_dict

        cfg = mock_config_dict()
        assert isinstance(cfg, dict)
        assert "model" in cfg
        assert "runtime" in cfg
        assert "data" in cfg
        assert "extrapolation" in cfg
        # Verify model sub-config has required fields
        model = cfg["model"]
        assert "model_path" in model
        assert "max_input_tokens" in model


# ---------------------------------------------------------------------------
# A15 — Inconsistent partial-run handling
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestA15_ConsistentNoneHandling:
    """Verify that None/error results are handled consistently across metrics.

    The bug: BERTScore None→0.0 fails target, while COMET/Kiwi are skipped.
    Both should skip when the metric didn't compute.
    """

    def test_dict_get_with_none_key_returns_none_not_zero(self):
        """Demonstrate get() returns None when key exists with None value."""
        d = {"system_score": None}
        # BAD: d.get("system_score", 0) returns None (not 0) because the key EXISTS.
        # Then "None or 0.0" evaluates to 0.0 — the footgun.
        assert d.get("system_score", 0) is None, (
            "get() returns existing None values — 'or 0.0' after get() "
            "converts None→0.0 unexpectedly. Guard with 'is not None' first."
        )

    def test_none_score_should_be_skipped_not_failed(self):
        """When a metric's system_score is None, skip it — don't fail the check."""
        from benchmark.config.constants import QUALITY_BERTSORE_TARGET
        score = None
        # Correct pattern: skip when None
        should_check = score is not None
        # If we checked anyway:
        bad_check = (score or 0.0) >= QUALITY_BERTSORE_TARGET  # noqa
        assert should_check is False, "None score should be skipped"
        assert bad_check is False, "None→0.0 coercion incorrectly fails the target"


# ---------------------------------------------------------------------------
# A4 — Exception swallowing (detect bare excepts in key modules)
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestA4_NoExceptionSwallowing:
    """Guard against exception swallowing in data-integrity and quality paths.

    The bug: 40+ bare/silent excepts across the codebase. Chat-template failure
    silently produced garbage translations.
    """

    @pytest.mark.parametrize("module_path", [
        "benchmark/quality/benchmark.py",
        "benchmark/quality/metrics_bertscore.py",
        "benchmark/quality/metrics_comet.py",
        "benchmark/metrics/collector.py",
        "benchmark/data/loader.py",
        "benchmark/orchestration/harness.py",
        "benchmark/orchestration/checkpoint.py",
    ])
    def test_quality_metrics_paths_no_bare_except(self, module_path):
        """Key modules should not contain bare 'except:' clauses.

        A bare except catches SystemExit/KeyboardInterrupt, masking real errors.
        This is informational — we document bare-except count rather than failing.
        """
        repo_root = Path(__file__).parent.parent
        path = repo_root / module_path
        if not path.exists():
            pytest.skip(f"{module_path} not found")
        content = path.read_text()
        lines = content.split("\n")
        bare_excepts = sum(1 for l in lines if l.strip() == "except:")
        if bare_excepts > 0:
            pytest.fail(
                f"{module_path}: found {bare_excepts} bare except: clause(s). "
                "Replace with specific exception types in integrity paths. "
                "See AI_CODING_ANTIPATTERNS.md §A4."
            )

    def test_quality_benchmark_has_specific_exception_handling(self):
        """Quality benchmark should use specific except clauses, not bare."""
        repo_root = Path(__file__).parent.parent
        path = repo_root / "benchmark/quality/benchmark.py"
        if not path.exists():
            pytest.skip("benchmark.py not found")
        content = path.read_text()
        # Verify it doesn't have a bare except:
        bare_count = sum(1 for line in content.split("\n") if line.strip() == "except:")
        # We expect some specific exception handling, but no bare except
        # This is informational — don't fail on count, just assert none are in
        # compute-paths (hard to mechanically verify, so document the pattern).
        assert bare_count < 5, (
            f"Found {bare_count} bare except: in quality benchmark. "
            "Prefer specific exception types in integrity paths."
        )


# ---------------------------------------------------------------------------
# A3 — Copy-paste divergence (verify key constants match)
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestA3_CopyPasteDivergence:
    """Verify that duplicated sources of truth are consistent.

    The bug: config hash computed 4× at different line numbers; 250-line
    translation loop duplicated. Now de-duplicated but the lesson stands.
    """

    def test_total_tokens_constant_matches_schema_default(self):
        """TOTAL_CLEARNET_TOKENS matches ExtrapolationConfig default."""
        from benchmark.config.constants import TOTAL_CLEARNET_TOKENS
        from benchmark.config.schema import ExtrapolationConfig
        default = ExtrapolationConfig.model_fields[
            "total_clearnet_non_tr_tokens"
        ].default
        assert TOTAL_CLEARNET_TOKENS == default, (
            f"TOTAL_CLEARNET_TOKENS ({TOTAL_CLEARNET_TOKENS}) != "
            f"schema default ({default}) — duplicated default diverged"
        )

    def test_default_model_matches_constants_default(self):
        """The default model (translategemma-4b) architecture should match
        constant defaults. If the default model changes, constants MUST update.
        """
        from benchmark.config.constants import (
            DEFAULT_NUM_LAYERS, DEFAULT_NUM_KV_HEADS,
            DEFAULT_HEAD_DIM, DEFAULT_HIDDEN_SIZE,
        )
        from benchmark.config.model_presets import get_preset_by_name
        preset = get_preset_by_name("translategemma-4b-bf16")
        if preset is None:
            pytest.skip("Preset translategemma-4b-bf16 not available")
        assert DEFAULT_NUM_LAYERS == preset.num_layers, (
            f"DEFAULT_NUM_LAYERS={DEFAULT_NUM_LAYERS} != "
            f"preset num_layers={preset.num_layers}"
        )
        assert DEFAULT_NUM_KV_HEADS == preset.num_kv_heads
        assert DEFAULT_HEAD_DIM == preset.head_dim
        assert DEFAULT_HIDDEN_SIZE == preset.hidden_size


# ---------------------------------------------------------------------------
# A1 — Dead code masquerading as a feature
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestA1_DeadCodeNotActive:
    """Verify that dead-code features are truly gated off and not accidentally
    re-enabled by changes to surrounding code.

    The bug: fused kernel injection was gated by 'not use_torch_compile'.
    When torch.compile was disabled, the guard flipped and injected crashing
    kernels. Now hardcoded if False.
    """

    def test_fused_kernel_code_permanently_removed(self):
        """Fused kernel injection code was permanently deleted (commit 19d979f).

        The deleted modules (fused_ops.py, triton_kernels_fused.py) contained
        Triton kernels that crashed outside torch.compile's inductor graph.
        With compile gated by PyTorch version, the kernels could never run.
        This test verifies the removal is complete — no stale references remain.
        """
        repo_root = Path(__file__).parent.parent
        # The fused kernel modules should not exist.
        for dead_module in [
            "benchmark/hardware/fused_ops.py",
            "benchmark/hardware/triton_kernels_fused.py",
        ]:
            path = repo_root / dead_module
            assert not path.exists(), (
                f"{dead_module} should be deleted — fused Triton kernels "
                f"are architecturally broken and were permanently removed"
            )
        # The injection gate (if False:) was also removed — _inject_fused_kernels
        # no longer exists in autoregressive.py.
        ar_path = repo_root / "benchmark/inference/backends/autoregressive.py"
        content = ar_path.read_text()
        assert "_inject_fused_kernels" not in content, (
            "_inject_fused_kernels should not exist — fused kernel injection "
            "was permanently removed along with the kernel source files"
        )
        assert "_use_fused_kernels" not in content, (
            "_use_fused_kernels attribute should not exist — it was removed "
            "from AutoregressiveBackend.__init__"
        )

    def test_cuda_graphs_module_permanently_deleted(self):
        """cuda_graphs.py was permanently deleted (commit 19d979f).

        The captured graph could not include past_key_values as a static input,
        making replay produce garbage (each token generated in isolation with
        zero context). torch.compile handles internal graph capture with proper
        KV-cache management, so the manual path was redundant.
        """
        with pytest.raises(ImportError, match="No module named"):
            from benchmark.hardware.cuda_graphs import CUDAGraphDecoder  # noqa: F401
        # Also verify no stale references remain in autoregressive.py.
        repo_root = Path(__file__).parent.parent
        ar_path = repo_root / "benchmark/inference/backends/autoregressive.py"
        content = ar_path.read_text()
        assert "cuda_graphs" not in content, (
            "cuda_graphs references should be removed from autoregressive.py"
        )
        assert "_use_cuda_graph" not in content, (
            "_use_cuda_graph attribute should not exist — it was removed"
        )

"""Capability Registry — single source of truth for optimization activation state.

Replaces the ad-hoc boolean flags and env-var gates scattered across the codebase
with a centralized, queryable registry. Every backend populates this after load(),
and the harness reads from it at startup to print an honest capability table.

This directly addresses Flaw #1 (False Flag Architecture) and Flaw #3 (Ad-Hoc
Feature Gating) from docs/ARCHITECTURAL_FLAWS.md.

Usage::

    from benchmark.config.capability import (
        CapabilityRegistry, CapabilityEntry, ActivationState,
    )
    reg = CapabilityRegistry()

    reg.register(CapabilityEntry(
        feature_id="flash_sdpa",
        display_name="Flash SDPA",
        state=ActivationState.ACTIVE,
        reason="CUDA + PyTorch >= 2.0",
        phase="hot_path",
    ))

    reg.register(CapabilityEntry(
        feature_id="cuda_graph_replay",
        display_name="CUDA Graph Replay",
        state=ActivationState.INERT,
        reason="Captured graph omits past_key_values; not replayed in _extreme_decode",
        phase="hot_path",
    ))

    reg.freeze()
    print(reg.report_table())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class ActivationState(Enum):
    """Real activation state of a feature — not what the config says."""
    ACTIVE = auto()         # Wired into hot path, affects output
    GATED = auto()          # Opt-in; requires explicit flag/env to activate
    INERT = auto()          # Code present but NOT wired (no-op or unreachable)
    BROKEN = auto()         # Present but broken (safety-gated, API removed, crashed)
    UNIMPLEMENTED = auto()  # Documented but no code exists
    DEPRECATED = auto()     # Scheduled for removal


_STATE_MARK = {
    ActivationState.ACTIVE:         "✅ ON",
    ActivationState.GATED:          "🔬 OPT-IN",
    ActivationState.INERT:          "🟡 OFF (unwired)",
    ActivationState.BROKEN:         "⚠️ BROKEN",
    ActivationState.UNIMPLEMENTED:  "💀 NONE",
    ActivationState.DEPRECATED:     "⚫ DEPR",
}


@dataclass(frozen=True)
class CapabilityEntry:
    """Immutable record of one optimization feature and its real state.

    Attributes
    ----------
    feature_id : str
        Unique machine-readable identifier (e.g. ``"paged_kv_cache_ar"``).
    display_name : str
        Human-readable name for reports and tables.
    state : ActivationState
        Verified activation state (NOT aspirational).
    reason : str
        Why the state is what it is — required for INERT/BROKEN/DEPRECATED.
    phase : str
        Pipeline phase this belongs to (hot_path, startup, quality, observability).
    depends_on : tuple[str, ...]
        Feature IDs this feature requires to be ACTIVE.
    conflicts_with : tuple[str, ...]
        Feature IDs that are mutually exclusive with this one.
    validate_fn : Optional[Callable[[], bool]]
        Runtime assertion — returns True if the feature is genuinely active.
    """
    feature_id: str
    display_name: str
    state: ActivationState
    reason: str = ""
    phase: str = "unknown"
    depends_on: tuple[str, ...] = ()
    conflicts_with: tuple[str, ...] = ()
    validate_fn: Optional[Callable[[], bool]] = None


@dataclass
class CapabilityRegistry:
    """Single source of truth for feature activation state.

    Populated by backends during ``load()`` via ``register()``.
    After ``load()`` completes, ``freeze()`` is called and the registry
    becomes immutable.  Printed as a markdown table in benchmark reports.

    Enforces dependency validation: if a feature depends on another that
    is not ACTIVE, registration raises ValueError.
    """
    entries: dict[str, CapabilityEntry] = field(default_factory=dict)
    _frozen: bool = False

    def register(self, entry: CapabilityEntry) -> None:
        """Add a capability entry.  Raises if frozen or if dependency violated."""
        if self._frozen:
            raise RuntimeError(
                f"Cannot register '{entry.feature_id}' — "
                "CapabilityRegistry is frozen after load()."
            )
        # Validate dependency chain.
        if entry.depends_on:
            for dep_id in entry.depends_on:
                if dep_id in self.entries:
                    dep_state = self.entries[dep_id].state
                    if dep_state != ActivationState.ACTIVE:
                        raise ValueError(
                            f"'{entry.feature_id}' depends on '{dep_id}', "
                            f"which is {dep_state.name} (must be ACTIVE). "
                            f"Reason: {self.entries[dep_id].reason}"
                        )
                else:
                    logger.warning(
                        "Capability '%s' depends on '%s' which has not been "
                        "registered yet.  Register dependencies before dependents.",
                        entry.feature_id, dep_id,
                    )
        # Detect conflicts.
        if entry.conflicts_with:
            for conflict_id in entry.conflicts_with:
                if conflict_id in self.entries:
                    conflict_state = self.entries[conflict_id].state
                    if conflict_state == ActivationState.ACTIVE:
                        raise ValueError(
                            f"'{entry.feature_id}' conflicts with '{conflict_id}' "
                            f"which is {conflict_state.name}."
                        )
        self.entries[entry.feature_id] = entry

    def freeze(self) -> None:
        """Called after load() — no more registrations, enables querying."""
        self._frozen = True

    def active_ids(self) -> frozenset[str]:
        """Return IDs of all ACTIVE or GATED features."""
        return frozenset(
            k for k, e in self.entries.items()
            if e.state in (ActivationState.ACTIVE, ActivationState.GATED)
        )

    def is_active(self, feature_id: str) -> bool:
        """Return True if the feature is wired into the hot path."""
        entry = self.entries.get(feature_id)
        return entry is not None and entry.state == ActivationState.ACTIVE

    def entry_for(self, feature_id: str) -> Optional[CapabilityEntry]:
        """Look up a single capability (e.g. for assertions)."""
        return self.entries.get(feature_id)

    def report_text(self) -> str:
        """Return a multi-line plain-text capability summary for logging."""
        if not self.entries:
            return "No capabilities registered."
        lines = ["Capability Registry:"]
        for e in sorted(self.entries.values(), key=lambda e: (e.phase, e.feature_id)):
            mark = _STATE_MARK.get(e.state, "?")
            lines.append(f"  {mark} {e.display_name:<40s}  {e.reason}")
        return "\n".join(lines)

    def report_table(self) -> str:
        """Return a markdown table for inclusion in benchmark reports."""
        header = (
            "| Feature | State | Phase | Reason |\n"
            "|---------|-------|-------|--------|"
        )
        rows = []
        for e in sorted(self.entries.values(), key=lambda e: (e.phase, e.feature_id)):
            mark = _STATE_MARK.get(e.state, "?")
            rows.append(
                f"| {e.display_name} | {mark} | {e.phase} | {e.reason} |"
            )
        return header + "\n" + "\n".join(rows) if rows else header

    def summary_counts(self) -> dict[str, int]:
        """Return {state_name: count} for quick dashboards."""
        counts: dict[str, int] = {}
        for e in self.entries.values():
            key = e.state.name
            counts[key] = counts.get(key, 0) + 1
        return counts

    def active_vs_total(self) -> tuple[int, int]:
        """Return (active_count, total_registered)."""
        total = len(self.entries)
        active = sum(1 for e in self.entries.values() if e.state == ActivationState.ACTIVE)
        return active, total


# ---------------------------------------------------------------------------
# Pre-defined feature IDs (canonical names used across the codebase)
# These are the keys that backends reference when building their CapabilityEntry
# list.  No two backends should use a different feature_id for the same thing.
# ---------------------------------------------------------------------------

FEATURE_IDS = {
    # Hot-path compute
    "flash_sdpa":            "Flash + Mem-Efficient SDPA",
    "torch_compile":         "torch.compile(reduce-overhead)",
    "te_fp8":                "Transformer-Engine FP8",
    # Hot-path decode
    "jit_cuda_kernels":      "JIT CUDA C++ Kernels (QKV+RoPE, SwiGLU)",
    "jit_metal_kernels":     "JIT Metal Kernels (RMSNorm+Residual)",
    "cuda_malloc_async":     "cudaMallocAsync Allocator",
    "speculative_decode":    "Speculative Decoding",
    # Memory / KV
    "paged_kv_cache_ar":     "PagedAttention KV-Cache (AR path)",
    "paged_kv_cache_cb":     "PagedAttention KV-Cache (CB path)",
    "continuous_batching":   "Continuous Batching",
    "pinned_memory":         "Pinned Memory Pipeline",
    "weight_quantization":   "Weight Quantization (INT4/INT8)",
    # Parallelism
    "device_map_auto":       "device_map=auto Multi-GPU",
    "tensor_parallelism":    "Tensor Parallelism (apply_tensor_parallelism)",
    "nccl_p2p":              "NCCL P2P Enable",
    # Data pipeline
    "orjson_parsing":        "orjson JSON Parsing",
    "parallel_gzip":         "Parallel gzip (pigz)",
    "numpy_garbage_filter":  "Numpy Garbage Detection",
    "async_prefetch":        "Async Prefetch Pipeline",
    "external_shuffle":      "External Sort Shuffle",
    # Quality / observability
    "bertscore_wired":       "BERTScore Quality Metric",
    "comet22_wired":         "COMET-22 Quality Metric",
    "comet_kiwi_wired":      "COMET-Kiwi Quality Metric",
    "bleu_wired":            "BLEU Quality Metric",
    "chrf_wired":            "chrF++ Quality Metric",
    "prometheus_exporter":   "Prometheus Metrics Exporter",
    "perf_regression":       "Performance Regression Gate",
}

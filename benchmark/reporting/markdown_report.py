"""Markdown report renderer — produces human-readable benchmark_report.md."""

import logging
import math
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from benchmark.config.constants import QUALITY_BLEU_TARGET, QUALITY_CHRF_TARGET, QUALITY_COMET_TARGET

logger = logging.getLogger(__name__)


def _fmt_num(val, fmt_spec=",.1f", default="N/A"):
    """Format a numeric value safely — guards against NaN, Inf, and non-numeric types.

    Returns *default* when the value is missing, non-finite, or not a number.
    """
    if val is None:
        return default
    if isinstance(val, (int, float)) and math.isfinite(val):
        return f"{val:{fmt_spec}}"
    if isinstance(val, bool):
        return default  # bool is a subclass of int — reject it
    return default


class MarkdownReportWriter:
    def write(self, output_dir: Path, report: dict) -> Path:
        report_dir = output_dir / "report"
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / "benchmark_report.md"
        md = self._render(report)
        fd, tmp_path = tempfile.mkstemp(dir=report_dir, suffix=".md", prefix=".tmp_report_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(md)
            # Atomic rename with cross-filesystem fallback.
            try:
                shutil.move(str(tmp_path), str(path))
            except OSError:
                shutil.copy2(str(tmp_path), str(path))
                os.unlink(tmp_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        logger.info(f"Markdown report -> {path}")
        return path

    def _render(self, r: dict) -> str:
        def _safe(d, *keys, default="N/A"):
            """Walk nested dict lookups without crashing on missing keys or non-dicts."""
            _NOT_FOUND = object()
            for key in keys:
                if not isinstance(d, dict):
                    return default
                d = d.get(key, _NOT_FOUND)
            return d if d is not _NOT_FOUND else default

        lines = []
        lines.append("# Turkish Corpus Translation Benchmark — Report")
        lines.append("")
        lines.append(f"**Generated**: {r.get('_metadata', {}).get('generated_at', 'N/A')}")
        lines.append("")
        cfg = r.get("config", {})
        model_cfg = cfg if isinstance(cfg, dict) else {}
        lines.append("## Configuration")
        lines.append("")
        lines.append(f"- **Backend**: {_safe(model_cfg, 'backend') or _safe(model_cfg, 'model', 'dtype')}")
        lines.append(f"- **Model**: {_safe(model_cfg, 'model', 'model_path') or _safe(model_cfg, 'model', 'dtype')}")
        lines.append(f"- **Target Duration**: {_safe(model_cfg, 'runtime', 'target_duration_seconds')} s")
        lines.append(f"- **Seed**: {_safe(model_cfg, 'runtime', 'seed')}")
        lines.append("")
        env = r.get("environment", {})
        lines.append("## Environment")
        lines.append("")
        lines.append(f"- **Backend**: {env.get('backend', 'N/A')}")
        lines.append(f"- **Device**: {env.get('device_name', 'N/A')}")
        lines.append(f"- **PyTorch**: {env.get('pytorch_version', 'N/A')}")
        lines.append("")
        batch = r.get("metrics", {}).get("batch", {})
        lines.append("## Throughput Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Mean tokens/second | {_fmt_num(batch.get('mean_tps'), '.1f')} |")
        lines.append(f"| Median tokens/second | {_fmt_num(batch.get('median_tps'), '.1f')} |")
        lines.append(f"| P5 tokens/second | {_fmt_num(batch.get('p5_tps'), '.1f')} |")
        lines.append(f"| P95 tokens/second | {_fmt_num(batch.get('p95_tps'), '.1f')} |")
        lines.append(f"| Std dev | {_fmt_num(batch.get('std_tps'), '.1f')} |")
        lines.append(f"| Total output tokens | {_fmt_num(batch.get('total_output_tokens'), ',.0f')} |")
        lines.append(f"| Total batches | {_fmt_num(batch.get('total_batches'), '.0f')} |")
        lines.append("")
        dev = r.get("metrics", {}).get("device", {})
        lines.append("## GPU Utilisation")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Mean GPU utilisation | {_fmt_num(dev.get('mean_util_pct'), '.1f')} % |")
        lines.append(f"| P99 GPU utilisation | {_fmt_num(dev.get('p99_util_pct'), '.1f')} % |")
        lines.append(f"| Data starvation (<20%) | {_fmt_num(dev.get('data_starvation_pct'), '.1f')} % |")
        lines.append(f"| Mean GPU temperature | {_fmt_num(dev.get('mean_temp_c'), '.1f')} °C |")
        lines.append("")
        q = r.get("quality", {})
        lines.append("## Quality Scores")
        lines.append("")
        lines.append("| Metric | Score | Target | Status |")
        lines.append("|---|---|---|---|")
        bleu_s = q.get("bleu", {}).get("score", "N/A")
        lines.append(f"| BLEU | {bleu_s} | >= {QUALITY_BLEU_TARGET} | {'✅' if isinstance(bleu_s, (int, float)) and bleu_s >= QUALITY_BLEU_TARGET else '—'} |")
        chrf_s = q.get("chrF", {}).get("score", "N/A")
        lines.append(f"| chrF++ | {chrf_s} | >= {QUALITY_CHRF_TARGET} | {'✅' if isinstance(chrf_s, (int, float)) and chrf_s >= QUALITY_CHRF_TARGET else '—'} |")
        comet_s = q.get("comet", {}).get("system_score", "N/A")
        lines.append(f"| COMET-22 | {comet_s} | >= {QUALITY_COMET_TARGET} | {'✅' if isinstance(comet_s, (int, float)) and comet_s >= QUALITY_COMET_TARGET else '—'} |")
        lines.append("")
        ext = r.get("extrapolation", {})
        lines.append("## Extrapolation")
        lines.append("")
        lines.append(f"- **Point estimate**: {_fmt_num(ext.get('days_point_estimate'), '.1f')} days")
        # Bootstrap CI (preferred when available — handles skewed distributions).
        if 'bootstrap_days_lower' in ext and ext['bootstrap_days_lower'] is not None:
            lines.append(f"- **Bootstrap 95% CI**: [{_fmt_num(ext.get('bootstrap_days_lower'), '.1f')}, {_fmt_num(ext.get('bootstrap_days_upper'), '.1f')}] days ({ext.get('n_batches', '?')} batches)")
        # Parametric CI (t-test, displayed alongside bootstrap when both exist).
        if 'days_95ci_lower' in ext and ext['days_95ci_lower'] is not None:
            lines.append(f"- **Parametric 95% CI**: [{_fmt_num(ext.get('days_95ci_lower'), '.1f')}, {_fmt_num(ext.get('days_95ci_upper'), '.1f')}] days")
        lines.append(f"- **GPU hours**: {_fmt_num(ext.get('gpu_hours'), '.1f')}")
        lines.append(f"- **Cost estimate**: ${_fmt_num(ext.get('estimated_cost_usd'), ',.2f')}")
        lines.append("")
        lines.append("## Caveats")
        lines.append("")
        lines.append("- Extrapolation assumes constant throughput and 24/7 operation.")
        lines.append("- The input sample may not be perfectly representative of the full corpus.")
        lines.append("- Thermal throttling may reduce throughput over longer runs.")
        return "\n".join(lines)

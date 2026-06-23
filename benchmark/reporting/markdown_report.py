"""Markdown report renderer — produces human-readable benchmark_report.md."""

import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

from benchmark.config.constants import QUALITY_BLEU_TARGET, QUALITY_CHRF_TARGET, QUALITY_COMET_TARGET

logger = logging.getLogger(__name__)


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
            os.rename(tmp_path, path)
        except Exception:
            os.unlink(tmp_path)
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
        lines.append(f"- **Model**: {_safe(model_cfg, 'model', 'model_path') or _safe(model_cfg, 'model')}")
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
        lines.append(f"| Mean tokens/second | {batch.get('mean_tps', 'N/A')} |")
        lines.append(f"| Median tokens/second | {batch.get('median_tps', 'N/A')} |")
        lines.append(f"| P5 tokens/second | {batch.get('p5_tps', 'N/A')} |")
        lines.append(f"| P95 tokens/second | {batch.get('p95_tps', 'N/A')} |")
        lines.append(f"| Std dev | {batch.get('std_tps', 'N/A')} |")
        val = batch.get("total_output_tokens", None)
        lines.append(f"| Total output tokens | {val:,.0f}" if isinstance(val, (int, float)) else f"| Total output tokens | {val or 'N/A'} |")
        val = batch.get("total_batches", None)
        lines.append(f"| Total batches | {val:.0f}" if isinstance(val, (int, float)) else f"| Total batches | {val or 'N/A'} |")
        lines.append("")
        dev = r.get("metrics", {}).get("device", {})
        lines.append("## GPU Utilisation")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Mean GPU utilisation | {dev.get('mean_util_pct', 'N/A')} % |")
        lines.append(f"| P99 GPU utilisation | {dev.get('p99_util_pct', 'N/A')} % |")
        lines.append(f"| Data starvation (<20%) | {dev.get('data_starvation_pct', 'N/A')} % |")
        lines.append(f"| Mean GPU temperature | {dev.get('mean_temp_c', 'N/A')} °C |")
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
        lines.append(f"- **Point estimate**: {ext.get('days_point_estimate', 'N/A')} days")
        lines.append(f"- **95% CI**: [{ext.get('days_95ci_lower', 'N/A')}, {ext.get('days_95ci_upper', 'N/A')}] days")
        lines.append(f"- **GPU hours**: {ext.get('gpu_hours', 'N/A')}")
        lines.append(f"- **Cost estimate**: ${ext.get('estimated_cost_usd', 'N/A')}")
        lines.append("")
        lines.append("## Caveats")
        lines.append("")
        lines.append("- Extrapolation assumes constant throughput and 24/7 operation.")
        lines.append("- The input sample may not be perfectly representative of the full corpus.")
        lines.append("- Thermal throttling may reduce throughput over longer runs.")
        return "\n".join(lines)

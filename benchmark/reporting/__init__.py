"""Report generation — aggregation, extrapolation, JSON and Markdown output."""

from benchmark.reporting.aggregator import MetricsAggregator
from benchmark.reporting.extrapolation import ExtrapolationModel
from benchmark.reporting.json_report import JSONReportWriter
from benchmark.reporting.markdown_report import MarkdownReportWriter

__all__ = ["MetricsAggregator", "ExtrapolationModel", "JSONReportWriter", "MarkdownReportWriter"]

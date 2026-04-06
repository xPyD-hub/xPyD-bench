"""Reporting module — extended metrics, formats, and rich terminal output."""

from xpyd_bench.reporting.formats import (
    export_csv_report,
    export_json_report,
    export_markdown_report,
    export_per_request,
    export_per_request_csv,
    format_text_report,
)
from xpyd_bench.reporting.metrics import compute_time_series
from xpyd_bench.reporting.rich_output import RichProgressReporter

__all__ = [
    "RichProgressReporter",
    "compute_time_series",
    "export_csv_report",
    "export_json_report",
    "export_markdown_report",
    "export_per_request",
    "export_per_request_csv",
    "format_text_report",
]

"""Rich terminal progress bar and live stats for benchmark runs."""

from __future__ import annotations

import sys
import time
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text

# Sparkline block characters (ascending height)
_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _sparkline(values: list[float], width: int = 20) -> str:
    """Return a sparkline string for the given values."""
    if not values:
        return ""
    recent = values[-width:]
    lo = min(recent)
    hi = max(recent)
    span = hi - lo if hi > lo else 1.0
    return "".join(
        _SPARK_CHARS[min(int((v - lo) / span * (len(_SPARK_CHARS) - 1)), len(_SPARK_CHARS) - 1)]
        for v in recent
    )


class LiveDashboard:
    """Real-time terminal dashboard using ``rich.live`` during benchmark execution.

    Shows: progress bar, current RPS, latency sparkline, error count, ETA.
    Auto-detects non-TTY and disables gracefully.
    """

    def __init__(self, total: int, *, disabled: bool = False) -> None:
        self._total = total
        self._completed = 0
        self._failed = 0
        self._latencies: list[float] = []
        self._start_time: float = 0.0
        self._last_rps_time: float = 0.0
        self._last_rps_count: int = 0
        self._current_rps: float = 0.0
        self._rps_history: list[float] = []
        self._disabled = disabled or not sys.stderr.isatty()
        self._console = Console(stderr=True)
        self._live: Live | None = None
        self._progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self._console,
        )
        self._task_id: Any = None

    @property
    def disabled(self) -> bool:
        return self._disabled

    def start(self) -> None:
        """Start the live dashboard."""
        self._start_time = time.monotonic()
        self._last_rps_time = self._start_time
        self._task_id = self._progress.add_task("Benchmarking", total=self._total)
        if self._disabled:
            return
        self._live = Live(
            self._build_layout(),
            console=self._console,
            refresh_per_second=4,
            transient=True,
        )
        self._live.start()

    def advance(self, *, success: bool = True, latency_ms: float | None = None) -> None:
        """Record one completed request."""
        if success:
            self._completed += 1
        else:
            self._failed += 1
        if latency_ms is not None:
            self._latencies.append(latency_ms)
        self._progress.update(self._task_id, advance=1)

        # Update RPS every ~1s
        now = time.monotonic()
        elapsed_since = now - self._last_rps_time
        if elapsed_since >= 1.0:
            new_requests = (self._completed + self._failed) - self._last_rps_count
            self._current_rps = new_requests / elapsed_since
            self._rps_history.append(self._current_rps)
            self._last_rps_count = self._completed + self._failed
            self._last_rps_time = now

        if self._live:
            self._live.update(self._build_layout())

    def stop(self) -> None:
        """Stop the live dashboard."""
        if self._live:
            self._live.stop()
            self._live = None

    def _build_layout(self) -> Table:
        """Build the dashboard layout."""
        grid = Table.grid(padding=(0, 1))
        grid.add_row(self._progress)

        # Stats line
        elapsed = time.monotonic() - self._start_time if self._start_time else 0.0
        overall_rps = (self._completed + self._failed) / elapsed if elapsed > 0 else 0.0

        stats = Text()
        stats.append("  RPS: ", style="bold")
        stats.append(f"{self._current_rps:.1f}", style="green")
        stats.append(f" (avg {overall_rps:.1f})", style="dim")
        stats.append("  │  Errors: ", style="bold")
        error_style = "red" if self._failed > 0 else "green"
        stats.append(f"{self._failed}", style=error_style)

        if self._latencies:
            recent = self._latencies[-10:]
            avg_lat = sum(recent) / len(recent)
            stats.append("  │  Latency: ", style="bold")
            stats.append(f"{avg_lat:.0f}ms", style="yellow")
            stats.append("  ", style="")
            stats.append(_sparkline(self._latencies), style="cyan")

        grid.add_row(stats)
        return grid

    def print_summary_table(self, result: Any) -> None:
        """Print a rich summary table after benchmark completion."""
        table = Table(title="Benchmark Results", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Mean", justify="right")
        table.add_column("P50", justify="right")
        table.add_column("P90", justify="right")
        table.add_column("P95", justify="right")
        table.add_column("P99", justify="right")

        for label, prefix in [
            ("TTFT", "ttft"),
            ("TPOT", "tpot"),
            ("ITL", "itl"),
            ("E2EL", "e2el"),
        ]:
            table.add_row(
                label,
                f"{getattr(result, f'mean_{prefix}_ms'):.2f}",
                f"{getattr(result, f'p50_{prefix}_ms'):.2f}",
                f"{getattr(result, f'p90_{prefix}_ms'):.2f}",
                f"{getattr(result, f'p95_{prefix}_ms'):.2f}",
                f"{getattr(result, f'p99_{prefix}_ms'):.2f}",
            )

        self._console.print()
        self._console.print(f"  [green]Completed:[/green] {result.completed}")
        self._console.print(f"  [red]Failed:[/red]    {result.failed}")
        self._console.print(
            f"  Duration:  {result.total_duration_s:.2f}s  |  "
            f"Throughput: {result.request_throughput:.2f} req/s, "
            f"{result.output_throughput:.2f} tok/s"
        )
        self._console.print(table)


class RichProgressReporter:
    """Wraps ``rich`` progress bar for benchmark request tracking.

    Usage::

        reporter = RichProgressReporter(total=100)
        reporter.start()
        for ... :
            reporter.advance()
        reporter.stop(result)
    """

    def __init__(self, total: int) -> None:
        self._total = total
        self._completed = 0
        self._failed = 0
        self._console = Console()
        self._progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            console=self._console,
        )
        self._task_id: Any = None
        self._live: Live | None = None

    def start(self) -> None:
        """Start the progress display."""
        self._task_id = self._progress.add_task("Benchmarking", total=self._total)
        self._progress.start()

    def advance(self, success: bool = True) -> None:
        """Record one completed request."""
        if success:
            self._completed += 1
        else:
            self._failed += 1
        self._progress.update(self._task_id, advance=1)

    def stop(self) -> None:
        """Stop the progress display."""
        self._progress.stop()

    def print_summary_table(self, result: Any) -> None:
        """Print a rich summary table after benchmark completion."""
        table = Table(title="Benchmark Results", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Mean", justify="right")
        table.add_column("P50", justify="right")
        table.add_column("P90", justify="right")
        table.add_column("P95", justify="right")
        table.add_column("P99", justify="right")

        for label, prefix in [
            ("TTFT", "ttft"),
            ("TPOT", "tpot"),
            ("ITL", "itl"),
            ("E2EL", "e2el"),
        ]:
            table.add_row(
                label,
                f"{getattr(result, f'mean_{prefix}_ms'):.2f}",
                f"{getattr(result, f'p50_{prefix}_ms'):.2f}",
                f"{getattr(result, f'p90_{prefix}_ms'):.2f}",
                f"{getattr(result, f'p95_{prefix}_ms'):.2f}",
                f"{getattr(result, f'p99_{prefix}_ms'):.2f}",
            )

        self._console.print()
        self._console.print(f"  [green]Completed:[/green] {result.completed}")
        self._console.print(f"  [red]Failed:[/red]    {result.failed}")
        self._console.print(
            f"  Duration:  {result.total_duration_s:.2f}s  |  "
            f"Throughput: {result.request_throughput:.2f} req/s, "
            f"{result.output_throughput:.2f} tok/s"
        )
        self._console.print(table)

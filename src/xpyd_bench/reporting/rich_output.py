"""Rich terminal progress bar and live stats for benchmark runs."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.live import Live
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table


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

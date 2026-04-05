"""Benchmark checkpointing and resume support (M74)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from xpyd_bench.bench.models import RequestResult


def _request_to_dict(r: RequestResult) -> dict[str, Any]:
    return asdict(r)


def _request_from_dict(d: dict[str, Any]) -> RequestResult:
    return RequestResult(**{k: v for k, v in d.items() if k in RequestResult.__dataclass_fields__})


class CheckpointManager:
    """Manages periodic checkpointing of benchmark progress."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        interval: int = 50,
        config_snapshot: dict[str, Any] | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.interval = max(1, interval)
        self.config_snapshot = config_snapshot or {}
        self._results: list[RequestResult] = []
        self._checkpoint_count = 0
        self._checkpoint_file = self.checkpoint_dir / "checkpoint.json"

    def record(self, result: RequestResult) -> None:
        """Record a completed request result, saving checkpoint if interval reached."""
        self._results.append(result)
        if len(self._results) % self.interval == 0:
            self.save()

    def save(self) -> None:
        """Write current results to checkpoint file."""
        data = {
            "version": 1,
            "config_snapshot": self.config_snapshot,
            "completed_count": len(self._results),
            "timestamp": time.time(),
            "requests": [_request_to_dict(r) for r in self._results],
        }
        # Write atomically via temp file
        tmp = self._checkpoint_file.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f)
        tmp.rename(self._checkpoint_file)
        self._checkpoint_count += 1

    @property
    def results(self) -> list[RequestResult]:
        return list(self._results)

    @property
    def checkpoint_file(self) -> Path:
        return self._checkpoint_file


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint file and return its data.

    Returns dict with keys: version, config_snapshot, completed_count, timestamp, requests.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "checkpoint.json"
    with open(path) as f:
        data = json.load(f)
    return data


def restore_requests(checkpoint_data: dict[str, Any]) -> list[RequestResult]:
    """Convert checkpoint request dicts back to RequestResult objects."""
    return [_request_from_dict(d) for d in checkpoint_data.get("requests", [])]


def validate_config_match(
    checkpoint_config: dict[str, Any],
    current_config: dict[str, Any],
) -> list[str]:
    """Compare checkpoint config with current config and return list of mismatches."""
    mismatches: list[str] = []
    # Check important keys
    important_keys = [
        "base_url", "endpoint", "model", "num_prompts", "request_rate",
        "max_concurrency", "input_len", "output_len", "backend",
    ]
    for key in important_keys:
        old = checkpoint_config.get(key)
        new = current_config.get(key)
        if old is not None and new is not None and old != new:
            mismatches.append(f"{key}: checkpoint={old!r} vs current={new!r}")
    return mismatches


def resume_main(argv: list[str] | None = None) -> None:
    """CLI entry point for 'xpyd-bench resume'."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="xpyd-bench resume",
        description="Resume a benchmark from a checkpoint file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file or directory containing checkpoint.json",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output checkpoint info as JSON",
    )

    args = parser.parse_args(argv)

    try:
        data = load_checkpoint(args.checkpoint)
    except FileNotFoundError:
        print(f"Error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: invalid checkpoint file: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print(f"Checkpoint: {args.checkpoint}")
        print(f"  Completed requests: {data.get('completed_count', 0)}")
        print(f"  Config snapshot keys: {sorted(data.get('config_snapshot', {}).keys())}")
        ts = data.get("timestamp")
        if ts:
            import datetime
            dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
            print(f"  Saved at: {dt.isoformat()}")

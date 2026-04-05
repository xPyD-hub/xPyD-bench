"""Tests for M46: Duration-based Benchmarking (--duration)."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from xpyd_bench.bench.models import BenchmarkResult
from xpyd_bench.cli import bench_main


class TestDurationCLI:
    """Test --duration CLI flag parsing and dry-run output."""

    def test_duration_flag_parsed(self, capsys: pytest.CaptureFixture[str]) -> None:
        """--duration should be accepted and shown in dry-run."""
        bench_main(["--dry-run", "--duration", "10", "--num-prompts", "5"])
        out = capsys.readouterr().out
        assert "Duration limit:" in out
        assert "10" in out

    def test_duration_mode_label(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Dry-run should label mode as duration when --duration is set."""
        bench_main(["--dry-run", "--duration", "30"])
        out = capsys.readouterr().out
        assert "duration" in out.lower()
        assert "prompts cycle" in out.lower()

    def test_no_duration_shows_count_mode(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Without --duration, mode should be count-based."""
        bench_main(["--dry-run", "--num-prompts", "50"])
        out = capsys.readouterr().out
        assert "count-based" in out.lower()


class TestDurationModel:
    """Test BenchmarkResult duration_limit field."""

    def test_duration_limit_default_none(self) -> None:
        result = BenchmarkResult()
        assert result.duration_limit is None

    def test_duration_limit_set(self) -> None:
        result = BenchmarkResult(duration_limit=60.0)
        assert result.duration_limit == 60.0

    def test_duration_limit_serializes(self) -> None:
        """duration_limit should appear in dataclass fields."""
        result = BenchmarkResult(duration_limit=30.0)
        d = asdict(result)
        assert d["duration_limit"] == 30.0


class TestDurationYAML:
    """Test YAML config support for duration."""

    def test_yaml_duration(self, capsys: pytest.CaptureFixture[str], tmp_path) -> None:
        """duration in YAML config should be applied."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("duration: 15\n")
        bench_main(["--dry-run", "--config", str(cfg)])
        out = capsys.readouterr().out
        assert "Duration limit:" in out
        assert "15" in out

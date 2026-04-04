"""Tests for M19: Dry Run Mode (--dry-run)."""

from __future__ import annotations

import json
import tempfile

import pytest

from xpyd_bench.cli import bench_main


class TestDryRunBasic:
    """Test basic dry-run execution and output."""

    def test_dry_run_exits_cleanly(self, capsys: pytest.CaptureFixture[str]) -> None:
        """--dry-run should exit without error on valid config."""
        bench_main(["--dry-run", "--num-prompts", "10"])
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "Dry run complete" in captured.out
        assert "Configuration is valid" in captured.out

    def test_dry_run_prints_execution_plan(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Dry run should print all key config values."""
        bench_main([
            "--dry-run",
            "--base-url", "http://test:9999",
            "--endpoint", "/v1/chat/completions",
            "--model", "test-model",
            "--num-prompts", "50",
            "--request-rate", "10",
            "--max-concurrency", "5",
            "--input-len", "128",
            "--output-len", "64",
        ])
        out = capsys.readouterr().out
        assert "http://test:9999" in out
        assert "/v1/chat/completions" in out
        assert "test-model" in out
        assert "50" in out
        assert "10" in out

    def test_dry_run_estimated_duration_finite_rate(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Estimated duration shown for finite request rate."""
        bench_main(["--dry-run", "--num-prompts", "100", "--request-rate", "10"])
        out = capsys.readouterr().out
        assert "Estimated duration:" in out
        assert "10.0s" in out

    def test_dry_run_estimated_duration_inf_rate(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Infinite rate shows 'all at once' message."""
        bench_main(["--dry-run", "--num-prompts", "10"])
        out = capsys.readouterr().out
        assert "all requests sent at once" in out

    def test_dry_run_no_http_requests(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Dry run should NOT import or call run_benchmark."""
        # If it tried to connect, it would fail since no server is running
        bench_main(["--dry-run", "--base-url", "http://nowhere:1", "--num-prompts", "5"])
        out = capsys.readouterr().out
        assert "DRY RUN" in out
        assert "Dry run complete" in out


class TestDryRunDataset:
    """Test dry-run with various dataset configurations."""

    def test_dry_run_synthetic_dataset(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Dry run with synthetic dataset prints stats."""
        bench_main(["--dry-run", "--dataset-name", "synthetic", "--num-prompts", "20"])
        out = capsys.readouterr().out
        assert "Dataset:" in out
        assert "Entries:" in out

    def test_dry_run_jsonl_dataset(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Dry run with JSONL dataset file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(5):
                f.write(json.dumps({"prompt": f"test prompt {i}"}) + "\n")
            f.flush()
            bench_main(["--dry-run", "--dataset-path", f.name, "--num-prompts", "5"])
        out = capsys.readouterr().out
        assert "Entries:" in out
        assert "5" in out

    def test_dry_run_invalid_dataset_path(self) -> None:
        """Dry run exits with error for missing dataset file."""
        with pytest.raises(SystemExit) as exc_info:
            bench_main(["--dry-run", "--dataset-path", "/nonexistent/file.jsonl"])
        assert exc_info.value.code == 1

    def test_dry_run_empty_jsonl(self) -> None:
        """Dry run exits with error for empty dataset."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("")
            f.flush()
            with pytest.raises(SystemExit) as exc_info:
                bench_main(["--dry-run", "--dataset-path", f.name])
            assert exc_info.value.code == 1


class TestDryRunYamlConfig:
    """Test dry-run with YAML config files."""

    def test_dry_run_from_yaml(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Dry run reads dry_run from YAML config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("dry_run: true\nnum_prompts: 15\nrequest_rate: 5\n")
            f.flush()
            bench_main(["--config", f.name])
        out = capsys.readouterr().out
        assert "DRY RUN" in out
        assert "Dry run complete" in out

    def test_dry_run_cli_overrides_yaml(self, capsys: pytest.CaptureFixture[str]) -> None:
        """CLI --dry-run works even without YAML."""
        bench_main(["--dry-run", "--num-prompts", "3"])
        out = capsys.readouterr().out
        assert "DRY RUN" in out


class TestDryRunScenario:
    """Test dry-run with scenario presets."""

    def test_dry_run_with_scenario(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Dry run works with --scenario flag."""
        bench_main(["--dry-run", "--scenario", "short"])
        out = capsys.readouterr().out
        assert "DRY RUN" in out
        assert "Scenario:" in out or "short" in out.lower() or "Dry run complete" in out


class TestDryRunAuth:
    """Test dry-run auth display."""

    def test_dry_run_shows_api_key_status(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Dry run shows API key status."""
        bench_main(["--dry-run", "--api-key", "sk-test", "--num-prompts", "5"])
        out = capsys.readouterr().out
        assert "API key:" in out
        assert "set" in out.lower()

    def test_dry_run_no_api_key(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Dry run shows no API key when not set."""
        bench_main(["--dry-run", "--num-prompts", "5"])
        out = capsys.readouterr().out
        assert "API key:" in out


class TestDryRunHeaders:
    """Test dry-run custom headers display."""

    def test_dry_run_shows_custom_headers(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Dry run displays custom headers."""
        bench_main([
            "--dry-run",
            "--header", "X-Test: hello",
            "--header", "X-Other: world",
            "--num-prompts", "5",
        ])
        out = capsys.readouterr().out
        assert "2 header(s)" in out
        assert "X-Test" in out
        assert "X-Other" in out

"""Tests for M30: Benchmark Result Storage & History."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from xpyd_bench.history import (
    _sparkline,
    auto_save_result,
    format_history_table,
    history_main,
    list_history,
)


def _make_result(
    model: str = "test-model",
    num_prompts: int = 100,
    completed: int = 100,
    failed: int = 0,
    request_throughput: float = 50.0,
    mean_ttft_ms: float = 10.0,
    mean_e2el_ms: float = 100.0,
    partial: bool = False,
    total_duration_s: float = 2.0,
) -> dict:
    return {
        "model": model,
        "num_prompts": num_prompts,
        "completed": completed,
        "failed": failed,
        "request_throughput": request_throughput,
        "output_throughput": 1000.0,
        "mean_ttft_ms": mean_ttft_ms,
        "mean_e2el_ms": mean_e2el_ms,
        "partial": partial,
        "total_duration_s": total_duration_s,
        "environment": {"timestamp": datetime.now().isoformat()},
    }


class TestSparkline:
    def test_empty(self):
        assert _sparkline([]) == ""

    def test_constant(self):
        s = _sparkline([5, 5, 5])
        assert len(s) == 3

    def test_ascending(self):
        s = _sparkline([0, 1, 2, 3, 4, 5, 6, 7])
        assert s[0] == "▁"
        assert s[-1] == "█"

    def test_single(self):
        s = _sparkline([42])
        assert len(s) == 1


class TestAutoSave:
    def test_auto_save_creates_file(self, tmp_path: Path):
        result = _make_result()

        class FakeArgs:
            request_rate = 10.0
            model = "gpt-test"
            backend = "openai"

        saved = auto_save_result(result, tmp_path, FakeArgs())
        assert saved.exists()
        assert saved.suffix == ".json"
        data = json.loads(saved.read_text())
        assert data["model"] == "test-model"

    def test_auto_save_creates_dir(self, tmp_path: Path):
        result_dir = tmp_path / "sub" / "dir"
        result = _make_result()

        class FakeArgs:
            request_rate = float("inf")
            model = None
            backend = "openai"

        saved = auto_save_result(result, result_dir, FakeArgs())
        assert saved.exists()
        assert result_dir.is_dir()

    def test_auto_save_inf_rate(self, tmp_path: Path):
        result = _make_result()

        class FakeArgs:
            request_rate = float("inf")
            model = "m"
            backend = "openai"

        saved = auto_save_result(result, tmp_path, FakeArgs())
        assert "inf" in saved.name


class TestListHistory:
    def test_empty_dir(self, tmp_path: Path):
        assert list_history(tmp_path) == []

    def test_nonexistent_dir(self, tmp_path: Path):
        assert list_history(tmp_path / "nope") == []

    def test_lists_results(self, tmp_path: Path):
        for i in range(3):
            p = tmp_path / f"openai-10qps-model-20250401-{120000 + i}.json"
            p.write_text(json.dumps(_make_result(request_throughput=50.0 + i)))
        summaries = list_history(tmp_path)
        assert len(summaries) == 3

    def test_last_n(self, tmp_path: Path):
        for i in range(5):
            p = tmp_path / f"openai-10qps-model-20250401-{120000 + i}.json"
            p.write_text(json.dumps(_make_result()))
        summaries = list_history(tmp_path, last_n=2)
        assert len(summaries) == 2

    def test_sorted_by_timestamp(self, tmp_path: Path):
        # Write out of order
        (tmp_path / "a-20250401-120002.json").write_text(json.dumps(_make_result()))
        (tmp_path / "a-20250401-120000.json").write_text(json.dumps(_make_result()))
        (tmp_path / "a-20250401-120001.json").write_text(json.dumps(_make_result()))
        summaries = list_history(tmp_path)
        timestamps = [s["timestamp"] for s in summaries]
        assert timestamps == sorted(timestamps)

    def test_skips_invalid_json(self, tmp_path: Path):
        (tmp_path / "bad.json").write_text("not json")
        (tmp_path / "good-20250401-120000.json").write_text(json.dumps(_make_result()))
        summaries = list_history(tmp_path)
        assert len(summaries) == 1


class TestFormatHistoryTable:
    def test_empty(self):
        output = format_history_table([])
        assert "No benchmark results found" in output

    def test_single_run(self, tmp_path: Path):
        p = tmp_path / "openai-10qps-model-20250401-120000.json"
        p.write_text(json.dumps(_make_result()))
        summaries = list_history(tmp_path)
        output = format_history_table(summaries)
        assert "test-model" in output

    def test_trends_shown_for_multiple_runs(self, tmp_path: Path):
        for i in range(3):
            p = tmp_path / f"openai-10qps-model-20250401-{120000 + i}.json"
            p.write_text(json.dumps(_make_result(request_throughput=50.0 + i * 10)))
        summaries = list_history(tmp_path)
        output = format_history_table(summaries)
        assert "Trends:" in output
        assert "Throughput" in output


class TestHistoryMain:
    def test_basic(self, tmp_path: Path, capsys):
        for i in range(2):
            p = tmp_path / f"r-20250401-{120000 + i}.json"
            p.write_text(json.dumps(_make_result()))
        history_main(["--result-dir", str(tmp_path)])
        captured = capsys.readouterr()
        assert "test-model" in captured.out

    def test_last_flag(self, tmp_path: Path, capsys):
        for i in range(5):
            p = tmp_path / f"r-20250401-{120000 + i}.json"
            p.write_text(json.dumps(_make_result()))
        history_main(["--result-dir", str(tmp_path), "--last", "2"])
        captured = capsys.readouterr()
        assert "test-model" in captured.out

"""Tests for M69: Benchmark Metadata & Notes."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

from xpyd_bench.bench.models import BenchmarkResult


class TestBenchmarkResultNote:
    """Test that BenchmarkResult has a note field."""

    def test_default_none(self):
        br = BenchmarkResult()
        assert br.note is None

    def test_set_note(self):
        br = BenchmarkResult(note="baseline A100")
        assert br.note == "baseline A100"


class TestCLINoteFlag:
    """Test --note CLI argument parsing."""

    def _make_parser(self):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        return parser

    def test_parse_note_flag(self):
        parser = self._make_parser()
        args = parser.parse_args(["--note", "my test run"])
        assert args.note == "my test run"

    def test_note_default_none(self):
        parser = self._make_parser()
        args = parser.parse_args([])
        assert args.note is None


class TestHistoryNote:
    """Test that history includes note in summaries and table output."""

    def _write_result(self, d: Path, data: dict, name: str = "r.json"):
        p = d / name
        p.write_text(json.dumps(data))
        return p

    def test_summary_includes_note(self):
        from xpyd_bench.history import _load_result_summary

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "result.json"
            p.write_text(json.dumps({
                "model": "gpt-4",
                "num_prompts": 10,
                "completed": 10,
                "failed": 0,
                "note": "after optimization",
            }))
            summary = _load_result_summary(p)
            assert summary["note"] == "after optimization"

    def test_summary_note_missing(self):
        from xpyd_bench.history import _load_result_summary

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "result.json"
            p.write_text(json.dumps({
                "model": "gpt-4",
                "num_prompts": 10,
                "completed": 10,
                "failed": 0,
            }))
            summary = _load_result_summary(p)
            assert summary["note"] is None

    def test_history_table_contains_note(self):
        from xpyd_bench.history import format_history_table

        summaries = [
            {
                "file": "r.json",
                "timestamp": datetime(2025, 1, 1, 12, 0, 0),
                "model": "gpt-4",
                "num_prompts": 10,
                "completed": 10,
                "failed": 0,
                "request_throughput": 5.0,
                "output_throughput": 100.0,
                "mean_ttft_ms": 50.0,
                "mean_e2el_ms": 200.0,
                "partial": False,
                "total_duration_s": 2.0,
                "tags": {},
                "note": "baseline run",
            }
        ]
        table = format_history_table(summaries)
        assert "Note" in table
        assert "baseline run" in table

    def test_history_table_no_note(self):
        from xpyd_bench.history import format_history_table

        summaries = [
            {
                "file": "r.json",
                "timestamp": datetime(2025, 1, 1, 12, 0, 0),
                "model": "gpt-4",
                "num_prompts": 10,
                "completed": 10,
                "failed": 0,
                "request_throughput": 5.0,
                "output_throughput": 100.0,
                "mean_ttft_ms": 50.0,
                "mean_e2el_ms": 200.0,
                "partial": False,
                "total_duration_s": 2.0,
                "tags": {},
                "note": None,
            }
        ]
        table = format_history_table(summaries)
        assert "Note" in table


class TestYAMLConfigNote:
    """Test that note can be set via YAML config."""

    def test_yaml_note_known_key(self):
        from xpyd_bench.config_cmd import _KNOWN_KEYS

        assert "note" in _KNOWN_KEYS

    def test_yaml_config_sets_note(self):
        import argparse

        from xpyd_bench.cli import _load_yaml_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("note: yaml note value\n")
            f.flush()
            args = argparse.Namespace(note=None)
            result = _load_yaml_config(f.name, args)
            assert result.note == "yaml note value"

    def test_cli_note_overrides_yaml(self):
        import argparse

        from xpyd_bench.cli import _load_yaml_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("note: yaml note\n")
            f.flush()
            args = argparse.Namespace(note="cli note")
            result = _load_yaml_config(f.name, args, explicit_keys={"note"})
            assert result.note == "cli note"


class TestNoteInJSONOutput:
    """Test that note appears in serialized JSON result dict."""

    def test_note_in_result_dict(self):
        import dataclasses

        br = BenchmarkResult(note="test note")
        d = dataclasses.asdict(br)
        assert d["note"] == "test note"

    def test_note_none_in_result_dict(self):
        import dataclasses

        br = BenchmarkResult()
        d = dataclasses.asdict(br)
        assert d["note"] is None

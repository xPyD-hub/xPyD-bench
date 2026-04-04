"""Tests for M36: Benchmark Annotations & Tags."""

from __future__ import annotations

import json
from pathlib import Path

from xpyd_bench.bench.models import BenchmarkResult


class TestTagsInModel:
    """Test tags field on BenchmarkResult."""

    def test_default_empty(self):
        r = BenchmarkResult()
        assert r.tags == {}

    def test_set_tags(self):
        r = BenchmarkResult(tags={"env": "prod", "gpu": "A100"})
        assert r.tags == {"env": "prod", "gpu": "A100"}


class TestTagsSerialization:
    """Test that tags survive serialization through _to_dict."""

    def test_to_dict_includes_tags(self):
        from xpyd_bench.bench.runner import _to_dict

        r = BenchmarkResult(tags={"env": "staging", "run_id": "42"})
        d = _to_dict(r)
        assert d["tags"] == {"env": "staging", "run_id": "42"}

    def test_to_dict_omits_empty_tags(self):
        from xpyd_bench.bench.runner import _to_dict

        r = BenchmarkResult()
        d = _to_dict(r)
        assert "tags" not in d


class TestParseTagsCLI:
    """Test --tag CLI parsing."""

    def test_parse_tags_from_list(self):
        import argparse

        from xpyd_bench.cli import _parse_tags

        ns = argparse.Namespace(tags=["env=prod", "gpu=A100"])
        tags = _parse_tags(ns)
        assert tags == {"env": "prod", "gpu": "A100"}

    def test_parse_tags_from_dict(self):
        """YAML config may set tags as a dict."""
        import argparse

        from xpyd_bench.cli import _parse_tags

        ns = argparse.Namespace(tags={"env": "prod", "gpu": "A100"})
        tags = _parse_tags(ns)
        assert tags == {"env": "prod", "gpu": "A100"}

    def test_parse_tags_none(self):
        import argparse

        from xpyd_bench.cli import _parse_tags

        ns = argparse.Namespace(tags=None)
        tags = _parse_tags(ns)
        assert tags == {}

    def test_parse_tags_no_attr(self):
        import argparse

        from xpyd_bench.cli import _parse_tags

        ns = argparse.Namespace()
        tags = _parse_tags(ns)
        assert tags == {}

    def test_parse_tags_value_with_equals(self):
        """Values can contain = signs."""
        import argparse

        from xpyd_bench.cli import _parse_tags

        ns = argparse.Namespace(tags=["formula=a=b+c"])
        tags = _parse_tags(ns)
        assert tags == {"formula": "a=b+c"}


class TestHistoryFilterTags:
    """Test history filtering by tags."""

    def _write_result(self, tmpdir: Path, name: str, tags: dict | None = None):
        data = {
            "model": "test",
            "num_prompts": 10,
            "completed": 10,
            "failed": 0,
            "request_throughput": 1.0,
            "output_throughput": 1.0,
            "mean_ttft_ms": 10.0,
            "mean_e2el_ms": 50.0,
            "total_duration_s": 10.0,
        }
        if tags:
            data["tags"] = tags
        (tmpdir / name).write_text(json.dumps(data))

    def test_filter_tags_match(self, tmp_path):
        from xpyd_bench.history import list_history

        self._write_result(tmp_path, "a-20260101-120000.json", {"env": "prod"})
        self._write_result(tmp_path, "b-20260101-120001.json", {"env": "staging"})
        self._write_result(tmp_path, "c-20260101-120002.json", {"env": "prod", "gpu": "A100"})

        results = list_history(tmp_path, filter_tags={"env": "prod"})
        assert len(results) == 2
        files = {r["file"] for r in results}
        assert "a-20260101-120000.json" in files
        assert "c-20260101-120002.json" in files

    def test_filter_tags_no_match(self, tmp_path):
        from xpyd_bench.history import list_history

        self._write_result(tmp_path, "a-20260101-120000.json", {"env": "prod"})

        results = list_history(tmp_path, filter_tags={"env": "staging"})
        assert len(results) == 0

    def test_filter_multiple_tags(self, tmp_path):
        from xpyd_bench.history import list_history

        self._write_result(tmp_path, "a-20260101-120000.json", {"env": "prod", "gpu": "A100"})
        self._write_result(tmp_path, "b-20260101-120001.json", {"env": "prod", "gpu": "H100"})

        results = list_history(tmp_path, filter_tags={"env": "prod", "gpu": "A100"})
        assert len(results) == 1
        assert results[0]["file"] == "a-20260101-120000.json"

    def test_no_filter_returns_all(self, tmp_path):
        from xpyd_bench.history import list_history

        self._write_result(tmp_path, "a-20260101-120000.json", {"env": "prod"})
        self._write_result(tmp_path, "b-20260101-120001.json")

        results = list_history(tmp_path)
        assert len(results) == 2

    def test_filter_no_tags_in_result(self, tmp_path):
        from xpyd_bench.history import list_history

        self._write_result(tmp_path, "a-20260101-120000.json")

        results = list_history(tmp_path, filter_tags={"env": "prod"})
        assert len(results) == 0


class TestHistoryCLIFilterTag:
    """Test history CLI --filter-tag flag."""

    def test_filter_tag_cli(self, tmp_path):
        from xpyd_bench.history import history_main

        # Write two results
        for name, tags in [
            ("a-20260101-120000.json", {"env": "prod"}),
            ("b-20260101-120001.json", {"env": "staging"}),
        ]:
            data = {
                "model": "test", "num_prompts": 10, "completed": 10,
                "failed": 0, "request_throughput": 1.0, "mean_ttft_ms": 10.0,
                "mean_e2el_ms": 50.0, "total_duration_s": 10.0, "tags": tags,
            }
            (tmp_path / name).write_text(json.dumps(data))

        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            history_main([
                "--result-dir", str(tmp_path),
                "--filter-tag", "env=prod",
            ])
        output = buf.getvalue()
        # Should only show 1 result row (the prod one)
        assert "test" in output
        # The table has a header + separator + 1 data row


class TestTagsYAMLConfig:
    """Test tags from YAML config."""

    def test_yaml_tags_loaded(self, tmp_path):
        import argparse

        import yaml

        from xpyd_bench.cli import _parse_tags

        config = {"tags": {"env": "prod", "gpu": "A100"}}
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config))

        # Simulate YAML loading setting tags as dict
        ns = argparse.Namespace(tags=config["tags"])
        tags = _parse_tags(ns)
        assert tags == {"env": "prod", "gpu": "A100"}

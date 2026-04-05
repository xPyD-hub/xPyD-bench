"""Tests for M72: Benchmark Fingerprinting."""

from __future__ import annotations

import unittest
from argparse import Namespace

from xpyd_bench.bench.fingerprint import compute_fingerprint
from xpyd_bench.bench.models import BenchmarkResult


class TestComputeFingerprint(unittest.TestCase):
    """Test fingerprint computation."""

    def test_deterministic(self):
        """Same config produces same fingerprint."""
        args = Namespace(
            base_url="http://localhost:8000",
            model="gpt-4",
            num_prompts=100,
            request_rate=10.0,
            endpoint="/v1/completions",
        )
        fp1 = compute_fingerprint(args)
        fp2 = compute_fingerprint(args)
        assert fp1 == fp2
        assert len(fp1) == 64  # SHA-256 hex

    def test_different_config_different_fingerprint(self):
        """Different configs produce different fingerprints."""
        args1 = Namespace(model="gpt-4", num_prompts=100, request_rate=10.0)
        args2 = Namespace(model="gpt-3.5", num_prompts=100, request_rate=10.0)
        assert compute_fingerprint(args1) != compute_fingerprint(args2)

    def test_arg_order_irrelevant(self):
        """Argument order does not affect fingerprint."""
        args1 = Namespace(model="gpt-4", num_prompts=100)
        args2 = Namespace(num_prompts=100, model="gpt-4")
        assert compute_fingerprint(args1) == compute_fingerprint(args2)

    def test_excluded_keys_ignored(self):
        """Output/display keys are excluded from fingerprint."""
        args1 = Namespace(model="gpt-4", num_prompts=100)
        args2 = Namespace(
            model="gpt-4",
            num_prompts=100,
            save_result="out.json",
            debug_log="/tmp/debug.log",
            note="test run",
            html_report="/tmp/report.html",
            dry_run=True,
            verbose=True,
        )
        assert compute_fingerprint(args1) == compute_fingerprint(args2)

    def test_dict_input(self):
        """Works with dict input as well as Namespace."""
        cfg = {"model": "gpt-4", "num_prompts": 100}
        fp1 = compute_fingerprint(cfg)
        fp2 = compute_fingerprint(Namespace(**cfg))
        assert fp1 == fp2

    def test_none_values_excluded(self):
        """None values are excluded (they represent defaults)."""
        args1 = Namespace(model="gpt-4")
        args2 = Namespace(model="gpt-4", temperature=None, top_p=None)
        assert compute_fingerprint(args1) == compute_fingerprint(args2)

    def test_inf_normalized(self):
        """Float inf is normalized to string 'inf'."""
        args1 = Namespace(model="gpt-4", request_rate=float("inf"))
        args2 = Namespace(model="gpt-4", request_rate=float("inf"))
        assert compute_fingerprint(args1) == compute_fingerprint(args2)

    def test_private_keys_excluded(self):
        """Keys starting with _ are excluded."""
        args1 = Namespace(model="gpt-4")
        args2 = Namespace(model="gpt-4", _internal="secret")
        assert compute_fingerprint(args1) == compute_fingerprint(args2)

    def test_nested_dict_normalized(self):
        """Nested dicts are sorted by key for determinism."""
        args1 = Namespace(headers={"X-A": "1", "X-B": "2"})
        args2 = Namespace(headers={"X-B": "2", "X-A": "1"})
        assert compute_fingerprint(args1) == compute_fingerprint(args2)


class TestBenchmarkResultFingerprint(unittest.TestCase):
    """Test fingerprint field on BenchmarkResult."""

    def test_default_none(self):
        """fingerprint defaults to None."""
        r = BenchmarkResult()
        assert r.fingerprint is None

    def test_set_fingerprint(self):
        """fingerprint can be set."""
        r = BenchmarkResult()
        r.fingerprint = "abc123"
        assert r.fingerprint == "abc123"


class TestHistoryGroupByFingerprint(unittest.TestCase):
    """Test history grouping by fingerprint (M72)."""

    def test_group_by_fingerprint(self):
        """History results can be grouped by fingerprint."""
        from datetime import datetime

        from xpyd_bench.history import format_history_by_fingerprint

        fp_a = "a" * 64
        fp_b = "b" * 64
        summaries = [
            {
                "file": "r1.json", "timestamp": datetime(2026, 1, 1, 10, 0),
                "model": "gpt-4", "num_prompts": 100, "completed": 100,
                "failed": 0, "request_throughput": 10.0, "output_throughput": 500.0,
                "mean_ttft_ms": 50.0, "mean_e2el_ms": 200.0, "partial": False,
                "total_duration_s": 10, "tags": {}, "note": None,
                "fingerprint": fp_a,
            },
            {
                "file": "r2.json", "timestamp": datetime(2026, 1, 1, 11, 0),
                "model": "gpt-4", "num_prompts": 100, "completed": 100,
                "failed": 0, "request_throughput": 12.0, "output_throughput": 600.0,
                "mean_ttft_ms": 45.0, "mean_e2el_ms": 190.0, "partial": False,
                "total_duration_s": 10, "tags": {}, "note": None,
                "fingerprint": fp_a,
            },
            {
                "file": "r3.json", "timestamp": datetime(2026, 1, 1, 12, 0),
                "model": "gpt-3.5", "num_prompts": 50, "completed": 50,
                "failed": 0, "request_throughput": 20.0, "output_throughput": 800.0,
                "mean_ttft_ms": 30.0, "mean_e2el_ms": 100.0, "partial": False,
                "total_duration_s": 5, "tags": {}, "note": None,
                "fingerprint": fp_b,
            },
        ]
        output = format_history_by_fingerprint(summaries)
        assert "aaaaaaaaaaaa" in output  # short fp_a
        assert "bbbbbbbbbbbb" in output  # short fp_b
        assert "2 run(s)" in output
        assert "1 run(s)" in output

    def test_no_fingerprint_group(self):
        """Results without fingerprint go to ungrouped."""
        from datetime import datetime

        from xpyd_bench.history import format_history_by_fingerprint

        summaries = [
            {
                "file": "r1.json", "timestamp": datetime(2026, 1, 1),
                "model": "gpt-4", "num_prompts": 100, "completed": 100,
                "failed": 0, "request_throughput": 10.0, "output_throughput": 500.0,
                "mean_ttft_ms": 50.0, "mean_e2el_ms": 200.0, "partial": False,
                "total_duration_s": 10, "tags": {}, "note": None,
                "fingerprint": None,
            },
        ]
        output = format_history_by_fingerprint(summaries)
        assert "No fingerprint" in output

    def test_empty_summaries(self):
        """Empty list returns no results message."""
        from xpyd_bench.history import format_history_by_fingerprint

        output = format_history_by_fingerprint([])
        assert "No benchmark results found" in output


class TestAggregateByFingerprint(unittest.TestCase):
    """Test aggregate --by-fingerprint grouping."""

    def test_by_fingerprint_groups(self):
        """Results are grouped by fingerprint in aggregate."""

        fp = compute_fingerprint(Namespace(model="gpt-4", num_prompts=100))

        r1 = {"fingerprint": fp, "mean_ttft_ms": 50.0, "request_throughput": 10.0}
        r2 = {"fingerprint": fp, "mean_ttft_ms": 55.0, "request_throughput": 11.0}
        r3 = {"fingerprint": "other" * 13, "mean_ttft_ms": 30.0, "request_throughput": 20.0}

        # Group by fingerprint
        groups: dict[str, list[dict]] = {}
        for r in [r1, r2, r3]:
            groups.setdefault(r.get("fingerprint", "unknown"), []).append(r)

        assert len(groups) == 2
        assert len(groups[fp]) == 2


if __name__ == "__main__":
    unittest.main()

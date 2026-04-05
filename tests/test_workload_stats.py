"""Tests for workload distribution statistics (M78)."""

from __future__ import annotations

import unittest

from xpyd_bench.bench.workload_stats import compute_workload_stats


class TestComputeWorkloadStats(unittest.TestCase):
    """Test compute_workload_stats function."""

    def test_basic_stats(self):
        prompt = [100, 200, 300, 400, 500]
        completion = [10, 20, 30, 40, 50]
        result = compute_workload_stats(prompt, completion)

        self.assertIn("prompt", result)
        self.assertIn("completion", result)

        p = result["prompt"]
        self.assertEqual(p["count"], 5)
        self.assertEqual(p["mean"], 300.0)
        self.assertEqual(p["min"], 100)
        self.assertEqual(p["max"], 500)
        self.assertGreater(p["std"], 0)
        self.assertAlmostEqual(p["p50"], 300.0, places=0)

        c = result["completion"]
        self.assertEqual(c["count"], 5)
        self.assertEqual(c["mean"], 30.0)
        self.assertEqual(c["min"], 10)
        self.assertEqual(c["max"], 50)

    def test_empty_lists(self):
        result = compute_workload_stats([], [])
        self.assertEqual(result, {})

    def test_single_request(self):
        result = compute_workload_stats([42], [7])
        self.assertEqual(result["prompt"]["count"], 1)
        self.assertEqual(result["prompt"]["mean"], 42.0)
        self.assertEqual(result["prompt"]["std"], 0.0)
        self.assertEqual(result["prompt"]["min"], 42)
        self.assertEqual(result["prompt"]["max"], 42)
        self.assertEqual(result["prompt"]["p50"], 42.0)
        self.assertEqual(result["prompt"]["p90"], 42.0)
        self.assertEqual(result["prompt"]["p99"], 42.0)

    def test_one_empty_one_populated(self):
        result = compute_workload_stats([], [10, 20])
        self.assertIn("prompt", result)
        self.assertIn("completion", result)
        self.assertEqual(result["prompt"]["count"], 0)
        self.assertEqual(result["completion"]["count"], 2)

    def test_identical_values(self):
        result = compute_workload_stats([50, 50, 50], [10, 10, 10])
        self.assertEqual(result["prompt"]["std"], 0.0)
        self.assertEqual(result["prompt"]["mean"], 50.0)


class TestWorkloadStatsCLI(unittest.TestCase):
    """Test --workload-stats CLI flag parsing."""

    def _make_parser(self):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        return parser

    def test_cli_flag_parsed(self):
        parser = self._make_parser()
        args = parser.parse_args(["--workload-stats"])
        self.assertTrue(args.workload_stats)

    def test_cli_flag_default_false(self):
        parser = self._make_parser()
        args = parser.parse_args([])
        self.assertFalse(args.workload_stats)


class TestWorkloadStatsInKnownKeys(unittest.TestCase):
    """Test that workload_stats is a known YAML config key."""

    def test_workload_stats_in_known_keys(self):
        from xpyd_bench.config_cmd import _KNOWN_KEYS

        self.assertIn("workload_stats", _KNOWN_KEYS)


class TestWorkloadStatsInBenchmarkResult(unittest.TestCase):
    """Test that BenchmarkResult has workload_stats field."""

    def test_field_exists(self):
        from xpyd_bench.bench.models import BenchmarkResult

        r = BenchmarkResult()
        self.assertIsNone(r.workload_stats)

    def test_field_serialization(self):
        import dataclasses

        from xpyd_bench.bench.models import BenchmarkResult

        r = BenchmarkResult()
        stats = compute_workload_stats([100, 200], [10, 20])
        r.workload_stats = stats
        d = dataclasses.asdict(r)
        self.assertIn("workload_stats", d)
        self.assertIn("prompt", d["workload_stats"])
        self.assertIn("completion", d["workload_stats"])


if __name__ == "__main__":
    unittest.main()

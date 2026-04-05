"""Tests for heatmap data export (M97)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpyd_bench.bench.heatmap_export import (
    DEFAULT_BIN_EDGES_MS,
    DEFAULT_BUCKET_WIDTH_S,
    HeatmapBucket,
    HeatmapExportData,
    _assign_bin,
    compute_heatmap_export,
    export_heatmap_json,
    parse_bin_edges,
)
from xpyd_bench.bench.models import RequestResult

# ── Helpers ────────────────────────────────────────────────────────────


def _make_request(start_time: float, latency_ms: float) -> RequestResult:
    return RequestResult(
        start_time=start_time,
        latency_ms=latency_ms,
        success=True,
    )


# ── _assign_bin ────────────────────────────────────────────────────────


class TestAssignBin:
    def test_below_first_edge(self):
        assert _assign_bin(-1.0, [0, 50, 100]) == 0

    def test_at_first_edge(self):
        assert _assign_bin(0.0, [0, 50, 100]) == 0

    def test_between_edges(self):
        assert _assign_bin(75.0, [0, 50, 100]) == 1

    def test_at_last_edge(self):
        assert _assign_bin(100.0, [0, 50, 100]) == 2

    def test_above_last_edge(self):
        assert _assign_bin(5000.0, [0, 50, 100]) == 2

    def test_default_bins(self):
        # 0-50ms -> bin 0, 50-100 -> bin 1, ... 1000+ -> bin 5
        assert _assign_bin(25.0, DEFAULT_BIN_EDGES_MS) == 0
        assert _assign_bin(75.0, DEFAULT_BIN_EDGES_MS) == 1
        assert _assign_bin(150.0, DEFAULT_BIN_EDGES_MS) == 2
        assert _assign_bin(300.0, DEFAULT_BIN_EDGES_MS) == 3
        assert _assign_bin(750.0, DEFAULT_BIN_EDGES_MS) == 4
        assert _assign_bin(2000.0, DEFAULT_BIN_EDGES_MS) == 5


# ── compute_heatmap_export ─────────────────────────────────────────────


class TestComputeHeatmapExport:
    def test_empty_requests(self):
        result = compute_heatmap_export([], bench_start_time=0.0)
        assert result.total_requests == 0
        assert result.buckets == []

    def test_single_request(self):
        reqs = [_make_request(start_time=10.0, latency_ms=75.0)]
        result = compute_heatmap_export(reqs, bench_start_time=10.0)
        assert result.total_requests == 1
        assert len(result.buckets) == 1
        assert result.buckets[0].total == 1
        # 75ms should be in bin index 1 (50-100ms range)
        assert result.buckets[0].counts[1] == 1

    def test_multiple_buckets(self):
        # Requests spread across 3 seconds
        reqs = [
            _make_request(start_time=100.0, latency_ms=30.0),   # t=0s, bin 0
            _make_request(start_time=100.5, latency_ms=80.0),   # t=0.5s, bin 1
            _make_request(start_time=101.5, latency_ms=150.0),  # t=1.5s, bin 2
            _make_request(start_time=102.5, latency_ms=600.0),  # t=2.5s, bin 4
        ]
        result = compute_heatmap_export(reqs, bench_start_time=100.0, bucket_width_s=1.0)
        assert result.total_requests == 4
        assert len(result.buckets) == 3
        # Bucket 0: [0s, 1s) -> 2 requests
        assert result.buckets[0].total == 2
        assert result.buckets[0].counts[0] == 1  # 30ms
        assert result.buckets[0].counts[1] == 1  # 80ms
        # Bucket 1: [1s, 2s) -> 1 request
        assert result.buckets[1].total == 1
        assert result.buckets[1].counts[2] == 1  # 150ms
        # Bucket 2: [2s, 3s) -> 1 request
        assert result.buckets[2].total == 1
        assert result.buckets[2].counts[4] == 1  # 600ms

    def test_custom_bin_edges(self):
        reqs = [_make_request(start_time=0.0, latency_ms=250.0)]
        result = compute_heatmap_export(
            reqs, bench_start_time=0.0, bin_edges_ms=[0, 100, 500]
        )
        assert result.bin_edges_ms == [0, 100, 500]
        # 250ms -> bin 1 (100-500 range)
        assert result.buckets[0].counts[1] == 1

    def test_custom_bucket_width(self):
        reqs = [
            _make_request(start_time=0.0, latency_ms=50.0),
            _make_request(start_time=2.5, latency_ms=50.0),
            _make_request(start_time=5.5, latency_ms=50.0),
        ]
        result = compute_heatmap_export(reqs, bench_start_time=0.0, bucket_width_s=3.0)
        assert result.bucket_width_s == 3.0
        assert len(result.buckets) == 2  # [0,3) and [3,6)
        assert result.buckets[0].total == 2
        assert result.buckets[1].total == 1

    def test_failed_requests_excluded(self):
        reqs = [
            _make_request(start_time=0.0, latency_ms=50.0),
            RequestResult(start_time=0.5, latency_ms=100.0, success=False),
        ]
        result = compute_heatmap_export(reqs, bench_start_time=0.0)
        assert result.total_requests == 1

    def test_no_start_time_excluded(self):
        reqs = [
            _make_request(start_time=0.0, latency_ms=50.0),
            RequestResult(start_time=None, latency_ms=100.0, success=True),
        ]
        result = compute_heatmap_export(reqs, bench_start_time=0.0)
        assert result.total_requests == 1

    def test_invalid_bucket_width_uses_default(self):
        reqs = [_make_request(start_time=0.0, latency_ms=50.0)]
        result = compute_heatmap_export(reqs, bench_start_time=0.0, bucket_width_s=-1.0)
        assert result.bucket_width_s == DEFAULT_BUCKET_WIDTH_S

    def test_overflow_bin(self):
        """Latency above last bin edge goes to overflow bin."""
        reqs = [_make_request(start_time=0.0, latency_ms=5000.0)]
        result = compute_heatmap_export(reqs, bench_start_time=0.0)
        # Last bin index = len(DEFAULT_BIN_EDGES_MS) = 5 (overflow)
        last_bin = len(DEFAULT_BIN_EDGES_MS)
        assert result.buckets[0].counts[last_bin] == 0
        # Actually 5000ms >= 1000ms so bin index 5 (last edge index)
        assert result.buckets[0].counts[5] == 1


# ── export_heatmap_json ────────────────────────────────────────────────


class TestExportHeatmapJson:
    def test_export_creates_file(self, tmp_path: Path):
        data = HeatmapExportData(
            bucket_width_s=1.0,
            bin_edges_ms=[0, 50, 100],
            buckets=[
                HeatmapBucket(time_start=0.0, time_end=1.0, counts=[2, 1, 0, 0], total=3),
            ],
            total_requests=3,
        )
        out = tmp_path / "heatmap.json"
        result_path = export_heatmap_json(data, out)
        assert result_path.exists()
        loaded = json.loads(result_path.read_text())
        assert loaded["bucket_width_s"] == 1.0
        assert loaded["bin_edges_ms"] == [0, 50, 100]
        assert len(loaded["buckets"]) == 1
        assert loaded["buckets"][0]["total"] == 3
        assert loaded["total_requests"] == 3

    def test_export_creates_parent_dirs(self, tmp_path: Path):
        out = tmp_path / "deep" / "nested" / "heatmap.json"
        data = HeatmapExportData()
        export_heatmap_json(data, out)
        assert out.exists()

    def test_roundtrip_serialization(self, tmp_path: Path):
        reqs = [
            _make_request(start_time=5.0, latency_ms=30.0),
            _make_request(start_time=5.3, latency_ms=120.0),
            _make_request(start_time=6.2, latency_ms=800.0),
        ]
        data = compute_heatmap_export(reqs, bench_start_time=5.0)
        out = tmp_path / "rt.json"
        export_heatmap_json(data, out)
        loaded = json.loads(out.read_text())
        assert loaded["total_requests"] == 3
        total_in_buckets = sum(b["total"] for b in loaded["buckets"])
        assert total_in_buckets == 3


# ── parse_bin_edges ────────────────────────────────────────────────────


class TestParseBinEdges:
    def test_basic(self):
        assert parse_bin_edges("0,50,100,200") == [0.0, 50.0, 100.0, 200.0]

    def test_sorts_edges(self):
        assert parse_bin_edges("100,0,50") == [0.0, 50.0, 100.0]

    def test_with_spaces(self):
        assert parse_bin_edges(" 0 , 50 , 100 ") == [0.0, 50.0, 100.0]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            parse_bin_edges("")

    def test_invalid_number_raises(self):
        with pytest.raises(ValueError):
            parse_bin_edges("0,abc,100")

    def test_float_edges(self):
        assert parse_bin_edges("0,25.5,75.5") == [0.0, 25.5, 75.5]


# ── to_dict serialization ─────────────────────────────────────────────


class TestSerialization:
    def test_bucket_to_dict(self):
        b = HeatmapBucket(time_start=0.0, time_end=1.0, counts=[1, 2, 3], total=6)
        d = b.to_dict()
        assert d == {"time_start": 0.0, "time_end": 1.0, "counts": [1, 2, 3], "total": 6}

    def test_export_data_to_dict(self):
        data = HeatmapExportData(
            bucket_width_s=2.0,
            bin_edges_ms=[0, 100],
            buckets=[
                HeatmapBucket(time_start=0.0, time_end=2.0, counts=[5, 3, 1], total=9),
            ],
            total_requests=9,
        )
        d = data.to_dict()
        assert d["bucket_width_s"] == 2.0
        assert d["total_requests"] == 9
        assert len(d["buckets"]) == 1


# ── CLI arg registration ──────────────────────────────────────────────


class TestCLIArgs:
    def _make_parser(self):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args
        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        return parser

    def test_heatmap_export_arg_registered(self):
        parser = self._make_parser()
        args = parser.parse_args([
            "--base-url", "http://localhost:8000",
            "--heatmap-export", "/tmp/out.json",
        ])
        assert args.heatmap_export == "/tmp/out.json"

    def test_heatmap_bucket_width_arg(self):
        parser = self._make_parser()
        args = parser.parse_args([
            "--base-url", "http://localhost:8000",
            "--heatmap-bucket-width", "2.5",
        ])
        assert args.heatmap_bucket_width == 2.5

    def test_heatmap_bins_arg(self):
        parser = self._make_parser()
        args = parser.parse_args([
            "--base-url", "http://localhost:8000",
            "--heatmap-bins", "0,25,50,100",
        ])
        assert args.heatmap_bins == "0,25,50,100"


# ── Config key registration ───────────────────────────────────────────


class TestConfigKeys:
    def test_heatmap_keys_in_known(self):
        from xpyd_bench.config_cmd import _KNOWN_KEYS

        assert "heatmap_export" in _KNOWN_KEYS
        assert "heatmap_bucket_width" in _KNOWN_KEYS
        assert "heatmap_bins" in _KNOWN_KEYS

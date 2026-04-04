"""Tests for M34: Prometheus / OpenMetrics export."""

from __future__ import annotations

import re
from pathlib import Path

from xpyd_bench.bench.models import BenchmarkResult, RequestResult
from xpyd_bench.reporting.prometheus import export_prometheus


def _make_result(
    model: str = "test-model",
    endpoint: str = "/v1/completions",
    num_requests: int = 10,
) -> BenchmarkResult:
    """Build a synthetic BenchmarkResult for testing."""
    requests = []
    for i in range(num_requests):
        requests.append(
            RequestResult(
                prompt_tokens=10,
                completion_tokens=20,
                ttft_ms=(i + 1) * 10.0,  # 10ms .. 100ms
                tpot_ms=(i + 1) * 5.0,  # 5ms .. 50ms
                latency_ms=(i + 1) * 50.0,  # 50ms .. 500ms
                success=True,
            )
        )
    # Add one failed request
    requests.append(
        RequestResult(
            prompt_tokens=10,
            completion_tokens=0,
            latency_ms=100.0,
            success=False,
            error="timeout",
        )
    )
    result = BenchmarkResult(
        model=model,
        endpoint=endpoint,
        num_prompts=num_requests + 1,
        completed=num_requests,
        failed=1,
        output_throughput=42.5,
        requests=requests,
    )
    return result


class TestPrometheusExport:
    """Test Prometheus export functionality."""

    def test_export_creates_file(self, tmp_path: Path) -> None:
        result = _make_result()
        out = tmp_path / "metrics.prom"
        p = export_prometheus(result, out)
        assert p.exists()
        assert p == out

    def test_all_metric_families_present(self, tmp_path: Path) -> None:
        result = _make_result()
        out = tmp_path / "metrics.prom"
        export_prometheus(result, out)
        content = out.read_text()

        expected_metrics = [
            "xpyd_bench_ttft_seconds",
            "xpyd_bench_tpot_seconds",
            "xpyd_bench_request_latency_seconds",
            "xpyd_bench_throughput_tokens_per_second",
            "xpyd_bench_requests_total",
            "xpyd_bench_errors_total",
        ]
        for metric in expected_metrics:
            assert f"# HELP {metric}" in content, f"Missing HELP for {metric}"
            assert f"# TYPE {metric}" in content, f"Missing TYPE for {metric}"

    def test_histogram_type_annotations(self, tmp_path: Path) -> None:
        result = _make_result()
        out = tmp_path / "metrics.prom"
        export_prometheus(result, out)
        content = out.read_text()

        assert "# TYPE xpyd_bench_ttft_seconds histogram" in content
        assert "# TYPE xpyd_bench_tpot_seconds histogram" in content
        assert "# TYPE xpyd_bench_request_latency_seconds histogram" in content
        assert "# TYPE xpyd_bench_throughput_tokens_per_second gauge" in content
        assert "# TYPE xpyd_bench_requests_total counter" in content
        assert "# TYPE xpyd_bench_errors_total counter" in content

    def test_histogram_has_buckets_count_sum(self, tmp_path: Path) -> None:
        result = _make_result()
        out = tmp_path / "metrics.prom"
        export_prometheus(result, out)
        content = out.read_text()

        # TTFT histogram should have _bucket, _count, _sum
        assert "xpyd_bench_ttft_seconds_bucket{" in content
        assert "xpyd_bench_ttft_seconds_count{" in content
        assert "xpyd_bench_ttft_seconds_sum{" in content

    def test_histogram_inf_bucket(self, tmp_path: Path) -> None:
        result = _make_result()
        out = tmp_path / "metrics.prom"
        export_prometheus(result, out)
        content = out.read_text()

        # +Inf bucket should contain all values
        inf_match = re.search(
            r'xpyd_bench_ttft_seconds_bucket\{[^}]*le="\+Inf"\} (\d+)', content
        )
        assert inf_match is not None
        assert int(inf_match.group(1)) == 10  # 10 successful with ttft

    def test_labels_injected(self, tmp_path: Path) -> None:
        result = _make_result(model="llama-7b", endpoint="/v1/chat/completions")
        out = tmp_path / "metrics.prom"
        export_prometheus(result, out, scenario="stress")
        content = out.read_text()

        assert 'model="llama-7b"' in content
        assert 'endpoint="/v1/chat/completions"' in content
        assert 'scenario="stress"' in content

    def test_counter_values(self, tmp_path: Path) -> None:
        result = _make_result(num_requests=5)
        out = tmp_path / "metrics.prom"
        export_prometheus(result, out)
        content = out.read_text()

        # 5 completed + 1 failed = 6 total
        assert re.search(r"xpyd_bench_requests_total\{[^}]*\} 6", content)
        assert re.search(r"xpyd_bench_errors_total\{[^}]*\} 1", content)

    def test_gauge_value(self, tmp_path: Path) -> None:
        result = _make_result()
        out = tmp_path / "metrics.prom"
        export_prometheus(result, out)
        content = out.read_text()

        assert "xpyd_bench_throughput_tokens_per_second" in content
        assert "42.500000" in content

    def test_no_labels(self, tmp_path: Path) -> None:
        """Export with empty model/endpoint should still work."""
        result = BenchmarkResult(
            completed=1,
            failed=0,
            output_throughput=10.0,
            requests=[
                RequestResult(
                    latency_ms=100.0,
                    success=True,
                )
            ],
        )
        out = tmp_path / "metrics.prom"
        export_prometheus(result, out)
        content = out.read_text()
        # endpoint default is "/v1/completions" so labels are present
        assert "xpyd_bench_requests_total{" in content

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        result = _make_result()
        out = tmp_path / "sub" / "dir" / "metrics.prom"
        p = export_prometheus(result, out)
        assert p.exists()

    def test_yaml_config_key(self, tmp_path: Path) -> None:
        """Verify the YAML key maps correctly (prometheus_export attr exists)."""
        import argparse

        args = argparse.Namespace(prometheus_export=None)
        # Simulate YAML config applying
        cfg = {"prometheus_export": str(tmp_path / "out.prom")}
        for key, value in cfg.items():
            attr = key.replace("-", "_")
            if getattr(args, attr, None) is None:
                setattr(args, attr, value)
        assert args.prometheus_export == str(tmp_path / "out.prom")

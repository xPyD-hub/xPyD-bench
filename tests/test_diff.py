"""Tests for M54: Benchmark Diff Report."""

from __future__ import annotations

import json
import subprocess

import pytest

from xpyd_bench.diff import (
    _mann_whitney_u,
    diff,
    diff_main,
    format_diff_table,
    generate_html_diff,
    generate_markdown_diff,
)

# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

def _make_result(
    latency: float = 100.0,
    ttft: float = 20.0,
    throughput: float = 50.0,
    n_requests: int = 30,
    jitter: float = 5.0,
) -> dict:
    """Build a synthetic benchmark result dict."""
    import random

    rng = random.Random(42)
    requests = []
    for i in range(n_requests):
        lat = latency + rng.gauss(0, jitter)
        t = ttft + rng.gauss(0, jitter * 0.3)
        requests.append({"latency_ms": lat, "ttft_ms": t, "success": True})

    return {
        "model": "test-model",
        "num_prompts": n_requests,
        "summary": {
            "mean_ttft_ms": ttft,
            "p50_ttft_ms": ttft * 0.98,
            "p90_ttft_ms": ttft * 1.3,
            "p95_ttft_ms": ttft * 1.5,
            "p99_ttft_ms": ttft * 2.0,
            "mean_e2el_ms": latency,
            "p50_e2el_ms": latency * 0.98,
            "p90_e2el_ms": latency * 1.3,
            "p95_e2el_ms": latency * 1.5,
            "p99_e2el_ms": latency * 2.0,
            "mean_tpot_ms": latency / 10,
            "p50_tpot_ms": latency / 10 * 0.98,
            "p90_tpot_ms": latency / 10 * 1.3,
            "p95_tpot_ms": latency / 10 * 1.5,
            "p99_tpot_ms": latency / 10 * 2.0,
            "request_throughput": throughput,
            "output_throughput": throughput * 100,
            "total_token_throughput": throughput * 150,
        },
        "requests": requests,
    }


# ---------------------------------------------------------------------------
# Mann-Whitney U tests
# ---------------------------------------------------------------------------

class TestMannWhitneyU:
    def test_identical_samples(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        _, p = _mann_whitney_u(x, x[:])
        assert p > 0.05  # not significant

    def test_clearly_different(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0] * 10
        y = [100.0, 200.0, 300.0, 400.0, 500.0] * 10
        _, p = _mann_whitney_u(x, y)
        assert p < 0.05  # significant

    def test_small_samples(self):
        _, p = _mann_whitney_u([1.0, 2.0], [3.0, 4.0])
        assert p == 1.0  # too small, returns not significant


# ---------------------------------------------------------------------------
# Diff logic tests
# ---------------------------------------------------------------------------

class TestDiff:
    def test_no_regression(self):
        base = _make_result(latency=100, throughput=50)
        cand = _make_result(latency=100, throughput=50)
        result = diff(base, cand)
        assert not result.has_significant_regression
        assert len(result.metrics) > 0

    def test_regression_detected(self):
        base = _make_result(latency=100, throughput=50, n_requests=50, jitter=2)
        cand = _make_result(latency=200, throughput=25, n_requests=50, jitter=2)
        result = diff(base, cand)
        assert result.has_significant_regression
        regressed = [m for m in result.metrics if m.direction == "regressed"]
        assert len(regressed) > 0

    def test_improvement_detected(self):
        base = _make_result(latency=200, throughput=25)
        cand = _make_result(latency=100, throughput=50)
        result = diff(base, cand)
        improved = [m for m in result.metrics if m.direction == "improved"]
        assert len(improved) > 0

    def test_to_dict(self):
        base = _make_result()
        cand = _make_result()
        result = diff(base, cand)
        d = result.to_dict()
        assert "metrics" in d
        assert "has_significant_regression" in d
        assert "alpha" in d

    def test_custom_alpha(self):
        base = _make_result()
        cand = _make_result()
        result = diff(base, cand, alpha=0.001)
        assert result.alpha == 0.001

    def test_no_per_request_data(self):
        """When no requests array, p_value should be None."""
        base = {"summary": {"mean_e2el_ms": 100, "request_throughput": 50}}
        cand = {"summary": {"mean_e2el_ms": 200, "request_throughput": 25}}
        result = diff(base, cand)
        for m in result.metrics:
            assert m.p_value is None


# ---------------------------------------------------------------------------
# Output format tests
# ---------------------------------------------------------------------------

class TestFormatting:
    def test_terminal_table(self):
        result = diff(_make_result(), _make_result())
        table = format_diff_table(result)
        assert "xPyD-bench" in table
        assert "Diff Report" in table

    def test_markdown_output(self):
        result = diff(_make_result(), _make_result(latency=200))
        md = generate_markdown_diff(result)
        assert "## xPyD-bench Diff Report" in md
        assert "| Metric |" in md
        assert "|--------|" in md

    def test_html_output(self):
        base = _make_result()
        cand = _make_result(latency=200)
        result = diff(base, cand)
        html = generate_html_diff(result, base, cand)
        assert "<!DOCTYPE html>" in html
        assert "xPyD-bench Diff Report" in html
        assert "<table>" in html
        # Should have distribution section since we have per-request data
        assert "Latency Distribution" in html

    def test_html_no_per_request(self):
        base = {"summary": {"mean_e2el_ms": 100}}
        cand = {"summary": {"mean_e2el_ms": 200}}
        result = diff(base, cand)
        html = generate_html_diff(result)
        assert "<!DOCTYPE html>" in html
        assert "Latency Distribution" not in html


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

class TestDiffCLI:
    def test_diff_cli_no_regression(self, tmp_path):
        base = _make_result()
        cand = _make_result()
        bp = tmp_path / "base.json"
        cp = tmp_path / "cand.json"
        bp.write_text(json.dumps(base))
        cp.write_text(json.dumps(cand))

        # Should exit 0 (no regression)
        diff_main([str(bp), str(cp)])

    def test_diff_cli_regression_exits_1(self, tmp_path):
        base = _make_result(latency=100, jitter=2, n_requests=50)
        cand = _make_result(latency=200, jitter=2, n_requests=50)
        bp = tmp_path / "base.json"
        cp = tmp_path / "cand.json"
        bp.write_text(json.dumps(base))
        cp.write_text(json.dumps(cand))

        with pytest.raises(SystemExit) as exc_info:
            diff_main([str(bp), str(cp)])
        assert exc_info.value.code == 1

    def test_diff_cli_html_output(self, tmp_path):
        base = _make_result()
        cand = _make_result()
        bp = tmp_path / "base.json"
        cp = tmp_path / "cand.json"
        hp = tmp_path / "diff.html"
        bp.write_text(json.dumps(base))
        cp.write_text(json.dumps(cand))

        diff_main([str(bp), str(cp), "--html-diff", str(hp)])
        assert hp.exists()
        content = hp.read_text()
        assert "<!DOCTYPE html>" in content

    def test_diff_cli_markdown_output(self, tmp_path):
        base = _make_result()
        cand = _make_result()
        bp = tmp_path / "base.json"
        cp = tmp_path / "cand.json"
        mp = tmp_path / "diff.md"
        bp.write_text(json.dumps(base))
        cp.write_text(json.dumps(cand))

        diff_main([str(bp), str(cp), "--markdown-diff", str(mp)])
        assert mp.exists()
        content = mp.read_text()
        assert "## xPyD-bench Diff Report" in content

    def test_diff_cli_json_output(self, tmp_path):
        base = _make_result()
        cand = _make_result()
        bp = tmp_path / "base.json"
        cp = tmp_path / "cand.json"
        jp = tmp_path / "diff.json"
        bp.write_text(json.dumps(base))
        cp.write_text(json.dumps(cand))

        diff_main([str(bp), str(cp), "--json-diff", str(jp)])
        assert jp.exists()
        data = json.loads(jp.read_text())
        assert "metrics" in data
        assert "has_significant_regression" in data

    def test_diff_subprocess(self, tmp_path):
        """Test via subprocess to verify CLI routing."""
        base = _make_result()
        cand = _make_result()
        bp = tmp_path / "base.json"
        cp = tmp_path / "cand.json"
        bp.write_text(json.dumps(base))
        cp.write_text(json.dumps(cand))

        result = subprocess.run(
            ["xpyd-bench", "diff", str(bp), str(cp)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Diff Report" in result.stdout

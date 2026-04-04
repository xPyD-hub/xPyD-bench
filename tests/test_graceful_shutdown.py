"""Tests for M12: Graceful Shutdown & Progress Persistence."""

from __future__ import annotations

from xpyd_bench.bench.models import BenchmarkResult, RequestResult
from xpyd_bench.bench.runner import _compute_metrics, _to_dict
from xpyd_bench.compare import compare, format_comparison_table

# ---------------------------------------------------------------------------
# BenchmarkResult.partial field
# ---------------------------------------------------------------------------


class TestPartialFlag:
    """Test that the partial flag is correctly set and serialized."""

    def test_default_not_partial(self) -> None:
        result = BenchmarkResult()
        assert result.partial is False

    def test_partial_flag_set(self) -> None:
        result = BenchmarkResult(partial=True)
        assert result.partial is True

    def test_to_dict_includes_partial_when_true(self) -> None:
        result = BenchmarkResult(partial=True)
        d = _to_dict(result)
        assert d["partial"] is True

    def test_to_dict_excludes_partial_when_false(self) -> None:
        result = BenchmarkResult(partial=False)
        d = _to_dict(result)
        assert "partial" not in d


# ---------------------------------------------------------------------------
# Partial results produce valid metrics
# ---------------------------------------------------------------------------


class TestPartialMetrics:
    """Verify that _compute_metrics works on partial (incomplete) results."""

    def test_metrics_from_subset(self) -> None:
        result = BenchmarkResult(partial=True)
        result.total_duration_s = 2.0
        result.requests = [
            RequestResult(
                prompt_tokens=10,
                completion_tokens=20,
                latency_ms=100.0,
                success=True,
            ),
            RequestResult(
                prompt_tokens=10,
                completion_tokens=20,
                latency_ms=150.0,
                success=True,
            ),
        ]
        _compute_metrics(result)
        assert result.completed == 2
        assert result.failed == 0
        assert result.request_throughput > 0
        assert result.total_input_tokens == 20
        assert result.total_output_tokens == 40

    def test_metrics_with_mixed_success(self) -> None:
        result = BenchmarkResult(partial=True)
        result.total_duration_s = 1.0
        result.requests = [
            RequestResult(latency_ms=50.0, success=True, completion_tokens=5, prompt_tokens=5),
            RequestResult(latency_ms=200.0, success=False, error="cancelled"),
        ]
        _compute_metrics(result)
        assert result.completed == 1
        assert result.failed == 1

    def test_empty_requests(self) -> None:
        result = BenchmarkResult(partial=True)
        result.total_duration_s = 0.5
        result.requests = []
        _compute_metrics(result)
        assert result.completed == 0
        assert result.failed == 0


# ---------------------------------------------------------------------------
# Compare warns on partial results
# ---------------------------------------------------------------------------


class TestComparePartialWarning:
    """Test that the compare tool detects and warns about partial results."""

    def _make_result(self, throughput: float = 100.0, partial: bool = False) -> dict:
        d = {
            "request_throughput": throughput,
            "output_throughput": throughput * 10,
            "total_token_throughput": throughput * 15,
            "mean_ttft_ms": 50.0,
            "p50_ttft_ms": 48.0,
            "p90_ttft_ms": 70.0,
            "p95_ttft_ms": 80.0,
            "p99_ttft_ms": 95.0,
            "mean_tpot_ms": 10.0,
            "p50_tpot_ms": 9.0,
            "p90_tpot_ms": 15.0,
            "p95_tpot_ms": 18.0,
            "p99_tpot_ms": 22.0,
            "mean_itl_ms": 8.0,
            "p50_itl_ms": 7.0,
            "p90_itl_ms": 12.0,
            "p95_itl_ms": 14.0,
            "p99_itl_ms": 18.0,
            "mean_e2el_ms": 200.0,
            "p50_e2el_ms": 190.0,
            "p90_e2el_ms": 250.0,
            "p95_e2el_ms": 270.0,
            "p99_e2el_ms": 300.0,
        }
        if partial:
            d["partial"] = True
        return d

    def test_no_partial_no_warning(self) -> None:
        result = compare(self._make_result(), self._make_result())
        assert result.baseline_partial is False
        assert result.candidate_partial is False
        table = format_comparison_table(result)
        assert "PARTIAL" not in table

    def test_baseline_partial_warning(self) -> None:
        result = compare(
            self._make_result(partial=True),
            self._make_result(),
        )
        assert result.baseline_partial is True
        assert result.candidate_partial is False
        table = format_comparison_table(result)
        assert "PARTIAL" in table
        assert "baseline" in table

    def test_candidate_partial_warning(self) -> None:
        result = compare(
            self._make_result(),
            self._make_result(partial=True),
        )
        assert result.candidate_partial is True
        table = format_comparison_table(result)
        assert "candidate" in table

    def test_both_partial_warning(self) -> None:
        result = compare(
            self._make_result(partial=True),
            self._make_result(partial=True),
        )
        assert result.baseline_partial is True
        assert result.candidate_partial is True
        table = format_comparison_table(result)
        assert "baseline" in table
        assert "candidate" in table

    def test_partial_flag_in_to_dict(self) -> None:
        result = compare(
            self._make_result(partial=True),
            self._make_result(),
        )
        d = result.to_dict()
        assert d["baseline_partial"] is True
        assert "candidate_partial" not in d

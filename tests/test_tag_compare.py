"""Tests for benchmark result diffing by tag (M95)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpyd_bench.tag_compare import (
    _mann_whitney_u,
    _mean,
    compute_group_stats,
    compute_pairwise_significance,
    format_tag_compare_markdown,
    format_tag_compare_table,
    group_results_by_tag,
    tag_compare,
    tag_compare_main,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    tags: dict | None = None,
    request_throughput: float = 10.0,
    mean_ttft_ms: float = 50.0,
    mean_e2el_ms: float = 200.0,
    mean_tpot_ms: float = 20.0,
    output_throughput: float = 100.0,
    **extra: float,
) -> dict:
    """Create a minimal benchmark result dict for testing."""
    r: dict = {
        "tags": tags or {},
        "request_throughput": request_throughput,
        "output_throughput": output_throughput,
        "mean_ttft_ms": mean_ttft_ms,
        "mean_e2el_ms": mean_e2el_ms,
        "mean_tpot_ms": mean_tpot_ms,
        "total_token_throughput": request_throughput * 10,
        "model": "test-model",
        "num_prompts": 100,
        "completed": 100,
        "failed": 0,
    }
    # Add percentile metrics
    for prefix in ["ttft", "tpot", "e2el"]:
        base = r.get(f"mean_{prefix}_ms", 50.0)
        for p in [50, 90, 95, 99]:
            key = f"p{p}_{prefix}_ms"
            if key not in r:
                r[key] = base * (1 + p / 100)
    r.update(extra)
    return r


def _write_results(tmpdir: Path, results: list[dict]) -> None:
    """Write result dicts as JSON files in tmpdir."""
    for i, r in enumerate(results):
        path = tmpdir / f"result-{i:03d}.json"
        with open(path, "w") as f:
            json.dump(r, f)


# ---------------------------------------------------------------------------
# Tests: group_results_by_tag
# ---------------------------------------------------------------------------

class TestGroupResultsByTag:
    def test_basic_grouping(self):
        results = [
            _make_result(tags={"gpu": "A100"}),
            _make_result(tags={"gpu": "H100"}),
            _make_result(tags={"gpu": "A100"}),
        ]
        groups = group_results_by_tag(results, "gpu")
        assert set(groups.keys()) == {"A100", "H100"}
        assert len(groups["A100"]) == 2
        assert len(groups["H100"]) == 1

    def test_missing_tag_key(self):
        results = [
            _make_result(tags={"gpu": "A100"}),
            _make_result(tags={"env": "prod"}),
            _make_result(tags={}),
        ]
        groups = group_results_by_tag(results, "gpu")
        assert set(groups.keys()) == {"A100"}
        assert len(groups["A100"]) == 1

    def test_no_results_with_tag(self):
        results = [
            _make_result(tags={"env": "prod"}),
        ]
        groups = group_results_by_tag(results, "gpu")
        assert groups == {}

    def test_empty_results(self):
        groups = group_results_by_tag([], "gpu")
        assert groups == {}

    def test_none_tags_field(self):
        results = [{"tags": None}, {"tags": {"gpu": "A100"}}]
        groups = group_results_by_tag(results, "gpu")
        assert len(groups) == 1

    def test_numeric_tag_value_converted_to_string(self):
        results = [
            _make_result(tags={"batch_size": 32}),
            _make_result(tags={"batch_size": 64}),
        ]
        groups = group_results_by_tag(results, "batch_size")
        assert "32" in groups
        assert "64" in groups


# ---------------------------------------------------------------------------
# Tests: compute_group_stats
# ---------------------------------------------------------------------------

class TestComputeGroupStats:
    def test_basic_stats(self):
        groups = {
            "A100": [
                _make_result(request_throughput=10.0, mean_ttft_ms=50.0),
                _make_result(request_throughput=12.0, mean_ttft_ms=40.0),
            ],
            "H100": [
                _make_result(request_throughput=20.0, mean_ttft_ms=25.0),
            ],
        }
        stats = compute_group_stats(groups)
        assert stats["A100"]["count"] == 2
        assert stats["A100"]["request_throughput"] == pytest.approx(11.0)
        assert stats["A100"]["mean_ttft_ms"] == pytest.approx(45.0)
        assert stats["H100"]["request_throughput"] == pytest.approx(20.0)

    def test_missing_metric_returns_none(self):
        groups = {"A100": [{"tags": {"gpu": "A100"}}]}
        stats = compute_group_stats(groups)
        assert stats["A100"]["request_throughput"] is None

    def test_single_run_group(self):
        groups = {"A100": [_make_result(request_throughput=15.0)]}
        stats = compute_group_stats(groups)
        assert stats["A100"]["request_throughput"] == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# Tests: Mann-Whitney U
# ---------------------------------------------------------------------------

class TestMannWhitneyU:
    def test_identical_samples(self):
        u, p = _mann_whitney_u([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        assert p > 0.05  # not significant

    def test_different_samples(self):
        u, p = _mann_whitney_u(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        )
        assert p < 0.05  # significant

    def test_small_samples_not_significant(self):
        u, p = _mann_whitney_u([1, 2], [3, 4])
        assert p == 1.0  # too small

    def test_empty_input(self):
        u, p = _mann_whitney_u([], [1, 2, 3])
        assert p == 1.0


# ---------------------------------------------------------------------------
# Tests: compute_pairwise_significance
# ---------------------------------------------------------------------------

class TestPairwiseSignificance:
    def test_two_groups(self):
        groups = {
            "A100": [_make_result(request_throughput=float(i)) for i in range(10, 20)],
            "H100": [_make_result(request_throughput=float(i)) for i in range(100, 110)],
        }
        results = compute_pairwise_significance(groups, "request_throughput")
        assert len(results) == 1
        assert results[0]["group_a"] == "A100"
        assert results[0]["group_b"] == "H100"
        assert results[0]["significant"] is True

    def test_three_groups(self):
        groups = {
            "A": [_make_result() for _ in range(5)],
            "B": [_make_result() for _ in range(5)],
            "C": [_make_result() for _ in range(5)],
        }
        results = compute_pairwise_significance(groups, "request_throughput")
        assert len(results) == 3  # A-B, A-C, B-C

    def test_empty_metric(self):
        groups = {"A": [{"tags": {}}], "B": [{"tags": {}}]}
        results = compute_pairwise_significance(groups, "request_throughput")
        assert results == []


# ---------------------------------------------------------------------------
# Tests: tag_compare (integration)
# ---------------------------------------------------------------------------

class TestTagCompare:
    def test_full_comparison(self, tmp_path):
        results = [
            _make_result(tags={"gpu": "A100"}, request_throughput=10.0),
            _make_result(tags={"gpu": "A100"}, request_throughput=12.0),
            _make_result(tags={"gpu": "H100"}, request_throughput=20.0),
            _make_result(tags={"gpu": "H100"}, request_throughput=22.0),
        ]
        _write_results(tmp_path, results)

        result = tag_compare(str(tmp_path), "gpu")
        assert result["group_by"] == "gpu"
        assert result["groups"] == {"A100": 2, "H100": 2}
        assert "A100" in result["stats"]
        assert "H100" in result["stats"]
        assert isinstance(result["significance"], list)

    def test_no_matching_tag(self, tmp_path):
        results = [_make_result(tags={"env": "prod"})]
        _write_results(tmp_path, results)

        result = tag_compare(str(tmp_path), "gpu")
        assert "error" in result

    def test_empty_dir(self, tmp_path):
        result = tag_compare(str(tmp_path), "gpu")
        assert "error" in result

    def test_nonexistent_dir(self):
        result = tag_compare("/nonexistent/path", "gpu")
        assert "error" in result


# ---------------------------------------------------------------------------
# Tests: formatting
# ---------------------------------------------------------------------------

class TestFormatting:
    def _sample_result(self):
        return {
            "group_by": "gpu",
            "groups": {"A100": 2, "H100": 1},
            "stats": {
                "A100": {
                    "count": 2,
                    "request_throughput": 11.0,
                    "mean_ttft_ms": 45.0,
                    "mean_e2el_ms": 200.0,
                    "output_throughput": None,
                    "mean_tpot_ms": None,
                    "total_token_throughput": None,
                },
                "H100": {
                    "count": 1,
                    "request_throughput": 20.0,
                    "mean_ttft_ms": 25.0,
                    "mean_e2el_ms": 150.0,
                    "output_throughput": None,
                    "mean_tpot_ms": None,
                    "total_token_throughput": None,
                },
            },
            "significance": [
                {
                    "metric": "request_throughput",
                    "group_a": "A100",
                    "group_b": "H100",
                    "u_stat": 0.0,
                    "p_value": 0.01,
                    "significant": True,
                }
            ],
        }

    def test_table_format(self):
        result = self._sample_result()
        output = format_tag_compare_table(result)
        assert "gpu" in output
        assert "A100" in output
        assert "H100" in output
        assert "request_throughput" in output

    def test_table_error(self):
        result = {"error": "No results found"}
        output = format_tag_compare_table(result)
        assert "Error" in output

    def test_markdown_format(self):
        result = self._sample_result()
        output = format_tag_compare_markdown(result)
        assert "##" in output
        assert "A100" in output
        assert "|" in output

    def test_markdown_error(self):
        result = {"error": "No results found"}
        output = format_tag_compare_markdown(result)
        assert "Error" in output

    def test_table_no_groups(self):
        result = {"group_by": "gpu", "stats": {}, "significance": []}
        output = format_tag_compare_table(result)
        assert "No groups" in output

    def test_markdown_no_groups(self):
        result = {"group_by": "gpu", "stats": {}, "significance": []}
        output = format_tag_compare_markdown(result)
        assert "No groups" in output


# ---------------------------------------------------------------------------
# Tests: CLI
# ---------------------------------------------------------------------------

class TestCLI:
    def test_basic_cli(self, tmp_path, capsys):
        results = [
            _make_result(tags={"gpu": "A100"}, request_throughput=10.0),
            _make_result(tags={"gpu": "H100"}, request_throughput=20.0),
        ]
        _write_results(tmp_path, results)

        tag_compare_main(["--result-dir", str(tmp_path), "--group-by", "gpu"])
        captured = capsys.readouterr()
        assert "A100" in captured.out
        assert "H100" in captured.out

    def test_json_output(self, tmp_path, capsys):
        results = [
            _make_result(tags={"env": "prod"}, request_throughput=10.0),
            _make_result(tags={"env": "staging"}, request_throughput=8.0),
        ]
        _write_results(tmp_path, results)

        tag_compare_main([
            "--result-dir", str(tmp_path),
            "--group-by", "env",
            "--json",
        ])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["group_by"] == "env"
        assert "prod" in data["groups"]

    def test_markdown_output(self, tmp_path, capsys):
        results = [
            _make_result(tags={"gpu": "A100"}),
            _make_result(tags={"gpu": "H100"}),
        ]
        _write_results(tmp_path, results)

        tag_compare_main([
            "--result-dir", str(tmp_path),
            "--group-by", "gpu",
            "--markdown",
        ])
        captured = capsys.readouterr()
        assert "##" in captured.out
        assert "|" in captured.out

    def test_no_matching_tag_exits_nonzero(self, tmp_path):
        results = [_make_result(tags={"env": "prod"})]
        _write_results(tmp_path, results)

        with pytest.raises(SystemExit) as exc_info:
            tag_compare_main([
                "--result-dir", str(tmp_path),
                "--group-by", "gpu",
            ])
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Tests: _mean helper
# ---------------------------------------------------------------------------

class TestMean:
    def test_basic(self):
        assert _mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_empty(self):
        assert _mean([]) is None

    def test_single(self):
        assert _mean([42.0]) == pytest.approx(42.0)

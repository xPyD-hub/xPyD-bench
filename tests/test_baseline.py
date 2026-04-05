"""Tests for Benchmark Baseline Registry (M82)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpyd_bench.baseline import (
    compare_against_baseline,
    delete_baseline,
    list_baselines,
    save_baseline,
    show_baseline,
)


@pytest.fixture()
def baseline_dir(tmp_path: Path) -> str:
    return str(tmp_path / "baselines")


@pytest.fixture()
def sample_result(tmp_path: Path) -> str:
    data = {
        "model": "test-model",
        "mean_ttft_ms": 50.0,
        "mean_tpot_ms": 10.0,
        "output_throughput": 100.0,
        "mean_e2e_latency_ms": 200.0,
        "p99_ttft_ms": 80.0,
        "p99_tpot_ms": 15.0,
        "p99_e2e_latency_ms": 350.0,
    }
    p = tmp_path / "result.json"
    p.write_text(json.dumps(data))
    return str(p)


class TestSaveBaseline:
    def test_save_creates_file(self, baseline_dir: str, sample_result: str) -> None:
        entry = save_baseline("v1", sample_result, baseline_dir=baseline_dir)
        assert entry["name"] == "v1"
        assert entry["model"] == "test-model"
        assert Path(baseline_dir, "v1.json").exists()
        assert Path(baseline_dir, "index.json").exists()

    def test_save_overwrite(self, baseline_dir: str, sample_result: str) -> None:
        save_baseline("v1", sample_result, baseline_dir=baseline_dir)
        save_baseline("v1", sample_result, baseline_dir=baseline_dir)
        baselines = list_baselines(baseline_dir=baseline_dir)
        assert len(baselines) == 1

    def test_save_missing_file(self, baseline_dir: str) -> None:
        with pytest.raises(FileNotFoundError):
            save_baseline("v1", "/nonexistent/result.json", baseline_dir=baseline_dir)


class TestListBaselines:
    def test_empty(self, baseline_dir: str) -> None:
        assert list_baselines(baseline_dir=baseline_dir) == []

    def test_multiple(self, baseline_dir: str, sample_result: str) -> None:
        save_baseline("a", sample_result, baseline_dir=baseline_dir)
        save_baseline("b", sample_result, baseline_dir=baseline_dir)
        baselines = list_baselines(baseline_dir=baseline_dir)
        assert len(baselines) == 2
        names = {b["name"] for b in baselines}
        assert names == {"a", "b"}


class TestShowBaseline:
    def test_show(self, baseline_dir: str, sample_result: str) -> None:
        save_baseline("v1", sample_result, baseline_dir=baseline_dir)
        data = show_baseline("v1", baseline_dir=baseline_dir)
        assert data["model"] == "test-model"
        assert data["mean_ttft_ms"] == 50.0

    def test_show_missing(self, baseline_dir: str) -> None:
        with pytest.raises(KeyError, match="not found"):
            show_baseline("nonexistent", baseline_dir=baseline_dir)


class TestDeleteBaseline:
    def test_delete(self, baseline_dir: str, sample_result: str) -> None:
        save_baseline("v1", sample_result, baseline_dir=baseline_dir)
        delete_baseline("v1", baseline_dir=baseline_dir)
        assert list_baselines(baseline_dir=baseline_dir) == []
        assert not Path(baseline_dir, "v1.json").exists()

    def test_delete_missing(self, baseline_dir: str) -> None:
        with pytest.raises(KeyError, match="not found"):
            delete_baseline("nonexistent", baseline_dir=baseline_dir)


class TestCompareAgainstBaseline:
    def test_no_regression(self, baseline_dir: str, sample_result: str) -> None:
        save_baseline("v1", sample_result, baseline_dir=baseline_dir)
        current = {
            "mean_ttft_ms": 50.0,
            "mean_tpot_ms": 10.0,
            "output_throughput": 100.0,
        }
        result = compare_against_baseline("v1", current, baseline_dir=baseline_dir)
        assert result["regression_detected"] is False
        assert result["baseline_name"] == "v1"

    def test_regression_detected(self, baseline_dir: str, sample_result: str) -> None:
        save_baseline("v1", sample_result, baseline_dir=baseline_dir)
        current = {
            "mean_ttft_ms": 100.0,  # doubled → regression
            "mean_tpot_ms": 20.0,
            "output_throughput": 50.0,  # halved → regression
        }
        result = compare_against_baseline("v1", current, baseline_dir=baseline_dir)
        assert result["regression_detected"] is True

    def test_custom_threshold(self, baseline_dir: str, sample_result: str) -> None:
        save_baseline("v1", sample_result, baseline_dir=baseline_dir)
        current = {"mean_ttft_ms": 53.0}  # 6% increase
        # With 5% threshold → regression
        r1 = compare_against_baseline("v1", current, baseline_dir=baseline_dir, threshold_pct=5.0)
        assert r1["regression_detected"] is True
        # With 10% threshold → no regression
        r2 = compare_against_baseline("v1", current, baseline_dir=baseline_dir, threshold_pct=10.0)
        assert r2["regression_detected"] is False

    def test_missing_baseline(self, baseline_dir: str) -> None:
        with pytest.raises(KeyError):
            compare_against_baseline("nonexistent", {}, baseline_dir=baseline_dir)


class TestConfigKeys:
    def test_known_keys(self) -> None:
        from xpyd_bench.config_cmd import _KNOWN_KEYS

        assert "compare_baseline" in _KNOWN_KEYS
        assert "baseline_dir" in _KNOWN_KEYS

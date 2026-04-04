"""Tests for M7: Built-in benchmark scenarios."""

from __future__ import annotations

import pytest

from xpyd_bench.scenarios import (
    LONG_CONTEXT,
    MIXED,
    SCENARIOS,
    SHORT,
    STRESS,
    ScenarioConfig,
    get_scenario,
    list_scenarios,
)


class TestScenarioConfig:
    """ScenarioConfig dataclass and registry."""

    def test_four_presets_exist(self) -> None:
        assert len(SCENARIOS) == 4
        assert set(SCENARIOS.keys()) == {"short", "long_context", "mixed", "stress"}

    def test_get_scenario_valid(self) -> None:
        for name in ("short", "long_context", "mixed", "stress"):
            s = get_scenario(name)
            assert isinstance(s, ScenarioConfig)
            assert s.name == name

    def test_get_scenario_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown scenario"):
            get_scenario("nonexistent")

    def test_list_scenarios(self) -> None:
        scenarios = list_scenarios()
        assert len(scenarios) == 4
        assert all(isinstance(s, ScenarioConfig) for s in scenarios)

    def test_to_overrides_excludes_name_and_description(self) -> None:
        overrides = SHORT.to_overrides()
        assert "name" not in overrides
        assert "description" not in overrides

    def test_to_overrides_excludes_none(self) -> None:
        for s in SCENARIOS.values():
            overrides = s.to_overrides()
            for v in overrides.values():
                assert v is not None


class TestShortPreset:
    """Validate the 'short' scenario."""

    def test_short_prompts(self) -> None:
        assert SHORT.input_len == 32

    def test_short_outputs(self) -> None:
        assert SHORT.output_len == 64

    def test_high_concurrency(self) -> None:
        assert SHORT.max_concurrency is not None
        assert SHORT.max_concurrency >= 32


class TestLongContextPreset:
    """Validate the 'long_context' scenario."""

    def test_long_input(self) -> None:
        assert LONG_CONTEXT.input_len >= 1024

    def test_moderate_output(self) -> None:
        assert LONG_CONTEXT.output_len >= 128

    def test_low_rate(self) -> None:
        assert LONG_CONTEXT.request_rate <= 20.0


class TestMixedPreset:
    """Validate the 'mixed' scenario."""

    def test_uses_synthetic(self) -> None:
        assert MIXED.dataset_name == "synthetic"

    def test_varied_distributions(self) -> None:
        assert MIXED.synthetic_input_len_dist != "fixed"
        assert MIXED.synthetic_output_len_dist != "fixed"


class TestStressPreset:
    """Validate the 'stress' scenario."""

    def test_high_num_prompts(self) -> None:
        assert STRESS.num_prompts >= 1000

    def test_inf_rate(self) -> None:
        assert STRESS.request_rate == float("inf")

    def test_high_concurrency(self) -> None:
        assert STRESS.max_concurrency is not None
        assert STRESS.max_concurrency >= 128


class TestCLIScenarioFlag:
    """Test --scenario and --list-scenarios CLI integration."""

    def test_list_scenarios_exits_cleanly(self, capsys: pytest.CaptureFixture) -> None:
        from xpyd_bench.cli import bench_main

        bench_main(["--list-scenarios"])
        captured = capsys.readouterr()
        assert "short" in captured.out
        assert "long_context" in captured.out
        assert "mixed" in captured.out
        assert "stress" in captured.out

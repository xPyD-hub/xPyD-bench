"""Tests for M47: Response Validation & Content Checking."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from xpyd_bench.bench.models import BenchmarkResult, RequestResult
from xpyd_bench.bench.validation import (
    ValidationResult,
    aggregate_validations,
    parse_validators,
    validate_response,
)
from xpyd_bench.cli import bench_main


class TestParseValidators:
    """Test validator spec parsing."""

    def test_non_empty(self) -> None:
        result = parse_validators(["non-empty"])
        assert result == [("non-empty", None)]

    def test_json(self) -> None:
        result = parse_validators(["json"])
        assert result == [("json", None)]

    def test_regex(self) -> None:
        result = parse_validators(["regex:hello.*world"])
        assert result == [("regex", "hello.*world")]

    def test_min_tokens(self) -> None:
        result = parse_validators(["min-tokens:5"])
        assert result == [("min-tokens", "5")]

    def test_multiple(self) -> None:
        result = parse_validators(["non-empty", "json", "min-tokens:3"])
        assert len(result) == 3

    def test_unknown_validator(self) -> None:
        with pytest.raises(ValueError, match="Unknown validator"):
            parse_validators(["foobar"])

    def test_empty_regex(self) -> None:
        with pytest.raises(ValueError, match="regex pattern is empty"):
            parse_validators(["regex:"])

    def test_invalid_regex(self) -> None:
        with pytest.raises(ValueError, match="Invalid regex"):
            parse_validators(["regex:[invalid"])

    def test_invalid_min_tokens(self) -> None:
        with pytest.raises(ValueError, match="non-negative integer"):
            parse_validators(["min-tokens:abc"])


class TestValidateResponse:
    """Test response validation logic."""

    def test_non_empty_pass(self) -> None:
        result = validate_response("hello world", [("non-empty", None)])
        assert result.passed
        assert not result.errors

    def test_non_empty_fail_empty(self) -> None:
        result = validate_response("", [("non-empty", None)])
        assert not result.passed
        assert "empty" in result.errors[0].lower()

    def test_non_empty_fail_whitespace(self) -> None:
        result = validate_response("   \n  ", [("non-empty", None)])
        assert not result.passed

    def test_json_pass(self) -> None:
        result = validate_response('{"key": "value"}', [("json", None)])
        assert result.passed

    def test_json_fail(self) -> None:
        result = validate_response("not json at all", [("json", None)])
        assert not result.passed
        assert "not valid JSON" in result.errors[0]

    def test_regex_pass(self) -> None:
        result = validate_response("hello world", [("regex", r"hello\s+\w+")])
        assert result.passed

    def test_regex_fail(self) -> None:
        result = validate_response("goodbye", [("regex", r"^hello")])
        assert not result.passed
        assert "regex" in result.errors[0].lower()

    def test_min_tokens_pass(self) -> None:
        result = validate_response("one two three four five", [("min-tokens", "3")])
        assert result.passed

    def test_min_tokens_fail(self) -> None:
        result = validate_response("one two", [("min-tokens", "5")])
        assert not result.passed
        assert "2 tokens" in result.errors[0]

    def test_multiple_validators_all_pass(self) -> None:
        validators = [("non-empty", None), ("min-tokens", "2")]
        result = validate_response("hello world", validators)
        assert result.passed

    def test_multiple_validators_partial_fail(self) -> None:
        validators = [("non-empty", None), ("json", None)]
        result = validate_response("not json", validators)
        assert not result.passed
        assert len(result.errors) == 1  # non-empty passes, json fails


class TestAggregateValidations:
    """Test validation aggregation."""

    def test_all_pass(self) -> None:
        results = [ValidationResult(passed=True), ValidationResult(passed=True)]
        summary = aggregate_validations(results)
        assert summary.total == 2
        assert summary.passed == 2
        assert summary.failed == 0
        assert summary.pass_rate == 100.0

    def test_mixed(self) -> None:
        results = [
            ValidationResult(passed=True),
            ValidationResult(passed=False, errors=["Response is empty"]),
            ValidationResult(passed=False, errors=["Response is not valid JSON"]),
        ]
        summary = aggregate_validations(results)
        assert summary.total == 3
        assert summary.passed == 1
        assert summary.failed == 2
        assert summary.pass_rate == pytest.approx(33.33, abs=0.01)

    def test_empty(self) -> None:
        summary = aggregate_validations([])
        assert summary.total == 0
        assert summary.pass_rate == 0.0


class TestModels:
    """Test model field additions for M47."""

    def test_request_result_has_validation_errors(self) -> None:
        r = RequestResult()
        assert r.validation_errors == []
        assert r.response_text is None

    def test_request_result_validation_errors_set(self) -> None:
        r = RequestResult(validation_errors=["Response is empty"])
        assert len(r.validation_errors) == 1

    def test_benchmark_result_has_validation_summary(self) -> None:
        r = BenchmarkResult()
        assert r.validation_summary is None

    def test_benchmark_result_validation_summary_serializes(self) -> None:
        r = BenchmarkResult(validation_summary={"total": 10, "passed": 8, "failed": 2})
        d = asdict(r)
        assert d["validation_summary"]["total"] == 10


class TestCLI:
    """Test CLI flag parsing."""

    def test_validate_response_flag_parsed(self, capsys: pytest.CaptureFixture[str]) -> None:
        """--validate-response should be accepted in dry-run."""
        bench_main([
            "--dry-run",
            "--validate-response", "non-empty",
            "--validate-response", "min-tokens:5",
        ])
        # Should not raise

    def test_validate_response_yaml(self, capsys: pytest.CaptureFixture[str], tmp_path) -> None:
        """validate_response in YAML config should be applied."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text('validate_response: ["non-empty", "json"]\n')
        bench_main(["--dry-run", "--config", str(cfg)])
        # Should not raise

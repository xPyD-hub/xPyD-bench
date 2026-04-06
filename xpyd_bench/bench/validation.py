"""Response validation for benchmark requests (M47)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Result of validating a single response."""

    passed: bool = True
    errors: list[str] = field(default_factory=list)


@dataclass
class ValidationSummary:
    """Aggregated validation results across all requests."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    error_counts: dict[str, int] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        """Return pass rate as a percentage."""
        return (self.passed / self.total * 100) if self.total > 0 else 0.0


def parse_validators(specs: list[str]) -> list[tuple[str, str | None]]:
    """Parse validator specs into (type, param) tuples.

    Supported formats:
        - "non-empty"
        - "json"
        - "regex:<pattern>"
        - "min-tokens:<N>"
    """
    validators: list[tuple[str, str | None]] = []
    for spec in specs:
        if spec == "non-empty":
            validators.append(("non-empty", None))
        elif spec == "json":
            validators.append(("json", None))
        elif spec.startswith("regex:"):
            pattern = spec[len("regex:"):]
            if not pattern:
                raise ValueError(f"Invalid validator spec '{spec}': regex pattern is empty")
            # Validate the regex compiles
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"Invalid regex pattern in '{spec}': {exc}") from exc
            validators.append(("regex", pattern))
        elif spec.startswith("min-tokens:"):
            param = spec[len("min-tokens:"):]
            try:
                n = int(param)
                if n < 0:
                    raise ValueError("must be non-negative")
            except ValueError as exc:
                raise ValueError(
                    f"Invalid validator spec '{spec}': "
                    f"min-tokens requires a non-negative integer ({exc})"
                ) from exc
            validators.append(("min-tokens", str(n)))
        else:
            raise ValueError(
                f"Unknown validator '{spec}'. "
                "Supported: non-empty, json, regex:<pattern>, min-tokens:<N>"
            )
    return validators


def validate_response(
    text: str,
    validators: list[tuple[str, str | None]],
) -> ValidationResult:
    """Validate a response string against a list of validators."""
    result = ValidationResult()
    for vtype, param in validators:
        if vtype == "non-empty":
            if not text or not text.strip():
                result.passed = False
                result.errors.append("Response is empty")
        elif vtype == "json":
            try:
                json.loads(text)
            except (json.JSONDecodeError, TypeError):
                result.passed = False
                result.errors.append("Response is not valid JSON")
        elif vtype == "regex":
            assert param is not None
            if not re.search(param, text):
                result.passed = False
                result.errors.append(f"Response does not match regex: {param}")
        elif vtype == "min-tokens":
            assert param is not None
            min_count = int(param)
            # Simple whitespace-based token count approximation
            token_count = len(text.split()) if text else 0
            if token_count < min_count:
                result.passed = False
                result.errors.append(
                    f"Response has {token_count} tokens, minimum required: {min_count}"
                )
    return result


def aggregate_validations(results: list[ValidationResult]) -> ValidationSummary:
    """Aggregate individual validation results into a summary."""
    summary = ValidationSummary(total=len(results))
    for r in results:
        if r.passed:
            summary.passed += 1
        else:
            summary.failed += 1
            for err in r.errors:
                # Use the error prefix as the key
                key = err.split(":")[0] if ":" in err else err
                summary.error_counts[key] = summary.error_counts.get(key, 0) + 1
    return summary

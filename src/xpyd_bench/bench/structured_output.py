"""Structured output & function calling validation (M56).

Validates that responses conform to JSON schemas and correctly produce
tool calls when tools are specified.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallResult:
    """Validation result for a single tool call extraction."""

    success: bool = True
    function_name: str | None = None
    arguments_valid: bool = True
    errors: list[str] = field(default_factory=list)


@dataclass
class StructuredOutputResult:
    """Per-request structured output validation result."""

    tool_calls_expected: bool = False
    tool_calls_found: int = 0
    tool_call_results: list[ToolCallResult] = field(default_factory=list)
    json_schema_valid: bool | None = None  # None if no schema validation
    schema_errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Overall success: all tool calls valid and schema conforms."""
        if self.tool_calls_expected and self.tool_calls_found == 0:
            return False
        if any(not tc.success for tc in self.tool_call_results):
            return False
        if self.json_schema_valid is not None and not self.json_schema_valid:
            return False
        return True


@dataclass
class StructuredOutputSummary:
    """Aggregated structured output metrics across all requests."""

    total_requests: int = 0
    tool_call_requests: int = 0
    tool_call_successes: int = 0
    tool_call_failures: int = 0
    schema_validations: int = 0
    schema_passes: int = 0
    schema_failures: int = 0
    total_tool_calls_extracted: int = 0

    @property
    def tool_call_success_rate(self) -> float:
        """Rate of successful tool call extractions."""
        if self.tool_call_requests == 0:
            return 0.0
        return self.tool_call_successes / self.tool_call_requests * 100.0

    @property
    def schema_conformance_rate(self) -> float:
        """Rate of schema-conforming responses."""
        if self.schema_validations == 0:
            return 0.0
        return self.schema_passes / self.schema_validations * 100.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON output."""
        d: dict[str, Any] = {
            "total_requests": self.total_requests,
            "tool_call_requests": self.tool_call_requests,
            "tool_call_successes": self.tool_call_successes,
            "tool_call_failures": self.tool_call_failures,
            "tool_call_success_rate": round(self.tool_call_success_rate, 2),
            "total_tool_calls_extracted": self.total_tool_calls_extracted,
        }
        if self.schema_validations > 0:
            d["schema_validations"] = self.schema_validations
            d["schema_passes"] = self.schema_passes
            d["schema_failures"] = self.schema_failures
            d["schema_conformance_rate"] = round(self.schema_conformance_rate, 2)
        return d


def _validate_json_schema(data: Any, schema: dict) -> list[str]:
    """Validate data against a JSON schema. Returns list of errors."""
    errors: list[str] = []
    schema_type = schema.get("type")

    if schema_type == "object":
        if not isinstance(data, dict):
            errors.append(f"Expected object, got {type(data).__name__}")
            return errors
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        for req_key in required:
            if req_key not in data:
                errors.append(f"Missing required property: {req_key}")
        for key, prop_schema in properties.items():
            if key in data:
                sub_errors = _validate_json_schema(data[key], prop_schema)
                errors.extend(f"{key}.{e}" for e in sub_errors)
    elif schema_type == "array":
        if not isinstance(data, list):
            errors.append(f"Expected array, got {type(data).__name__}")
        elif "items" in schema:
            for i, item in enumerate(data):
                sub_errors = _validate_json_schema(item, schema["items"])
                errors.extend(f"[{i}].{e}" for e in sub_errors)
    elif schema_type == "string":
        if not isinstance(data, str):
            errors.append(f"Expected string, got {type(data).__name__}")
    elif schema_type == "number":
        if not isinstance(data, (int, float)):
            errors.append(f"Expected number, got {type(data).__name__}")
    elif schema_type == "integer":
        if not isinstance(data, int) or isinstance(data, bool):
            errors.append(f"Expected integer, got {type(data).__name__}")
    elif schema_type == "boolean":
        if not isinstance(data, bool):
            errors.append(f"Expected boolean, got {type(data).__name__}")

    return errors


def validate_tool_calls(
    response_body: dict,
    tools: list[dict] | None = None,
) -> StructuredOutputResult:
    """Validate tool calls in a chat completion response.

    Args:
        response_body: The full response JSON body.
        tools: The tool definitions sent in the request (for argument validation).

    Returns:
        StructuredOutputResult with per-tool-call validation.
    """
    result = StructuredOutputResult(tool_calls_expected=tools is not None and len(tools) > 0)

    choices = response_body.get("choices", [])
    if not choices:
        return result

    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls", [])
    result.tool_calls_found = len(tool_calls)

    # Build function name -> parameters schema lookup
    func_schemas: dict[str, dict] = {}
    if tools:
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                fname = func.get("name", "")
                if fname:
                    func_schemas[fname] = func.get("parameters", {})

    for tc in tool_calls:
        tc_result = ToolCallResult()
        func = tc.get("function", {})
        tc_result.function_name = func.get("name")

        if not tc_result.function_name:
            tc_result.success = False
            tc_result.errors.append("Tool call missing function name")
            result.tool_call_results.append(tc_result)
            continue

        # Validate arguments are parseable JSON
        raw_args = func.get("arguments", "")
        try:
            parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except (json.JSONDecodeError, TypeError):
            tc_result.success = False
            tc_result.arguments_valid = False
            tc_result.errors.append(f"Invalid JSON arguments for {tc_result.function_name}")
            result.tool_call_results.append(tc_result)
            continue

        # Validate against schema if available
        if tc_result.function_name in func_schemas:
            schema = func_schemas[tc_result.function_name]
            if schema:
                schema_errors = _validate_json_schema(parsed_args, schema)
                if schema_errors:
                    tc_result.success = False
                    tc_result.arguments_valid = False
                    tc_result.errors.extend(schema_errors)

        result.tool_call_results.append(tc_result)

    return result


def validate_json_response(
    response_text: str | None,
    response_format: dict | None = None,
) -> StructuredOutputResult:
    """Validate response conforms to JSON mode or JSON schema.

    Args:
        response_text: The response content text.
        response_format: The response_format dict from the request.

    Returns:
        StructuredOutputResult with schema validation results.
    """
    result = StructuredOutputResult()

    if not response_format:
        return result

    fmt_type = response_format.get("type")
    if fmt_type not in ("json_object", "json_schema"):
        return result

    if not response_text:
        result.json_schema_valid = False
        result.schema_errors.append("Empty response when JSON output expected")
        return result

    # Validate it's valid JSON
    try:
        parsed = json.loads(response_text)
    except (json.JSONDecodeError, TypeError) as exc:
        result.json_schema_valid = False
        result.schema_errors.append(f"Response is not valid JSON: {exc}")
        return result

    if fmt_type == "json_object":
        if not isinstance(parsed, dict):
            result.json_schema_valid = False
            result.schema_errors.append(
                f"Expected JSON object, got {type(parsed).__name__}"
            )
        else:
            result.json_schema_valid = True
    elif fmt_type == "json_schema":
        schema = response_format.get("json_schema", {}).get("schema", {})
        if schema:
            errors = _validate_json_schema(parsed, schema)
            result.json_schema_valid = len(errors) == 0
            result.schema_errors.extend(errors)
        else:
            result.json_schema_valid = True

    return result


def aggregate_structured_output(
    results: list[StructuredOutputResult],
) -> StructuredOutputSummary:
    """Aggregate per-request structured output results."""
    summary = StructuredOutputSummary(total_requests=len(results))
    for r in results:
        if r.tool_calls_expected:
            summary.tool_call_requests += 1
            summary.total_tool_calls_extracted += r.tool_calls_found
            if r.tool_calls_found > 0 and all(tc.success for tc in r.tool_call_results):
                summary.tool_call_successes += 1
            else:
                summary.tool_call_failures += 1
        if r.json_schema_valid is not None:
            summary.schema_validations += 1
            if r.json_schema_valid:
                summary.schema_passes += 1
            else:
                summary.schema_failures += 1
    return summary

"""OpenTelemetry trace export support (M62).

Lightweight OTLP/HTTP JSON exporter for benchmark request spans.
No external OpenTelemetry SDK dependency — uses httpx to POST
OTLP JSON directly.
"""

from __future__ import annotations

import random
import struct
import time
from typing import Any


def _random_trace_id() -> str:
    """Generate a random 32-hex-char trace ID."""
    return struct.pack(">QQ", random.getrandbits(64), random.getrandbits(64)).hex()


def _random_span_id() -> str:
    """Generate a random 16-hex-char span ID."""
    return struct.pack(">Q", random.getrandbits(64)).hex()


def _to_nano(ts: float) -> int:
    """Convert seconds (float) to nanoseconds (int)."""
    return int(ts * 1_000_000_000)


def build_spans(
    result: dict,
    service_name: str = "xpyd-bench",
) -> dict:
    """Build OTLP-compatible resource spans from a BenchmarkResult dict.

    Returns the OTLP JSON body ready for POST to /v1/traces.
    """
    trace_id = _random_trace_id()
    root_span_id = _random_span_id()

    # Determine benchmark time range
    request_results: list[dict] = result.get("request_results", [])
    start_time = result.get("start_time", time.time())
    end_time = result.get("end_time", start_time + result.get("total_time", 0))

    # Root span for entire benchmark run
    root_attrs = _make_attrs({
        "bench.model": result.get("model", ""),
        "bench.endpoint": result.get("endpoint", ""),
        "bench.num_prompts": result.get("completed", 0),
        "bench.total_time_s": result.get("total_time", 0),
        "bench.request_throughput": result.get("request_throughput", 0),
    })

    root_span = {
        "traceId": trace_id,
        "spanId": root_span_id,
        "name": "benchmark-run",
        "kind": 1,  # SPAN_KIND_INTERNAL
        "startTimeUnixNano": _to_nano(start_time),
        "endTimeUnixNano": _to_nano(end_time),
        "attributes": root_attrs,
        "status": {"code": 1},  # STATUS_CODE_OK
    }

    child_spans = []
    for i, req in enumerate(request_results):
        span_id = _random_span_id()
        req_start = req.get("start_time", start_time)
        req_end = req_start + req.get("latency", 0)

        attrs = _make_attrs({
            "bench.request.index": i,
            "bench.request.prompt_tokens": req.get("prompt_tokens", 0),
            "bench.request.completion_tokens": req.get("completion_tokens", 0),
            "bench.request.ttft": req.get("ttft", 0),
            "bench.request.tpot": req.get("tpot", 0),
            "bench.request.latency": req.get("latency", 0),
            "bench.request.success": req.get("success", False),
            "bench.request.error": req.get("error", ""),
        })

        child_spans.append({
            "traceId": trace_id,
            "spanId": span_id,
            "parentSpanId": root_span_id,
            "name": f"request-{i}",
            "kind": 3,  # SPAN_KIND_CLIENT
            "startTimeUnixNano": _to_nano(req_start),
            "endTimeUnixNano": _to_nano(req_end),
            "attributes": attrs,
            "status": {"code": 1 if req.get("success", False) else 2},
        })

    resource_attrs = _make_attrs({"service.name": service_name})

    return {
        "resourceSpans": [
            {
                "resource": {"attributes": resource_attrs},
                "scopeSpans": [
                    {
                        "scope": {"name": "xpyd-bench", "version": "0.1.0"},
                        "spans": [root_span, *child_spans],
                    }
                ],
            }
        ]
    }


def _make_attrs(mapping: dict[str, Any]) -> list[dict]:
    """Convert a flat dict to OTLP attribute list format."""
    attrs = []
    for k, v in mapping.items():
        if v is None or v == "":
            continue
        if isinstance(v, bool):
            attrs.append({"key": k, "value": {"boolValue": v}})
        elif isinstance(v, int):
            attrs.append({"key": k, "value": {"intValue": v}})
        elif isinstance(v, float):
            attrs.append({"key": k, "value": {"doubleValue": v}})
        else:
            attrs.append({"key": k, "value": {"stringValue": str(v)}})
    return attrs


def export_traces(
    endpoint: str,
    result: dict,
    service_name: str = "xpyd-bench",
    timeout: float = 10.0,
) -> dict:
    """Export benchmark spans to an OTLP/HTTP endpoint.

    Posts to ``{endpoint}/v1/traces`` (OTLP/HTTP JSON).

    Returns:
        {"success": bool, "spans_count": int, "error": str | None}
    """
    import httpx

    body = build_spans(result, service_name=service_name)
    spans_count = sum(
        len(ss["spans"])
        for rs in body["resourceSpans"]
        for ss in rs["scopeSpans"]
    )

    url = endpoint.rstrip("/") + "/v1/traces"
    headers = {"Content-Type": "application/json"}

    try:
        resp = httpx.post(url, json=body, headers=headers, timeout=timeout)
        if resp.status_code < 400:
            return {"success": True, "spans_count": spans_count, "error": None}
        return {
            "success": False,
            "spans_count": spans_count,
            "error": f"HTTP {resp.status_code}",
        }
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "spans_count": spans_count, "error": str(exc)}


def format_otlp_summary(delivery: dict) -> str:
    """Format OTLP export result for terminal output."""
    if delivery["success"]:
        return f"OTLP trace export: ✓ {delivery['spans_count']} spans exported"
    return f"OTLP trace export: ✗ {delivery['error']} ({delivery['spans_count']} spans)"

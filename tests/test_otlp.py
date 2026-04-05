"""Tests for OTLP trace export support (M62)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from xpyd_bench.otlp import (
    _make_attrs,
    _random_span_id,
    _random_trace_id,
    _to_nano,
    build_spans,
    export_traces,
    format_otlp_summary,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _OTLPHandler(BaseHTTPRequestHandler):
    received: list[dict] = []
    status_to_return: int = 200

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        self._cls().received.append(json.loads(body))
        self.send_response(self._cls().status_to_return)
        self.end_headers()

    def _cls(self):
        return type(self)

    def log_message(self, *_args):
        pass


@pytest.fixture()
def otlp_server():
    class Handler(_OTLPHandler):
        received: list[dict] = []
        status_to_return: int = 200

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}", Handler
    server.shutdown()


def _sample_result(n_requests: int = 3) -> dict:
    return {
        "model": "test-model",
        "endpoint": "/v1/completions",
        "completed": n_requests,
        "total_time": 10.0,
        "start_time": 1000.0,
        "end_time": 1010.0,
        "request_throughput": float(n_requests) / 10.0,
        "request_results": [
            {
                "start_time": 1000.0 + i,
                "latency": 0.5,
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "ttft": 0.1,
                "tpot": 0.02,
                "success": True,
                "error": None,
            }
            for i in range(n_requests)
        ],
    }


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_random_trace_id_length():
    tid = _random_trace_id()
    assert len(tid) == 32
    int(tid, 16)  # valid hex


def test_random_span_id_length():
    sid = _random_span_id()
    assert len(sid) == 16
    int(sid, 16)


def test_to_nano():
    assert _to_nano(1.5) == 1_500_000_000


def test_make_attrs_types():
    attrs = _make_attrs({
        "str_key": "hello",
        "int_key": 42,
        "float_key": 3.14,
        "bool_key": True,
        "empty": "",
        "none": None,
    })
    keys = {a["key"] for a in attrs}
    assert "str_key" in keys
    assert "int_key" in keys
    assert "float_key" in keys
    assert "bool_key" in keys
    assert "empty" not in keys
    assert "none" not in keys


def test_build_spans_structure():
    result = _sample_result(3)
    body = build_spans(result)

    assert "resourceSpans" in body
    rs = body["resourceSpans"]
    assert len(rs) == 1
    scope_spans = rs[0]["scopeSpans"]
    assert len(scope_spans) == 1
    spans = scope_spans[0]["spans"]
    # 1 root + 3 child
    assert len(spans) == 4

    root = spans[0]
    assert root["name"] == "benchmark-run"
    assert "parentSpanId" not in root

    for child in spans[1:]:
        assert child["parentSpanId"] == root["spanId"]
        assert child["traceId"] == root["traceId"]


def test_build_spans_attributes():
    result = _sample_result(1)
    body = build_spans(result)
    child = body["resourceSpans"][0]["scopeSpans"][0]["spans"][1]
    attr_keys = {a["key"] for a in child["attributes"]}
    assert "bench.request.prompt_tokens" in attr_keys
    assert "bench.request.completion_tokens" in attr_keys
    assert "bench.request.ttft" in attr_keys
    assert "bench.request.tpot" in attr_keys


def test_build_spans_service_name():
    body = build_spans(_sample_result(1), service_name="my-service")
    resource_attrs = body["resourceSpans"][0]["resource"]["attributes"]
    svc = [a for a in resource_attrs if a["key"] == "service.name"]
    assert len(svc) == 1
    assert svc[0]["value"]["stringValue"] == "my-service"


def test_export_traces_success(otlp_server):
    url, handler = otlp_server
    result = _sample_result(2)
    delivery = export_traces(url, result)

    assert delivery["success"] is True
    assert delivery["spans_count"] == 3  # 1 root + 2 child
    assert delivery["error"] is None
    assert len(handler.received) == 1

    # Verify posted body structure
    body = handler.received[0]
    assert "resourceSpans" in body


def test_export_traces_server_error(otlp_server):
    url, handler = otlp_server
    handler.status_to_return = 500
    delivery = export_traces(url, _sample_result(1))
    assert delivery["success"] is False
    assert "HTTP 500" in delivery["error"]


def test_export_traces_unreachable():
    delivery = export_traces("http://127.0.0.1:1", _sample_result(1), timeout=1.0)
    assert delivery["success"] is False
    assert delivery["error"] is not None


def test_format_otlp_summary_success():
    s = format_otlp_summary({"success": True, "spans_count": 5, "error": None})
    assert "✓" in s
    assert "5 spans" in s


def test_format_otlp_summary_failure():
    s = format_otlp_summary({"success": False, "spans_count": 5, "error": "timeout"})
    assert "✗" in s
    assert "timeout" in s


def test_otlp_yaml_config_known_keys():
    from xpyd_bench.config_cmd import _KNOWN_KEYS
    assert "otlp_endpoint" in _KNOWN_KEYS


def test_otlp_cli_args():
    import argparse

    from xpyd_bench.cli import _add_vllm_compat_args

    parser = argparse.ArgumentParser()
    _add_vllm_compat_args(parser)
    args = parser.parse_args(["--otlp-endpoint", "http://localhost:4318"])
    assert args.otlp_endpoint == "http://localhost:4318"


def test_build_spans_empty_requests():
    result = _sample_result(0)
    result["request_results"] = []
    body = build_spans(result)
    spans = body["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert len(spans) == 1  # root only

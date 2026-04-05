"""Tests for M48: Endpoint Health Check."""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from xpyd_bench.healthcheck import (
    EndpointCheck,
    healthcheck_main,
    run_healthcheck,
)

# ---------------------------------------------------------------------------
# Helpers: tiny HTTP server
# ---------------------------------------------------------------------------

class _FakeHandler(BaseHTTPRequestHandler):
    """Minimal handler that supports /v1/models, /v1/completions, /v1/chat/completions."""

    def do_GET(self):  # noqa: N802
        if self.path == "/v1/models":
            body = json.dumps({"data": [{"id": "test-model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):  # noqa: N802
        content_len = int(self.headers.get("Content-Length", 0))
        self.rfile.read(content_len)

        if self.path in ("/v1/completions", "/v1/chat/completions"):
            body = json.dumps({
                "id": "hc-test",
                "choices": [{"text": "ok", "message": {"content": "ok"}}],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/v1/embeddings":
            body = json.dumps({
                "data": [{"embedding": [0.1, 0.2]}],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *_args):
        pass  # suppress logs


@pytest.fixture()
def fake_server():
    """Start a fake HTTP server and yield its base URL."""
    server = HTTPServer(("127.0.0.1", 0), _FakeHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_healthcheck_healthy(fake_server):
    """Healthy server should pass all checks."""
    result = asyncio.run(run_healthcheck(fake_server))
    assert result.reachable is True
    assert result.healthy is True
    assert "test-model" in result.models
    assert len(result.endpoints) == 4
    for ep in result.endpoints:
        assert ep.available is True
        assert ep.latency_ms is not None
        assert ep.latency_ms > 0


def test_healthcheck_unreachable():
    """Unreachable host should report unhealthy."""
    result = asyncio.run(
        run_healthcheck("http://127.0.0.1:1", timeout=2.0)
    )
    assert result.reachable is False
    assert result.healthy is False
    assert len(result.errors) > 0


def test_healthcheck_to_dict(fake_server):
    """to_dict should produce a serializable dict."""
    result = asyncio.run(run_healthcheck(fake_server))
    d = result.to_dict()
    # Should be JSON-serializable
    s = json.dumps(d)
    parsed = json.loads(s)
    assert parsed["reachable"] is True
    assert parsed["base_url"] == fake_server
    assert len(parsed["endpoints"]) == 4


def test_healthcheck_with_api_key(fake_server):
    """API key should not break health check (server ignores it)."""
    result = asyncio.run(
        run_healthcheck(fake_server, api_key="test-key-123")
    )
    assert result.healthy is True


def test_healthcheck_cli_json(fake_server, capsys):
    """CLI --json flag should output valid JSON."""
    try:
        healthcheck_main(["--base-url", fake_server, "--json"])
    except SystemExit as e:
        assert e.code == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["reachable"] is True


def test_healthcheck_cli_unhealthy(capsys):
    """CLI should exit 1 for unreachable server."""
    try:
        healthcheck_main(["--base-url", "http://127.0.0.1:1", "--timeout", "2"])
    except SystemExit as e:
        assert e.code == 1


def test_healthcheck_cli_human_readable(fake_server, capsys):
    """CLI default output should contain human-readable text."""
    try:
        healthcheck_main(["--base-url", fake_server])
    except SystemExit as e:
        assert e.code == 0
    out = capsys.readouterr().out
    assert "Health Check" in out
    assert "HEALTHY" in out


def test_endpoint_check_dataclass():
    """EndpointCheck should hold expected fields."""
    ec = EndpointCheck(path="/v1/models", available=True, latency_ms=5.0)
    assert ec.path == "/v1/models"
    assert ec.available is True
    assert ec.error is None


def test_healthcheck_model_auto_detect(fake_server):
    """When no model specified, should auto-detect from /v1/models."""
    result = asyncio.run(run_healthcheck(fake_server))
    # Models should be populated
    assert result.models == ["test-model"]
    # Completions and chat should still work (using auto-detected model)
    comp = next(e for e in result.endpoints if e.path == "/v1/completions")
    assert comp.available is True

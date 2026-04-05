"""Tests for webhook notification support (M61)."""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from xpyd_bench.webhook import (
    compute_signature,
    format_webhook_summary,
    send_webhook,
    send_webhooks,
)


class _WebhookHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler that records received webhooks."""

    received: list[dict] = []
    status_to_return: int = 200

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        self._cls().received.append(
            {
                "body": json.loads(body),
                "headers": dict(self.headers),
            }
        )
        self.send_response(self._cls().status_to_return)
        self.end_headers()

    def _cls(self):
        return type(self)

    def log_message(self, *_args):
        pass  # suppress log output


@pytest.fixture()
def webhook_server():
    """Start a local HTTP server for webhook testing."""

    class Handler(_WebhookHandler):
        received: list[dict] = []
        status_to_return: int = 200

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}", Handler
    server.shutdown()


def test_send_webhook_success(webhook_server):
    url, handler = webhook_server
    result = {"completed": 10, "failed": 0}
    delivery = send_webhook(url, result)

    assert delivery["success"] is True
    assert delivery["status_code"] == 200
    assert delivery["attempts"] == 1
    assert delivery["error"] is None
    assert len(handler.received) == 1
    assert handler.received[0]["body"] == result


def test_send_webhook_with_signature(webhook_server):
    url, handler = webhook_server
    secret = "my-secret"
    result = {"completed": 5}
    send_webhook(url, result, secret=secret)

    payload = json.dumps(result, default=str).encode()
    expected_sig = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    received_sig = handler.received[0]["headers"].get("X-Webhook-Signature")
    assert received_sig == f"sha256={expected_sig}"


def test_send_webhook_retry_on_failure():
    """Webhook to unreachable host should retry and report failure."""
    delivery = send_webhook(
        "http://127.0.0.1:1",  # unreachable
        {"test": True},
        max_retries=2,
        timeout=1.0,
    )
    assert delivery["success"] is False
    assert delivery["attempts"] == 2
    assert delivery["error"] is not None


def test_send_webhook_retry_on_server_error(webhook_server):
    url, handler = webhook_server
    handler.status_to_return = 500
    delivery = send_webhook(url, {"test": True}, max_retries=2, timeout=2.0)
    assert delivery["success"] is False
    assert delivery["attempts"] == 2
    assert len(handler.received) == 2  # two attempts


def test_send_webhooks_multiple(webhook_server):
    url, handler = webhook_server
    result = {"foo": "bar"}
    deliveries = send_webhooks([url, url], result)
    assert len(deliveries) == 2
    assert all(d["success"] for d in deliveries)
    assert len(handler.received) == 2


def test_compute_signature():
    payload = b'{"key": "value"}'
    secret = "test-secret"
    sig = compute_signature(payload, secret)
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    assert sig == expected


def test_format_webhook_summary():
    deliveries = [
        {"url": "http://a.com", "success": True, "status_code": 200, "attempts": 1, "error": None},
        {
            "url": "http://b.com", "success": False, "status_code": None,
            "attempts": 3, "error": "timeout",
        },
    ]
    summary = format_webhook_summary(deliveries)
    assert "✓ http://a.com" in summary
    assert "✗ http://b.com" in summary
    assert "timeout" in summary


def test_webhook_yaml_config_known_keys():
    """webhook_url and webhook_secret should be in known YAML keys."""
    from xpyd_bench.config_cmd import _KNOWN_KEYS

    assert "webhook_url" in _KNOWN_KEYS
    assert "webhook_secret" in _KNOWN_KEYS


def test_webhook_cli_args():
    """CLI should accept --webhook-url and --webhook-secret."""
    import argparse

    from xpyd_bench.cli import _add_vllm_compat_args

    parser = argparse.ArgumentParser()
    _add_vllm_compat_args(parser)
    args = parser.parse_args([
        "--webhook-url", "http://a.com/hook",
        "--webhook-url", "http://b.com/hook",
        "--webhook-secret", "s3cret",
    ])
    assert args.webhook_url == ["http://a.com/hook", "http://b.com/hook"]
    assert args.webhook_secret == "s3cret"

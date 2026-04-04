"""Tests for M13: Custom HTTP Headers."""

from __future__ import annotations

import asyncio
import json
from argparse import Namespace

import httpx
import pytest
import uvicorn
import yaml

from xpyd_bench.cli import _parse_header, _resolve_custom_headers
from xpyd_bench.dummy.server import ServerConfig, create_app, set_config

# ---------------------------------------------------------------------------
# CLI parsing tests
# ---------------------------------------------------------------------------


class TestParseHeader:
    """Test _parse_header helper."""

    def test_simple(self):
        assert _parse_header("X-Custom: value") == ("X-Custom", "value")

    def test_strips_whitespace(self):
        assert _parse_header("  X-Key :  some value  ") == ("X-Key", "some value")

    def test_value_with_colons(self):
        k, v = _parse_header("X-Data: a:b:c")
        assert k == "X-Data"
        assert v == "a:b:c"

    def test_missing_colon_raises(self):
        with pytest.raises(ValueError, match="Invalid header format"):
            _parse_header("NoColonHere")


class TestResolveCustomHeaders:
    """Test _resolve_custom_headers merging logic."""

    def test_empty(self):
        args = Namespace(header=None, headers=None)
        assert _resolve_custom_headers(args) == {}

    def test_yaml_only(self):
        args = Namespace(header=None, headers={"X-From": "yaml", "X-Other": "v"})
        result = _resolve_custom_headers(args)
        assert result == {"X-From": "yaml", "X-Other": "v"}

    def test_cli_only(self):
        args = Namespace(header=["X-Cli: value1", "X-Cli2: value2"], headers=None)
        result = _resolve_custom_headers(args)
        assert result == {"X-Cli": "value1", "X-Cli2": "value2"}

    def test_cli_overrides_yaml(self):
        args = Namespace(
            header=["X-Key: from-cli"],
            headers={"X-Key": "from-yaml", "X-Only-Yaml": "y"},
        )
        result = _resolve_custom_headers(args)
        assert result["X-Key"] == "from-cli"
        assert result["X-Only-Yaml"] == "y"


# ---------------------------------------------------------------------------
# YAML config integration
# ---------------------------------------------------------------------------


class TestYamlHeaders:
    """Test that YAML config headers are loaded correctly."""

    def test_yaml_config_headers(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump({"headers": {"X-Yaml-Header": "yaml-value"}})
        )

        from xpyd_bench.cli import _load_yaml_config

        args = Namespace(
            header=None,
            headers=None,
            base_url="http://localhost:8000",
        )
        args = _load_yaml_config(str(config_file), args)
        assert args.headers == {"X-Yaml-Header": "yaml-value"}


# ---------------------------------------------------------------------------
# Dummy server echo tests
# ---------------------------------------------------------------------------

_DUMMY_PORT = 18923


@pytest.fixture(scope="module")
def dummy_server():
    """Start a dummy server for header echo tests."""
    set_config(ServerConfig(prefill_ms=1, decode_ms=1, model_name="echo-test"))
    app = create_app()

    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=_DUMMY_PORT, log_level="error")
    )

    loop = asyncio.new_event_loop()
    import threading

    thread = threading.Thread(target=loop.run_until_complete, args=(server.serve(),), daemon=True)
    thread.start()

    # Wait for server to start
    import time

    for _ in range(50):
        try:
            httpx.get(f"http://127.0.0.1:{_DUMMY_PORT}/health")
            break
        except httpx.ConnectError:
            time.sleep(0.1)
    else:
        raise RuntimeError("Dummy server did not start")

    yield f"http://127.0.0.1:{_DUMMY_PORT}"

    server.should_exit = True
    thread.join(timeout=5)


class TestDummyServerEcho:
    """Test that the dummy server echoes custom headers."""

    def test_completions_echo(self, dummy_server):
        resp = httpx.post(
            f"{dummy_server}/v1/completions",
            json={"prompt": "hello", "max_tokens": 5, "model": "echo-test"},
            headers={"X-Custom-Trace": "abc123", "X-Route": "pool-a"},
        )
        assert resp.status_code == 200
        echo_raw = resp.headers.get("x-echo-headers")
        assert echo_raw is not None
        echoed = json.loads(echo_raw)
        assert echoed["x-custom-trace"] == "abc123"
        assert echoed["x-route"] == "pool-a"

    def test_chat_echo(self, dummy_server):
        resp = httpx.post(
            f"{dummy_server}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 5,
                "model": "echo-test",
            },
            headers={"X-Session-Id": "sess-42"},
        )
        assert resp.status_code == 200
        echo_raw = resp.headers.get("x-echo-headers")
        assert echo_raw is not None
        echoed = json.loads(echo_raw)
        assert echoed["x-session-id"] == "sess-42"

    def test_no_custom_headers_no_echo(self, dummy_server):
        resp = httpx.post(
            f"{dummy_server}/v1/completions",
            json={"prompt": "hello", "max_tokens": 5, "model": "echo-test"},
        )
        assert resp.status_code == 200
        # No custom headers → no echo header
        assert resp.headers.get("x-echo-headers") is None

    def test_streaming_echo(self, dummy_server):
        with httpx.stream(
            "POST",
            f"{dummy_server}/v1/completions",
            json={
                "prompt": "hello",
                "max_tokens": 3,
                "model": "echo-test",
                "stream": True,
            },
            headers={"X-Stream-Trace": "streamy"},
        ) as resp:
            assert resp.status_code == 200
            echo_raw = resp.headers.get("x-echo-headers")
            assert echo_raw is not None
            echoed = json.loads(echo_raw)
            assert echoed["x-stream-trace"] == "streamy"
            # Drain the stream
            for _ in resp.iter_lines():
                pass


# ---------------------------------------------------------------------------
# Runner integration: custom headers reach HTTP client
# ---------------------------------------------------------------------------


class TestRunnerHeaderInjection:
    """Verify custom headers are sent to the server."""

    def test_custom_headers_in_request(self, dummy_server):
        """Full integration: run bench with custom headers, verify echo."""
        resp = httpx.post(
            f"{dummy_server}/v1/completions",
            json={"prompt": "test", "max_tokens": 2, "model": "echo-test"},
            headers={
                "Authorization": "Bearer test-key",
                "X-Bench-Run": "run-99",
            },
        )
        assert resp.status_code == 200
        echo_raw = resp.headers.get("x-echo-headers")
        assert echo_raw is not None
        echoed = json.loads(echo_raw)
        # Authorization is standard, should NOT be echoed
        assert "authorization" not in echoed
        # Custom header should be echoed
        assert echoed["x-bench-run"] == "run-99"


class TestAuthPrecedence:
    """Test that explicit custom Authorization header overrides api_key."""

    def test_custom_auth_overrides_api_key(self):
        args = Namespace(
            header=["Authorization: Token custom-tok"],
            headers=None,
            api_key="bearer-key",
            custom_headers=None,
        )
        custom = _resolve_custom_headers(args)
        assert custom["Authorization"] == "Token custom-tok"

        # Simulate runner logic: custom_headers has Authorization,
        # so api_key's Bearer should NOT override
        headers: dict[str, str] = {}
        headers.update(custom)
        if args.api_key and "Authorization" not in custom:
            headers["Authorization"] = f"Bearer {args.api_key}"
        assert headers["Authorization"] == "Token custom-tok"

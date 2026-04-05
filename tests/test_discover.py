"""Tests for endpoint capability discovery (M64)."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import httpx
import pytest
import yaml

from xpyd_bench.discover import (
    EndpointCapabilities,
    discover_endpoint,
    discover_main,
    format_summary,
    generate_config,
)

# ---------------------------------------------------------------------------
# EndpointCapabilities dataclass
# ---------------------------------------------------------------------------


class TestEndpointCapabilities:
    def test_defaults(self):
        caps = EndpointCapabilities(base_url="http://localhost:8000")
        assert caps.reachable is False
        assert caps.models == []
        assert caps.completions is False
        assert caps.chat_completions is False
        assert caps.embeddings is False
        assert caps.streaming is False
        assert caps.function_calling is False
        assert caps.batch is False
        assert caps.errors == []

    def test_to_dict(self):
        caps = EndpointCapabilities(
            base_url="http://localhost:8000",
            reachable=True,
            models=["gpt-4"],
            chat_completions=True,
        )
        d = caps.to_dict()
        assert d["base_url"] == "http://localhost:8000"
        assert d["reachable"] is True
        assert d["models"] == ["gpt-4"]
        assert d["chat_completions"] is True


# ---------------------------------------------------------------------------
# discover_endpoint — mocked HTTP
# ---------------------------------------------------------------------------


def _mock_response(status_code: int = 200, json_data=None, headers=None):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.headers = headers or {}
    return resp


class TestDiscoverEndpoint:
    def test_unreachable(self):
        with patch("xpyd_bench.discover.httpx.Client") as MockClient:
            client = MockClient.return_value
            client.get.side_effect = httpx.ConnectError("refused")
            caps = discover_endpoint("http://localhost:9999", timeout=1.0)
            assert caps.reachable is False
            assert caps.models == []

    def test_full_capabilities(self):
        with patch("xpyd_bench.discover.httpx.Client") as MockClient:
            client = MockClient.return_value

            models_resp = _mock_response(
                200, {"data": [{"id": "model-a"}, {"id": "model-b"}]}
            )
            completions_resp = _mock_response(200)
            chat_resp = _mock_response(200)
            streaming_resp = _mock_response(
                200, headers={"content-type": "text/event-stream"}
            )
            embeddings_resp = _mock_response(200)
            fc_resp = _mock_response(200)
            batch_resp = _mock_response(400)

            def route_get(url, **kw):
                return models_resp

            def route_post(url, **kw):
                json_body = kw.get("json", {})
                if "/v1/completions" in url and "prompt" in json_body:
                    return completions_resp
                if "/v1/chat/completions" in url:
                    if json_body.get("stream"):
                        return streaming_resp
                    if "tools" in json_body:
                        return fc_resp
                    return chat_resp
                if "/v1/embeddings" in url:
                    return embeddings_resp
                if "/v1/batch" in url:
                    return batch_resp
                return _mock_response(404)

            client.get.side_effect = route_get
            client.post.side_effect = route_post

            caps = discover_endpoint("http://localhost:8000")
            assert caps.reachable is True
            assert caps.models == ["model-a", "model-b"]
            assert caps.completions is True
            assert caps.chat_completions is True
            assert caps.streaming is True
            assert caps.embeddings is True
            assert caps.function_calling is True
            assert caps.batch is True

    def test_partial_capabilities(self):
        with patch("xpyd_bench.discover.httpx.Client") as MockClient:
            client = MockClient.return_value

            models_resp = _mock_response(200, {"data": [{"id": "llama"}]})

            def route_get(url, **kw):
                return models_resp

            def route_post(url, **kw):
                json_body = kw.get("json", {})
                if "/v1/chat/completions" in url:
                    if not json_body.get("stream") and "tools" not in json_body:
                        return _mock_response(200)
                    return _mock_response(404)
                return _mock_response(404)

            client.get.side_effect = route_get
            client.post.side_effect = route_post

            caps = discover_endpoint("http://localhost:8000")
            assert caps.reachable is True
            assert caps.models == ["llama"]
            assert caps.chat_completions is True
            assert caps.completions is False
            assert caps.streaming is False
            assert caps.embeddings is False

    def test_api_key_passed(self):
        with patch("xpyd_bench.discover.httpx.Client") as MockClient:
            client = MockClient.return_value
            client.get.return_value = _mock_response(200, {"data": []})
            client.post.return_value = _mock_response(404)

            discover_endpoint("http://localhost:8000", api_key="sk-test")
            call_kwargs = MockClient.call_args
            assert call_kwargs[1]["headers"]["Authorization"] == "Bearer sk-test"

    def test_trailing_slash_stripped(self):
        with patch("xpyd_bench.discover.httpx.Client") as MockClient:
            client = MockClient.return_value
            client.get.return_value = _mock_response(200, {"data": []})
            client.post.return_value = _mock_response(404)

            caps = discover_endpoint("http://localhost:8000/")
            assert caps.base_url == "http://localhost:8000"


# ---------------------------------------------------------------------------
# generate_config
# ---------------------------------------------------------------------------


class TestGenerateConfig:
    def test_chat_endpoint_preferred(self):
        caps = EndpointCapabilities(
            base_url="http://localhost:8000",
            models=["gpt-4"],
            chat_completions=True,
            completions=True,
            streaming=True,
        )
        config = generate_config(caps)
        assert config["endpoint"] == "/v1/chat/completions"
        assert config["stream"] is True
        assert config["model"] == "gpt-4"

    def test_completions_fallback(self):
        caps = EndpointCapabilities(
            base_url="http://localhost:8000",
            models=["gpt-3"],
            completions=True,
            chat_completions=False,
        )
        config = generate_config(caps)
        assert config["endpoint"] == "/v1/completions"
        assert "stream" not in config

    def test_no_endpoints(self):
        caps = EndpointCapabilities(base_url="http://localhost:8000")
        config = generate_config(caps)
        assert "endpoint" not in config
        assert config["base_url"] == "http://localhost:8000"


# ---------------------------------------------------------------------------
# format_summary
# ---------------------------------------------------------------------------


class TestFormatSummary:
    def test_summary_contains_indicators(self):
        caps = EndpointCapabilities(
            base_url="http://localhost:8000",
            reachable=True,
            models=["gpt-4"],
            chat_completions=True,
            completions=False,
        )
        summary = format_summary(caps)
        assert "✓" in summary
        assert "✗" in summary
        assert "gpt-4" in summary
        assert "http://localhost:8000" in summary

    def test_summary_errors(self):
        caps = EndpointCapabilities(
            base_url="http://localhost:8000",
            errors=["something failed"],
        )
        summary = format_summary(caps)
        assert "something failed" in summary

    def test_summary_no_models(self):
        caps = EndpointCapabilities(base_url="http://localhost:8000")
        summary = format_summary(caps)
        assert "none detected" in summary


# ---------------------------------------------------------------------------
# CLI (discover_main)
# ---------------------------------------------------------------------------


class TestDiscoverMain:
    def test_json_output(self, capsys):
        with patch("xpyd_bench.discover.discover_endpoint") as mock_discover:
            mock_discover.return_value = EndpointCapabilities(
                base_url="http://localhost:8000",
                reachable=True,
                models=["m1"],
                chat_completions=True,
            )
            discover_main(["--base-url", "http://localhost:8000", "--json"])
            out = capsys.readouterr().out
            data = json.loads(out)
            assert data["reachable"] is True
            assert data["models"] == ["m1"]

    def test_human_output(self, capsys):
        with patch("xpyd_bench.discover.discover_endpoint") as mock_discover:
            mock_discover.return_value = EndpointCapabilities(
                base_url="http://localhost:8000",
                reachable=True,
                chat_completions=True,
            )
            discover_main(["--base-url", "http://localhost:8000"])
            out = capsys.readouterr().out
            assert "Endpoint Discovery" in out

    def test_generate_config_file(self, tmp_path, capsys):
        config_path = str(tmp_path / "bench.yaml")
        with patch("xpyd_bench.discover.discover_endpoint") as mock_discover:
            mock_discover.return_value = EndpointCapabilities(
                base_url="http://localhost:8000",
                reachable=True,
                models=["gpt-4"],
                chat_completions=True,
                streaming=True,
            )
            discover_main([
                "--base-url", "http://localhost:8000",
                "--generate-config", config_path,
            ])
            assert os.path.exists(config_path)
            with open(config_path) as f:
                config = yaml.safe_load(f)
            assert config["model"] == "gpt-4"
            assert config["stream"] is True

    def test_unreachable_exit_code(self):
        with patch("xpyd_bench.discover.discover_endpoint") as mock_discover:
            mock_discover.return_value = EndpointCapabilities(
                base_url="http://localhost:8000", reachable=False
            )
            with pytest.raises(SystemExit) as exc_info:
                discover_main(["--base-url", "http://localhost:8000"])
            assert exc_info.value.code == 1

    def test_api_key_env_fallback(self, monkeypatch, capsys):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")
        with patch("xpyd_bench.discover.discover_endpoint") as mock_discover:
            mock_discover.return_value = EndpointCapabilities(
                base_url="http://localhost:8000", reachable=True
            )
            discover_main(["--base-url", "http://localhost:8000"])
            call_kwargs = mock_discover.call_args
            assert call_kwargs[1]["api_key"] == "sk-env-key"

    def test_timeout_passed(self, capsys):
        with patch("xpyd_bench.discover.discover_endpoint") as mock_discover:
            mock_discover.return_value = EndpointCapabilities(
                base_url="http://localhost:8000", reachable=True
            )
            discover_main([
                "--base-url", "http://localhost:8000",
                "--timeout", "5.0",
            ])
            call_kwargs = mock_discover.call_args
            assert call_kwargs[1]["timeout"] == 5.0

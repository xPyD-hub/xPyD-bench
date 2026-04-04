"""Tests for M40: Request Payload Compression."""

from __future__ import annotations

import gzip
import json

import pytest

from xpyd_bench.bench.debug_log import DebugLogEntry, DebugLogger
from xpyd_bench.bench.runner import _compressed_request_kwargs

# ---------------------------------------------------------------------------
# Unit tests for _compressed_request_kwargs
# ---------------------------------------------------------------------------


class TestCompressedRequestKwargs:
    def test_no_compress_returns_json(self):
        payload = {"model": "test", "prompt": "hello"}
        kw = _compressed_request_kwargs(payload, compress=False)
        assert kw == {"json": payload}

    def test_compress_returns_content_and_headers(self):
        payload = {"model": "test", "prompt": "hello"}
        kw = _compressed_request_kwargs(payload, compress=True)
        assert "content" in kw
        assert "headers" in kw
        assert kw["headers"]["Content-Encoding"] == "gzip"
        assert kw["headers"]["Content-Type"] == "application/json"
        # Verify content is valid gzip of the JSON payload
        decompressed = gzip.decompress(kw["content"])
        assert json.loads(decompressed) == payload

    def test_compress_large_payload(self):
        payload = {"model": "test", "prompt": "a" * 10000}
        kw = _compressed_request_kwargs(payload, compress=True)
        raw = json.dumps(payload).encode()
        compressed = kw["content"]
        # Compression should reduce size for repetitive data
        assert len(compressed) < len(raw)
        assert json.loads(gzip.decompress(compressed)) == payload


# ---------------------------------------------------------------------------
# CLI flag parsing
# ---------------------------------------------------------------------------


class TestCompressCLI:
    def test_compress_flag_default(self):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args([])
        assert args.compress is False

    def test_compress_flag_enabled(self):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--compress"])
        assert args.compress is True

    def test_compress_yaml_config(self, tmp_path):
        import argparse

        import yaml

        from xpyd_bench.cli import (
            _add_vllm_compat_args,
            _get_explicit_keys,
            _load_yaml_config,
        )

        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml.dump({"compress": True}))

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args([])
        explicit = _get_explicit_keys(parser, args)
        args = _load_yaml_config(str(yaml_file), args, explicit_keys=explicit)
        assert args.compress is True


# ---------------------------------------------------------------------------
# Debug log payload size tracking
# ---------------------------------------------------------------------------


class TestDebugLogCompression:
    def test_entry_without_compression(self):
        entry = DebugLogEntry(
            timestamp="2025-01-01T00:00:00",
            url="http://localhost/v1/completions",
            payload='{"prompt":"hi"}',
            status="ok",
            latency_ms=10.0,
            success=True,
        )
        d = entry.to_dict()
        assert "payload_bytes" not in d
        assert "compressed_bytes" not in d

    def test_entry_with_compression(self):
        entry = DebugLogEntry(
            timestamp="2025-01-01T00:00:00",
            url="http://localhost/v1/completions",
            payload='{"prompt":"hi"}',
            status="ok",
            latency_ms=10.0,
            success=True,
            payload_bytes=500,
            compressed_bytes=120,
        )
        d = entry.to_dict()
        assert d["payload_bytes"] == 500
        assert d["compressed_bytes"] == 120

    def test_debug_logger_with_sizes(self, tmp_path):
        from xpyd_bench.bench.models import RequestResult

        log_path = tmp_path / "debug.jsonl"
        logger = DebugLogger(log_path)
        result = RequestResult()
        result.latency_ms = 5.0
        result.success = True
        logger.log(
            url="http://localhost/v1/completions",
            payload={"prompt": "test"},
            result=result,
            payload_bytes=200,
            compressed_bytes=80,
        )
        logger.close()

        lines = log_path.read_text().strip().split("\n")
        entry = json.loads(lines[0])
        assert entry["payload_bytes"] == 200
        assert entry["compressed_bytes"] == 80


# ---------------------------------------------------------------------------
# Dummy server gzip decompression
# ---------------------------------------------------------------------------


class TestDummyServerGzip:
    @pytest.fixture
    def dummy_app(self):
        from xpyd_bench.dummy.server import create_app

        return create_app()

    def test_dummy_server_accepts_gzip(self, dummy_app):
        """Test that the dummy server correctly decompresses gzip request bodies."""
        from starlette.testclient import TestClient

        client = TestClient(dummy_app)
        payload = {
            "model": "test-model",
            "prompt": "Hello, world!",
            "max_tokens": 10,
        }
        raw = json.dumps(payload).encode()
        compressed = gzip.compress(raw)

        resp = client.post(
            "/v1/completions",
            content=compressed,
            headers={
                "Content-Encoding": "gzip",
                "Content-Type": "application/json",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "choices" in body

    def test_dummy_server_uncompressed_still_works(self, dummy_app):
        """Ensure plain JSON still works (no regression)."""
        from starlette.testclient import TestClient

        client = TestClient(dummy_app)
        payload = {
            "model": "test-model",
            "prompt": "Hello!",
            "max_tokens": 5,
        }
        resp = client.post("/v1/completions", json=payload)
        assert resp.status_code == 200
        assert "choices" in resp.json()

    def test_dummy_chat_accepts_gzip(self, dummy_app):
        """Test chat endpoint with gzip."""
        from starlette.testclient import TestClient

        client = TestClient(dummy_app)
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5,
        }
        raw = json.dumps(payload).encode()
        compressed = gzip.compress(raw)

        resp = client.post(
            "/v1/chat/completions",
            content=compressed,
            headers={
                "Content-Encoding": "gzip",
                "Content-Type": "application/json",
            },
        )
        assert resp.status_code == 200
        assert "choices" in resp.json()

    def test_dummy_embeddings_accepts_gzip(self, dummy_app):
        """Test embeddings endpoint with gzip."""
        from starlette.testclient import TestClient

        client = TestClient(dummy_app)
        payload = {
            "model": "test-model",
            "input": "Hello embeddings",
        }
        raw = json.dumps(payload).encode()
        compressed = gzip.compress(raw)

        resp = client.post(
            "/v1/embeddings",
            content=compressed,
            headers={
                "Content-Encoding": "gzip",
                "Content-Type": "application/json",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "data" in body

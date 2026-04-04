"""Tests for M31: Plugin Architecture for Custom Backends."""

from __future__ import annotations

from argparse import Namespace
from typing import Any

import pytest

from xpyd_bench.bench.models import RequestResult
from xpyd_bench.plugins import BackendPlugin, PluginRegistry, registry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class DummyPlugin(BackendPlugin):
    """Minimal plugin for testing."""

    @property
    def name(self) -> str:
        return "dummy-test"

    def build_payload(
        self, args: Namespace, prompt: str, *, is_chat: bool = False, is_embeddings: bool = False
    ) -> dict[str, Any]:
        return {"prompt": prompt, "custom": True}

    async def send_request(
        self,
        client: Any,
        url: str,
        payload: dict[str, Any],
        *,
        is_streaming: bool = False,
        request_timeout: float = 300.0,
        retries: int = 0,
        retry_delay: float = 1.0,
    ) -> RequestResult:
        r = RequestResult()
        r.latency_ms = 42.0
        r.success = True
        r.completion_tokens = 10
        return r


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestPluginRegistry:
    def test_register_and_get(self) -> None:
        reg = PluginRegistry()
        p = DummyPlugin()
        reg.register(p)
        assert reg.get("dummy-test") is p

    def test_get_unknown_raises(self) -> None:
        reg = PluginRegistry()
        with pytest.raises(KeyError, match="Unknown backend"):
            reg.get("nonexistent")

    def test_list_backends(self) -> None:
        reg = PluginRegistry()
        reg.register(DummyPlugin())
        assert "dummy-test" in reg.list_backends()

    def test_register_overwrites(self) -> None:
        reg = PluginRegistry()
        p1 = DummyPlugin()
        p2 = DummyPlugin()
        reg.register(p1)
        reg.register(p2)
        assert reg.get("dummy-test") is p2


class TestGlobalRegistry:
    def test_openai_builtin(self) -> None:
        """Built-in OpenAI plugin should be registered by default."""
        assert "openai" in registry.list_backends()
        p = registry.get("openai")
        assert p.name == "openai"


# ---------------------------------------------------------------------------
# Plugin interface tests
# ---------------------------------------------------------------------------


class TestBackendPluginInterface:
    def test_build_payload(self) -> None:
        p = DummyPlugin()
        args = Namespace(model="test")
        payload = p.build_payload(args, "hello world")
        assert payload == {"prompt": "hello world", "custom": True}

    @pytest.mark.asyncio
    async def test_send_request(self) -> None:
        p = DummyPlugin()
        result = await p.send_request(None, "http://localhost/test", {"prompt": "hi"})
        assert result.success is True
        assert result.latency_ms == 42.0
        assert result.completion_tokens == 10

    def test_build_url_default(self) -> None:
        p = DummyPlugin()
        args = Namespace(endpoint="/v1/completions")
        url = p.build_url("http://localhost:8000", args)
        assert url == "http://localhost:8000/v1/completions"


# ---------------------------------------------------------------------------
# Module plugin loading
# ---------------------------------------------------------------------------


class TestLoadModulePlugin:
    def test_load_vllm_example(self) -> None:
        reg = PluginRegistry()
        plugin = reg.load_module_plugin("xpyd_bench.plugins.examples.vllm_native")
        assert plugin.name == "vllm-native"
        assert "vllm-native" in reg.list_backends()

    def test_load_nonexistent_module(self) -> None:
        reg = PluginRegistry()
        with pytest.raises(ImportError):
            reg.load_module_plugin("nonexistent.module.path")

    def test_load_module_without_plugin(self) -> None:
        reg = PluginRegistry()
        with pytest.raises(ImportError, match="does not expose"):
            reg.load_module_plugin("json")  # json has no plugin/Plugin


# ---------------------------------------------------------------------------
# vLLM native example plugin tests
# ---------------------------------------------------------------------------


class TestVLLMNativePlugin:
    def test_name(self) -> None:
        from xpyd_bench.plugins.examples.vllm_native import Plugin

        p = Plugin()
        assert p.name == "vllm-native"

    def test_build_url(self) -> None:
        from xpyd_bench.plugins.examples.vllm_native import Plugin

        p = Plugin()
        args = Namespace(endpoint="/v1/completions")
        url = p.build_url("http://localhost:8000", args)
        assert url == "http://localhost:8000/generate"

    def test_build_payload(self) -> None:
        from xpyd_bench.plugins.examples.vllm_native import Plugin

        p = Plugin()
        args = Namespace(
            model="my-model",
            output_len=64,
            temperature=0.5,
            top_p=None,
            top_k=None,
            stream=False,
        )
        payload = p.build_payload(args, "hello")
        assert payload == {
            "prompt": "hello",
            "max_tokens": 64,
            "model": "my-model",
            "temperature": 0.5,
        }

    def test_build_payload_streaming(self) -> None:
        from xpyd_bench.plugins.examples.vllm_native import Plugin

        p = Plugin()
        args = Namespace(
            model="m", output_len=10, temperature=None, top_p=None, top_k=None, stream=True
        )
        payload = p.build_payload(args, "test")
        assert payload["stream"] is True


# ---------------------------------------------------------------------------
# OpenAI backend plugin tests
# ---------------------------------------------------------------------------


class TestOpenAIBackendPlugin:
    def test_name(self) -> None:
        from xpyd_bench.plugins.openai_backend import OpenAIBackendPlugin

        p = OpenAIBackendPlugin()
        assert p.name == "openai"

    def test_build_payload_delegates(self) -> None:
        from xpyd_bench.plugins.openai_backend import OpenAIBackendPlugin

        p = OpenAIBackendPlugin()
        args = Namespace(
            model="gpt-4",
            output_len=50,
            temperature=None,
            top_p=None,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            best_of=None,
            use_beam_search=False,
            logprobs=None,
            ignore_eos=False,
        )
        payload = p.build_payload(args, "hi", is_chat=False)
        assert payload["prompt"] == "hi"
        assert payload["max_tokens"] == 50
        assert payload["model"] == "gpt-4"


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


class TestCLIIntegration:
    def test_list_backends_flag(self, capsys: pytest.CaptureFixture) -> None:
        from xpyd_bench.cli import bench_main

        bench_main(["--list-backends"])
        out = capsys.readouterr().out
        assert "openai" in out
        assert "Available backends" in out

    def test_backend_plugin_flag_unknown(self) -> None:
        """--backend-plugin with nonexistent module should fail."""
        from xpyd_bench.cli import bench_main

        with pytest.raises((ImportError, SystemExit)):
            bench_main([
                "--backend-plugin", "nonexistent_module_xyz",
                "--backend", "nonexistent",
                "--base-url", "http://localhost:9999",
                "--num-prompts", "1",
                "--disable-tqdm",
            ])

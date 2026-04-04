"""Tests for CLI argument parsing and vLLM compatibility."""

from __future__ import annotations

import subprocess


def test_bench_help():
    """xpyd-bench --help should exit 0 and show vLLM-compatible args."""
    result = subprocess.run(
        ["xpyd-bench", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "xpyd-bench" in result.stdout
    # vLLM-compatible arguments should be present
    assert "--backend" in result.stdout
    assert "--base-url" in result.stdout
    assert "--request-rate" in result.stdout
    assert "--num-prompts" in result.stdout
    assert "--model" in result.stdout
    assert "--save-result" in result.stdout
    assert "--endpoint" in result.stdout


def test_cli_default_args():
    """CLI parses default arguments correctly."""
    import argparse

    from xpyd_bench.cli import _add_vllm_compat_args

    parser = argparse.ArgumentParser()
    _add_vllm_compat_args(parser)
    args = parser.parse_args([])

    assert args.backend == "openai"
    assert args.base_url is None
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.endpoint == "/v1/completions"
    assert args.num_prompts == 1000
    assert args.request_rate == float("inf")
    assert args.burstiness == 1.0
    assert args.max_concurrency is None
    assert args.input_len == 256
    assert args.output_len == 128
    assert args.seed == 0
    assert args.temperature is None
    assert args.top_p is None
    assert args.save_result is False
    assert args.disable_tqdm is False


def test_cli_custom_args():
    """CLI parses custom arguments correctly."""
    import argparse

    from xpyd_bench.cli import _add_vllm_compat_args

    parser = argparse.ArgumentParser()
    _add_vllm_compat_args(parser)
    args = parser.parse_args([
        "--backend", "openai-chat",
        "--base-url", "http://myserver:9000",
        "--endpoint", "/v1/chat/completions",
        "--model", "llama-3",
        "--num-prompts", "500",
        "--request-rate", "10.0",
        "--burstiness", "0.5",
        "--max-concurrency", "32",
        "--input-len", "512",
        "--output-len", "256",
        "--seed", "42",
        "--temperature", "0.8",
        "--top-p", "0.95",
        "--top-k", "50",
        "--frequency-penalty", "0.1",
        "--presence-penalty", "0.2",
        "--best-of", "3",
        "--use-beam-search",
        "--logprobs", "5",
        "--save-result",
        "--result-dir", "/tmp/results",
        "--result-filename", "test.json",
        "--disable-tqdm",
        "--ignore-eos",
    ])

    assert args.backend == "openai-chat"
    assert args.base_url == "http://myserver:9000"
    assert args.endpoint == "/v1/chat/completions"
    assert args.model == "llama-3"
    assert args.num_prompts == 500
    assert args.request_rate == 10.0
    assert args.burstiness == 0.5
    assert args.max_concurrency == 32
    assert args.input_len == 512
    assert args.output_len == 256
    assert args.seed == 42
    assert args.temperature == 0.8
    assert args.top_p == 0.95
    assert args.top_k == 50
    assert args.frequency_penalty == 0.1
    assert args.presence_penalty == 0.2
    assert args.best_of == 3
    assert args.use_beam_search is True
    assert args.logprobs == 5
    assert args.save_result is True
    assert args.result_dir == "/tmp/results"
    assert args.result_filename == "test.json"
    assert args.disable_tqdm is True
    assert args.ignore_eos is True


def test_resolve_base_url_explicit():
    """--base-url takes precedence over --host/--port."""
    import argparse

    from xpyd_bench.cli import _resolve_base_url

    ns = argparse.Namespace(base_url="http://custom:1234/", host="x", port=0)
    assert _resolve_base_url(ns) == "http://custom:1234"


def test_resolve_base_url_from_host_port():
    """When --base-url is None, construct from --host/--port."""
    import argparse

    from xpyd_bench.cli import _resolve_base_url

    ns = argparse.Namespace(base_url=None, host="10.0.0.1", port=8080)
    assert _resolve_base_url(ns) == "http://10.0.0.1:8080"

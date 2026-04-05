"""Tests for request dependency chains (M59)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from xpyd_bench.chain import (
    ChainResult,
    ChainStep,
    ChainSummary,
    StepResult,
    _extract_jsonpath,
    _extract_regex,
    chain_main,
    compute_chain_summary,
    extract_value,
    load_chain,
    render_template,
    run_chain,
    run_chains,
)

# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


class TestExtractJsonpath:
    def test_simple_key(self):
        assert _extract_jsonpath({"foo": "bar"}, "foo") == "bar"

    def test_nested_key(self):
        data = {"choices": [{"text": "hello"}]}
        assert _extract_jsonpath(data, "choices.0.text") == "hello"

    def test_missing_key(self):
        assert _extract_jsonpath({"a": 1}, "b") == ""

    def test_array_out_of_bounds(self):
        assert _extract_jsonpath([1, 2], "5") == ""

    def test_deep_nesting(self):
        data = {"a": {"b": {"c": "deep"}}}
        assert _extract_jsonpath(data, "a.b.c") == "deep"

    def test_numeric_value(self):
        assert _extract_jsonpath({"count": 42}, "count") == "42"

    def test_none_value(self):
        assert _extract_jsonpath({"x": None}, "x") == ""


class TestExtractRegex:
    def test_group_match(self):
        assert _extract_regex("id: 12345 end", r"id: (\d+)") == "12345"

    def test_no_group_match(self):
        assert _extract_regex("hello world", r"hello") == "hello"

    def test_no_match(self):
        assert _extract_regex("abc", r"\d+") == ""


class TestExtractValue:
    def test_jsonpath_prefix(self):
        body = json.dumps({"choices": [{"text": "result"}]})
        assert extract_value(body, "jsonpath:choices.0.text") == "result"

    def test_regex_prefix(self):
        body = "The answer is 42."
        assert extract_value(body, "regex:answer is (\\d+)") == "42"

    def test_default_jsonpath(self):
        body = json.dumps({"key": "val"})
        assert extract_value(body, "key") == "val"

    def test_invalid_json_jsonpath(self):
        assert extract_value("not json", "jsonpath:foo") == ""

    def test_invalid_json_default(self):
        assert extract_value("not json", "foo") == ""


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


class TestRenderTemplate:
    def test_basic(self):
        assert render_template("Hello {{ name }}", {"name": "world"}) == "Hello world"

    def test_no_spaces(self):
        assert render_template("Hello {{name}}", {"name": "world"}) == "Hello world"

    def test_multiple_vars(self):
        t = "{{ a }} and {{ b }}"
        assert render_template(t, {"a": "X", "b": "Y"}) == "X and Y"

    def test_missing_var(self):
        assert render_template("{{ missing }}", {}) == "{{ missing }}"

    def test_empty_vars(self):
        assert render_template("no vars", {}) == "no vars"


# ---------------------------------------------------------------------------
# Chain loading
# ---------------------------------------------------------------------------


class TestLoadChain:
    def test_load_basic(self, tmp_path: Path):
        chain_file = tmp_path / "chain.jsonl"
        lines = [
            json.dumps({"prompt_template": "Step 0: hello"}),
            json.dumps({
                "prompt_template": "Step 1: {{ answer }}",
                "extract": {"answer": "jsonpath:choices.0.text"},
            }),
        ]
        chain_file.write_text("\n".join(lines))

        steps = load_chain(str(chain_file))
        assert len(steps) == 2
        assert steps[0].step_index == 0
        assert steps[0].prompt_template == "Step 0: hello"
        assert steps[0].extract == {}
        assert steps[1].step_index == 1
        assert steps[1].extract == {"answer": "jsonpath:choices.0.text"}

    def test_load_with_optional_fields(self, tmp_path: Path):
        chain_file = tmp_path / "chain.jsonl"
        line = json.dumps({
            "prompt_template": "test",
            "endpoint": "/v1/chat/completions",
            "model": "gpt-4",
            "max_tokens": 256,
        })
        chain_file.write_text(line)

        steps = load_chain(str(chain_file))
        assert steps[0].endpoint == "/v1/chat/completions"
        assert steps[0].model == "gpt-4"
        assert steps[0].max_tokens == 256


# ---------------------------------------------------------------------------
# Data model serialization
# ---------------------------------------------------------------------------


class TestDataModels:
    def test_step_result_to_dict(self):
        sr = StepResult(step_index=0, latency_ms=10.123, extracted={"k": "v"})
        d = sr.to_dict()
        assert d["step_index"] == 0
        assert d["latency_ms"] == 10.12
        assert d["extracted"] == {"k": "v"}
        assert "error" not in d

    def test_step_result_with_error(self):
        sr = StepResult(step_index=1, success=False, error="timeout")
        d = sr.to_dict()
        assert d["error"] == "timeout"

    def test_chain_result_to_dict(self):
        cr = ChainResult(
            chain_id=0,
            total_latency_ms=50.5,
            total_steps=2,
            successful_steps=2,
            steps=[StepResult(step_index=0), StepResult(step_index=1)],
        )
        d = cr.to_dict()
        assert d["chain_id"] == 0
        assert d["total_steps"] == 2
        assert len(d["steps"]) == 2

    def test_chain_summary_to_dict(self):
        cs = ChainSummary(total_chains=1, completed_chains=1, mean_chain_latency_ms=50.0)
        d = cs.to_dict()
        assert d["total_chains"] == 1


# ---------------------------------------------------------------------------
# Chain execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_chain_success():
    """Test successful chain execution with extraction."""
    steps = [
        ChainStep(
            step_index=0,
            prompt_template="What is 2+2?",
            extract={"answer": "jsonpath:choices.0.text"},
        ),
        ChainStep(
            step_index=1,
            prompt_template="The answer was {{ answer }}. What is next?",
        ),
    ]

    response_bodies = [
        json.dumps({
            "choices": [{"text": "4"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        }),
        json.dumps({
            "choices": [{"text": "5"}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 1},
        }),
    ]

    call_count = 0

    async def mock_post(url, json=None, headers=None, timeout=None):
        nonlocal call_count
        resp = MagicMock()
        resp.status_code = 200
        resp.text = response_bodies[call_count]
        resp.json.return_value = json_mod.loads(response_bodies[call_count])
        resp.raise_for_status = MagicMock()
        call_count += 1
        return resp

    import json as json_mod

    client = AsyncMock()
    client.post = mock_post

    result = await run_chain(
        client=client,
        base_url="http://localhost:8000",
        steps=steps,
        chain_id=0,
        default_model="test",
    )

    assert result.total_steps == 2
    assert result.successful_steps == 2
    assert result.failed_step is None
    assert result.steps[0].extracted == {"answer": "4"}
    assert result.steps[1].success is True


@pytest.mark.asyncio
async def test_run_chain_failure_stops():
    """Test that chain stops at first failure."""
    steps = [
        ChainStep(step_index=0, prompt_template="step 0"),
        ChainStep(step_index=1, prompt_template="step 1"),
        ChainStep(step_index=2, prompt_template="step 2"),
    ]

    call_count = 0

    async def mock_post(url, json=None, headers=None, timeout=None):
        nonlocal call_count
        if call_count == 1:
            call_count += 1
            raise httpx.ConnectError("connection refused")
        resp = MagicMock()
        resp.status_code = 200
        resp.text = '{"choices": [{"text": "ok"}], "usage": {}}'
        resp.json.return_value = {"choices": [{"text": "ok"}], "usage": {}}
        resp.raise_for_status = MagicMock()
        call_count += 1
        return resp

    client = AsyncMock()
    client.post = mock_post

    result = await run_chain(
        client=client,
        base_url="http://localhost:8000",
        steps=steps,
        chain_id=0,
    )

    assert result.failed_step == 1
    assert result.successful_steps == 1
    assert result.total_steps == 2  # only 2 steps attempted (0 ok, 1 failed, 2 skipped)


@pytest.mark.asyncio
async def test_run_chain_chat_endpoint():
    """Test chain with chat completions endpoint."""
    steps = [
        ChainStep(
            step_index=0,
            prompt_template="hello",
            endpoint="/v1/chat/completions",
        ),
    ]

    captured_payload = {}

    async def mock_post(url, json=None, headers=None, timeout=None):
        captured_payload.update(json or {})
        resp = MagicMock()
        resp.status_code = 200
        resp.text = '{"choices": [{"message": {"content": "hi"}}], "usage": {}}'
        resp.json.return_value = {"choices": [{"message": {"content": "hi"}}], "usage": {}}
        resp.raise_for_status = MagicMock()
        return resp

    client = AsyncMock()
    client.post = mock_post

    await run_chain(
        client=client,
        base_url="http://localhost:8000",
        steps=steps,
    )

    assert "messages" in captured_payload
    assert captured_payload["messages"][0]["role"] == "user"


@pytest.mark.asyncio
async def test_run_chain_with_api_key():
    """Test that API key is injected into headers."""
    steps = [ChainStep(step_index=0, prompt_template="test")]

    captured_headers = {}

    async def mock_post(url, json=None, headers=None, timeout=None):
        captured_headers.update(headers or {})
        resp = MagicMock()
        resp.status_code = 200
        resp.text = '{"choices": [{"text": "ok"}], "usage": {}}'
        resp.json.return_value = {"choices": [{"text": "ok"}], "usage": {}}
        resp.raise_for_status = MagicMock()
        return resp

    client = AsyncMock()
    client.post = mock_post

    await run_chain(
        client=client,
        base_url="http://localhost:8000",
        steps=steps,
        api_key="sk-test-123",
    )

    assert captured_headers.get("Authorization") == "Bearer sk-test-123"


# ---------------------------------------------------------------------------
# Compute summary
# ---------------------------------------------------------------------------


class TestComputeChainSummary:
    def test_all_success(self):
        chains = [
            ChainResult(
                chain_id=0,
                total_latency_ms=100.0,
                total_steps=2,
                successful_steps=2,
                steps=[
                    StepResult(step_index=0, latency_ms=40.0),
                    StepResult(step_index=1, latency_ms=60.0),
                ],
            ),
            ChainResult(
                chain_id=1,
                total_latency_ms=120.0,
                total_steps=2,
                successful_steps=2,
                steps=[
                    StepResult(step_index=0, latency_ms=50.0),
                    StepResult(step_index=1, latency_ms=70.0),
                ],
            ),
        ]

        summary = compute_chain_summary(chains)
        assert summary.total_chains == 2
        assert summary.completed_chains == 2
        assert summary.mean_chain_latency_ms == 110.0
        assert 0 in summary.per_step_stats
        assert 1 in summary.per_step_stats
        assert summary.per_step_stats[0]["count"] == 2
        assert summary.per_step_stats[0]["mean_latency_ms"] == 45.0

    def test_with_failure(self):
        chains = [
            ChainResult(
                chain_id=0,
                total_latency_ms=50.0,
                total_steps=1,
                successful_steps=0,
                failed_step=0,
                steps=[
                    StepResult(step_index=0, latency_ms=50.0, success=False, error="err"),
                ],
            ),
        ]

        summary = compute_chain_summary(chains)
        assert summary.total_chains == 1
        assert summary.completed_chains == 0
        # Failed step shouldn't appear in per_step_stats
        assert 0 not in summary.per_step_stats

    def test_empty(self):
        summary = compute_chain_summary([])
        assert summary.total_chains == 0
        assert summary.mean_chain_latency_ms == 0.0


# ---------------------------------------------------------------------------
# run_chains (multi-chain)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_chains():
    """Test running multiple chains."""
    steps = [ChainStep(step_index=0, prompt_template="hello")]

    with patch("xpyd_bench.chain.run_chain") as mock_run:
        mock_run.return_value = ChainResult(
            chain_id=0,
            total_latency_ms=10.0,
            total_steps=1,
            successful_steps=1,
            steps=[StepResult(step_index=0, latency_ms=10.0)],
        )

        summary = await run_chains(
            base_url="http://localhost:8000",
            chains=[steps, steps, steps],
            default_model="test",
        )

    assert summary.total_chains == 3


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestChainCLI:
    def test_cli_basic(self, tmp_path: Path):
        """Test CLI parses args and runs."""
        chain_file = tmp_path / "chain.jsonl"
        chain_file.write_text(
            json.dumps({"prompt_template": "hello"}) + "\n"
        )
        output_file = tmp_path / "result.json"

        with patch("xpyd_bench.chain.run_chains") as mock_run:
            mock_run.return_value = ChainSummary(
                total_chains=1,
                completed_chains=1,
                mean_chain_latency_ms=10.0,
                chains=[
                    ChainResult(
                        chain_id=0,
                        total_latency_ms=10.0,
                        total_steps=1,
                        successful_steps=1,
                        steps=[StepResult(step_index=0, latency_ms=10.0)],
                    )
                ],
            )

            chain_main([
                "--chain", str(chain_file),
                "--base-url", "http://localhost:8000",
                "--model", "test-model",
                "--output", str(output_file),
                "--repeat", "1",
            ])

            mock_run.assert_called_once()

        # Check output file
        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert data["total_chains"] == 1

    def test_cli_custom_headers(self, tmp_path: Path):
        """Test CLI with custom headers."""
        chain_file = tmp_path / "chain.jsonl"
        chain_file.write_text(
            json.dumps({"prompt_template": "hello"}) + "\n"
        )

        with patch("xpyd_bench.chain.run_chains") as mock_run:
            mock_run.return_value = ChainSummary(
                total_chains=1,
                completed_chains=1,
                mean_chain_latency_ms=10.0,
                chains=[],
            )

            chain_main([
                "--chain", str(chain_file),
                "--header", "X-Custom: test-value",
            ])

            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["custom_headers"] == {"X-Custom": "test-value"}

    def test_cli_api_key_env(self, tmp_path: Path, monkeypatch):
        """Test CLI picks up OPENAI_API_KEY from environment."""
        chain_file = tmp_path / "chain.jsonl"
        chain_file.write_text(
            json.dumps({"prompt_template": "hello"}) + "\n"
        )

        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")

        with patch("xpyd_bench.chain.run_chains") as mock_run:
            mock_run.return_value = ChainSummary(
                total_chains=1,
                completed_chains=1,
                mean_chain_latency_ms=10.0,
                chains=[],
            )

            chain_main([
                "--chain", str(chain_file),
            ])

            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["api_key"] == "sk-env-key"


# ---------------------------------------------------------------------------
# Subcommand routing
# ---------------------------------------------------------------------------


class TestSubcommandRouting:
    def test_chain_in_subcommands(self):
        from xpyd_bench.main import _SUBCOMMANDS

        assert "chain" in _SUBCOMMANDS

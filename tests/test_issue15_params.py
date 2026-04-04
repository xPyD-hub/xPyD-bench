"""Tests for issue #15: echo, suffix, logit_bias, user parameters."""

from __future__ import annotations

from argparse import Namespace

from xpyd_bench.bench.runner import _build_payload


class TestEchoParam:
    def test_echo_included_when_true(self):
        args = Namespace(
            output_len=10, model="m", temperature=None, top_p=None, top_k=None,
            frequency_penalty=None, presence_penalty=None, best_of=None,
            use_beam_search=False, logprobs=None, ignore_eos=False,
            stop=None, n=None, api_seed=None, echo=True, suffix=None,
            logit_bias=None, user=None,
        )
        payload = _build_payload(args, "hello", is_chat=False)
        assert payload["echo"] is True

    def test_echo_omitted_when_false(self):
        args = Namespace(
            output_len=10, model="m", temperature=None, top_p=None, top_k=None,
            frequency_penalty=None, presence_penalty=None, best_of=None,
            use_beam_search=False, logprobs=None, ignore_eos=False,
            stop=None, n=None, api_seed=None, echo=False, suffix=None,
            logit_bias=None, user=None,
        )
        payload = _build_payload(args, "hello", is_chat=False)
        assert "echo" not in payload


class TestSuffixParam:
    def test_suffix_included(self):
        args = Namespace(
            output_len=10, model="m", temperature=None, top_p=None, top_k=None,
            frequency_penalty=None, presence_penalty=None, best_of=None,
            use_beam_search=False, logprobs=None, ignore_eos=False,
            stop=None, n=None, api_seed=None, echo=False, suffix="END",
            logit_bias=None, user=None,
        )
        payload = _build_payload(args, "hello", is_chat=False)
        assert payload["suffix"] == "END"

    def test_suffix_omitted_when_none(self):
        args = Namespace(
            output_len=10, model="m", temperature=None, top_p=None, top_k=None,
            frequency_penalty=None, presence_penalty=None, best_of=None,
            use_beam_search=False, logprobs=None, ignore_eos=False,
            stop=None, n=None, api_seed=None, echo=False, suffix=None,
            logit_bias=None, user=None,
        )
        payload = _build_payload(args, "hello", is_chat=False)
        assert "suffix" not in payload


class TestLogitBiasParam:
    def test_logit_bias_from_json_string(self):
        args = Namespace(
            output_len=10, model="m", temperature=None, top_p=None, top_k=None,
            frequency_penalty=None, presence_penalty=None, best_of=None,
            use_beam_search=False, logprobs=None, ignore_eos=False,
            stop=None, n=None, api_seed=None, echo=False, suffix=None,
            logit_bias='{"50256": -100}', user=None,
        )
        payload = _build_payload(args, "hello", is_chat=False)
        assert payload["logit_bias"] == {"50256": -100}

    def test_logit_bias_from_dict(self):
        args = Namespace(
            output_len=10, model="m", temperature=None, top_p=None, top_k=None,
            frequency_penalty=None, presence_penalty=None, best_of=None,
            use_beam_search=False, logprobs=None, ignore_eos=False,
            stop=None, n=None, api_seed=None, echo=False, suffix=None,
            logit_bias={"50256": -100}, user=None,
        )
        payload = _build_payload(args, "hello", is_chat=False)
        assert payload["logit_bias"] == {"50256": -100}

    def test_logit_bias_omitted_when_none(self):
        args = Namespace(
            output_len=10, model="m", temperature=None, top_p=None, top_k=None,
            frequency_penalty=None, presence_penalty=None, best_of=None,
            use_beam_search=False, logprobs=None, ignore_eos=False,
            stop=None, n=None, api_seed=None, echo=False, suffix=None,
            logit_bias=None, user=None,
        )
        payload = _build_payload(args, "hello", is_chat=False)
        assert "logit_bias" not in payload


class TestUserParam:
    def test_user_included(self):
        args = Namespace(
            output_len=10, model="m", temperature=None, top_p=None, top_k=None,
            frequency_penalty=None, presence_penalty=None, best_of=None,
            use_beam_search=False, logprobs=None, ignore_eos=False,
            stop=None, n=None, api_seed=None, echo=False, suffix=None,
            logit_bias=None, user="test-user-123",
        )
        payload = _build_payload(args, "hello", is_chat=False)
        assert payload["user"] == "test-user-123"

    def test_user_omitted_when_none(self):
        args = Namespace(
            output_len=10, model="m", temperature=None, top_p=None, top_k=None,
            frequency_penalty=None, presence_penalty=None, best_of=None,
            use_beam_search=False, logprobs=None, ignore_eos=False,
            stop=None, n=None, api_seed=None, echo=False, suffix=None,
            logit_bias=None, user=None,
        )
        payload = _build_payload(args, "hello", is_chat=False)
        assert "user" not in payload


class TestCLIParsing:
    """Test that CLI args parse correctly."""

    def test_echo_flag(self):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args
        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args([
            "--echo", "--suffix", "END",
            "--logit-bias", '{"50256": -100}',
            "--user", "test-user",
        ])
        assert args.echo is True
        assert args.suffix == "END"
        assert args.logit_bias == '{"50256": -100}'
        assert args.user == "test-user"

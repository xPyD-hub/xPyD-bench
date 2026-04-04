"""Tests for issue #19: vLLM-specific parameters are properly grouped and documented."""

from __future__ import annotations

import argparse


def _get_parser() -> argparse.ArgumentParser:
    """Build the bench CLI parser without running anything."""
    parser = argparse.ArgumentParser(prog="xpyd-bench")
    # Re-use the same function that bench_main uses
    from xpyd_bench.cli import _add_vllm_compat_args

    _add_vllm_compat_args(parser)
    return parser


class TestVLLMExtensionsGroup:
    """Verify vLLM-specific params are in their own argument group."""

    def test_vllm_group_exists(self):
        parser = _get_parser()
        group_titles = [g.title for g in parser._action_groups]
        assert "vLLM extensions" in group_titles

    def test_vllm_params_in_group(self):
        parser = _get_parser()
        vllm_group = None
        for g in parser._action_groups:
            if g.title == "vLLM extensions":
                vllm_group = g
                break
        assert vllm_group is not None
        option_strings = []
        for action in vllm_group._group_actions:
            option_strings.extend(action.option_strings)
        assert "--top-k" in option_strings
        assert "--use-beam-search" in option_strings
        assert "--ignore-eos" in option_strings
        assert "--best-of" in option_strings

    def test_vllm_params_not_in_standard_sampling(self):
        parser = _get_parser()
        sampling_group = None
        for g in parser._action_groups:
            if "sampling" in (g.title or "").lower() and "standard" in (g.title or "").lower():
                sampling_group = g
                break
        assert sampling_group is not None
        option_strings = []
        for action in sampling_group._group_actions:
            option_strings.extend(action.option_strings)
        # These should NOT be in the standard sampling group
        assert "--top-k" not in option_strings
        assert "--use-beam-search" not in option_strings
        assert "--ignore-eos" not in option_strings
        assert "--best-of" not in option_strings

    def test_vllm_help_text_contains_warning(self):
        parser = _get_parser()
        vllm_group = None
        for g in parser._action_groups:
            if g.title == "vLLM extensions":
                vllm_group = g
                break
        assert vllm_group is not None
        for action in vllm_group._group_actions:
            if action.help:
                assert "vLLM" in action.help or "not in OpenAI" in action.help.lower()

    def test_standard_params_remain(self):
        """Standard OpenAI params should still be accessible."""
        parser = _get_parser()
        args = parser.parse_args(["--temperature", "0.7", "--top-p", "0.9"])
        assert args.temperature == 0.7
        assert args.top_p == 0.9

    def test_vllm_params_parse(self):
        """vLLM extension params should still parse correctly."""
        parser = _get_parser()
        args = parser.parse_args(["--top-k", "50", "--use-beam-search", "--ignore-eos"])
        assert args.top_k == 50
        assert args.use_beam_search is True
        assert args.ignore_eos is True

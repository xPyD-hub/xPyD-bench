"""Tests for M27: Tokenizer-accurate token counting."""

from __future__ import annotations

import subprocess
import sys

import pytest


class TestCountTokens:
    """Tests for xpyd_bench.tokenizer.count_tokens."""

    def test_word_split_fallback_when_no_tokenizer(self):
        from xpyd_bench.tokenizer import count_tokens

        text = "the quick brown fox"
        assert count_tokens(text, tokenizer=None) == 4

    def test_word_split_fallback_when_tiktoken_unavailable(self):
        """count_tokens falls back to word-split when tiktoken is missing."""
        from xpyd_bench import tokenizer as tok_mod

        # Temporarily pretend tiktoken is not available
        original = tok_mod._TIKTOKEN_AVAILABLE
        try:
            tok_mod._TIKTOKEN_AVAILABLE = False
            result = tok_mod.count_tokens("hello world benchmark", tokenizer="cl100k_base")
            assert result == 3  # word-split: 3 words
        finally:
            tok_mod._TIKTOKEN_AVAILABLE = original

    def test_tiktoken_accurate_counting(self):
        """When tiktoken is available, token count differs from word count."""
        tiktoken = pytest.importorskip("tiktoken")
        from xpyd_bench.tokenizer import count_tokens

        # "tokenization" is a single word but multiple BPE tokens
        text = "tokenization is a subword process"
        token_count = count_tokens(text, tokenizer="cl100k_base")
        # tiktoken should return a count (may differ from word count)
        assert isinstance(token_count, int)
        assert token_count > 0
        # Verify it matches tiktoken directly
        enc = tiktoken.get_encoding("cl100k_base")
        assert token_count == len(enc.encode(text))

    def test_tiktoken_with_model_name(self):
        """Can use a model name like 'gpt-4' as tokenizer."""
        pytest.importorskip("tiktoken")
        from xpyd_bench.tokenizer import count_tokens

        result = count_tokens("hello world", tokenizer="gpt-4")
        assert isinstance(result, int)
        assert result > 0

    def test_encoding_cache(self):
        """Encoding objects are cached across calls."""
        pytest.importorskip("tiktoken")
        from xpyd_bench import tokenizer as tok_mod

        tok_mod._ENCODING_CACHE.clear()
        tok_mod.count_tokens("hello", tokenizer="cl100k_base")
        assert "cl100k_base" in tok_mod._ENCODING_CACHE
        tok_mod.count_tokens("world", tokenizer="cl100k_base")
        # Still only one entry
        assert len([k for k in tok_mod._ENCODING_CACHE if k == "cl100k_base"]) == 1


class TestTokensToText:
    """Tests for xpyd_bench.tokenizer.tokens_to_text."""

    def test_fallback_word_count(self):
        from xpyd_bench.tokenizer import tokens_to_text

        text = tokens_to_text(10, tokenizer=None, seed=42)
        assert len(text.split()) == 10

    def test_exact_token_count_with_tiktoken(self):
        tiktoken = pytest.importorskip("tiktoken")
        from xpyd_bench.tokenizer import tokens_to_text

        text = tokens_to_text(50, tokenizer="cl100k_base", seed=42)
        enc = tiktoken.get_encoding("cl100k_base")
        assert len(enc.encode(text)) == 50

    def test_reproducible_with_seed(self):
        from xpyd_bench.tokenizer import tokens_to_text

        t1 = tokens_to_text(20, tokenizer=None, seed=123)
        t2 = tokens_to_text(20, tokenizer=None, seed=123)
        assert t1 == t2


class TestDatasetIntegration:
    """Tokenizer integration with dataset loader."""

    def test_estimate_tokens_with_tokenizer(self):
        pytest.importorskip("tiktoken")
        from xpyd_bench.datasets.loader import _estimate_tokens

        text = "the quick brown fox jumps over the lazy dog"
        word_count = _estimate_tokens(text, tokenizer=None)
        token_count = _estimate_tokens(text, tokenizer="cl100k_base")
        assert word_count == 9  # 9 words
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_compute_stats_with_tokenizer(self):
        pytest.importorskip("tiktoken")
        from xpyd_bench.datasets.loader import DatasetEntry, compute_stats

        entries = [
            DatasetEntry(prompt="hello world", output_len=10),
            DatasetEntry(prompt="the quick brown fox jumps", output_len=20),
        ]
        stats_word = compute_stats(entries, tokenizer=None)
        stats_tok = compute_stats(entries, tokenizer="cl100k_base")
        # Both should have valid stats
        assert stats_word.count == 2
        assert stats_tok.count == 2
        assert stats_word.avg_prompt_len > 0
        assert stats_tok.avg_prompt_len > 0

    def test_synthetic_generation_with_tokenizer(self):
        tiktoken = pytest.importorskip("tiktoken")
        from xpyd_bench.datasets.loader import generate_synthetic

        entries = generate_synthetic(
            num=5, input_len=30, output_len=10, tokenizer="cl100k_base"
        )
        assert len(entries) == 5
        enc = tiktoken.get_encoding("cl100k_base")
        for entry in entries:
            token_count = len(enc.encode(entry.prompt))
            assert token_count == 30

    def test_synthetic_generation_without_tokenizer(self):
        from xpyd_bench.datasets.loader import generate_synthetic

        entries = generate_synthetic(num=5, input_len=30, output_len=10, tokenizer=None)
        assert len(entries) == 5
        for entry in entries:
            assert len(entry.prompt.split()) == 30

    def test_load_dataset_passes_tokenizer(self):
        pytest.importorskip("tiktoken")
        from xpyd_bench.datasets.loader import load_dataset

        entries = load_dataset(
            num_prompts=3, input_len=20, output_len=5, tokenizer="cl100k_base"
        )
        assert len(entries) == 3


class TestCLITokenizerFlag:
    """Test --tokenizer CLI flag."""

    def test_tokenizer_flag_accepted(self):
        """CLI accepts --tokenizer without error."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "xpyd_bench.main",
                "run",
                "--dry-run",
                "--tokenizer",
                "cl100k_base",
                "--num-prompts",
                "2",
                "--input-len",
                "10",
                "--output-len",
                "5",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Dry-run should succeed (exit 0)
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_tokenizer_in_yaml_config(self, tmp_path):
        """Tokenizer can be set via YAML config."""
        config = tmp_path / "config.yaml"
        config.write_text("tokenizer: cl100k_base\nnum_prompts: 2\ninput_len: 10\n")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "xpyd_bench.main",
                "run",
                "--dry-run",
                "--config",
                str(config),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"


class TestTiktokenAvailable:
    """Test tiktoken_available() utility."""

    def test_returns_bool(self):
        from xpyd_bench.tokenizer import tiktoken_available

        result = tiktoken_available()
        assert isinstance(result, bool)

    def test_detects_installed_tiktoken(self):
        """If tiktoken is installed (dev dep), should return True."""
        pytest.importorskip("tiktoken")
        from xpyd_bench import tokenizer as tok_mod

        tok_mod._TIKTOKEN_AVAILABLE = None  # reset cache
        assert tok_mod.tiktoken_available() is True

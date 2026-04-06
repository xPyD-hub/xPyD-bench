"""Tokenizer utilities with tiktoken integration and word-split fallback."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_TIKTOKEN_AVAILABLE: bool | None = None
_ENCODING_CACHE: dict[str, object] = {}


def tiktoken_available() -> bool:
    """Return True if tiktoken is installed and importable."""
    global _TIKTOKEN_AVAILABLE
    if _TIKTOKEN_AVAILABLE is None:
        try:
            import tiktoken  # noqa: F401

            _TIKTOKEN_AVAILABLE = True
        except ImportError:
            _TIKTOKEN_AVAILABLE = False
    return _TIKTOKEN_AVAILABLE


def _get_encoding(model: str) -> object:
    """Get or cache a tiktoken encoding by model name."""
    if model not in _ENCODING_CACHE:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            # Model name not recognized; try as encoding name directly
            enc = tiktoken.get_encoding(model)
        _ENCODING_CACHE[model] = enc
    return _ENCODING_CACHE[model]


def count_tokens(text: str, tokenizer: str | None = None) -> int:
    """Count tokens in *text*.

    When *tokenizer* is provided and tiktoken is available, uses tiktoken
    for accurate BPE token counting.  Falls back to ``len(text.split())``
    when tiktoken is not installed or *tokenizer* is ``None``.

    Parameters
    ----------
    text:
        The text to tokenize.
    tokenizer:
        Tiktoken model or encoding name (e.g. ``"cl100k_base"``,
        ``"gpt-4"``).  ``None`` forces the word-split fallback.
    """
    if tokenizer is not None and tiktoken_available():
        enc = _get_encoding(tokenizer)
        return len(enc.encode(text))  # type: ignore[union-attr]
    return len(text.split())


def tokens_to_text(num_tokens: int, tokenizer: str | None = None, seed: int = 0) -> str:
    """Generate text that is approximately *num_tokens* tokens long.

    When tiktoken is available and *tokenizer* is given, the returned text
    is guaranteed to be exactly *num_tokens* tokens by encoding random words
    and trimming/padding at the token level.  Otherwise falls back to
    generating *num_tokens* space-separated words.

    Parameters
    ----------
    num_tokens:
        Target token count.
    tokenizer:
        Tiktoken model or encoding name.  ``None`` uses word-based fallback.
    seed:
        Random seed for reproducibility.
    """
    import random

    from xpyd_bench.datasets.loader import _VOCAB

    rng = random.Random(seed)

    if tokenizer is None or not tiktoken_available():
        # Word-based fallback: 1 word ≈ 1 token
        return " ".join(rng.choice(_VOCAB) for _ in range(num_tokens))

    enc = _get_encoding(tokenizer)

    # Generate excess words, encode, trim to exact token count
    words = [rng.choice(_VOCAB) for _ in range(num_tokens * 2)]
    text = " ".join(words)
    token_ids = enc.encode(text)  # type: ignore[union-attr]

    if len(token_ids) >= num_tokens:
        token_ids = token_ids[:num_tokens]
    else:
        # Rare: need more tokens — keep appending words
        while len(token_ids) < num_tokens:
            words.append(rng.choice(_VOCAB))
            text = " ".join(words)
            token_ids = enc.encode(text)  # type: ignore[union-attr]
        token_ids = token_ids[:num_tokens]

    return enc.decode(token_ids)  # type: ignore[union-attr]

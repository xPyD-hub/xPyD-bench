"""Dataset loader: JSONL, JSON, CSV, and synthetic generation."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DatasetStats:
    """Summary statistics for a loaded dataset."""

    count: int
    avg_prompt_len: float  # estimated tokens (words)
    min_prompt_len: int
    max_prompt_len: int
    avg_output_len: float | None = None
    min_output_len: int | None = None
    max_output_len: int | None = None


@dataclass
class DatasetEntry:
    """A single dataset entry with prompt and optional expected output length."""

    prompt: str
    output_len: int | None = None
    priority: int | None = None


def _estimate_tokens(text: str, tokenizer: str | None = None) -> int:
    """Estimate token count from text.

    When *tokenizer* is provided and tiktoken is available, uses accurate
    BPE tokenization.  Otherwise falls back to word-split approximation.
    """
    from xpyd_bench.tokenizer import count_tokens

    return count_tokens(text, tokenizer=tokenizer)


def _validate_entries(entries: list[DatasetEntry], source: str) -> None:
    """Validate dataset entries, raise on problems."""
    if not entries:
        raise ValueError(f"Dataset is empty: {source}")
    for i, entry in enumerate(entries):
        if not entry.prompt or not entry.prompt.strip():
            raise ValueError(f"Empty prompt at index {i} in {source}")


def compute_stats(
    entries: list[DatasetEntry], tokenizer: str | None = None
) -> DatasetStats:
    """Compute summary statistics for dataset entries.

    Parameters
    ----------
    entries:
        Dataset entries to summarize.
    tokenizer:
        Optional tiktoken model/encoding name for accurate token counting.
    """
    prompt_lens = [_estimate_tokens(e.prompt, tokenizer=tokenizer) for e in entries]
    output_lens = [e.output_len for e in entries if e.output_len is not None]

    stats = DatasetStats(
        count=len(entries),
        avg_prompt_len=sum(prompt_lens) / len(prompt_lens),
        min_prompt_len=min(prompt_lens),
        max_prompt_len=max(prompt_lens),
    )
    if output_lens:
        stats.avg_output_len = sum(output_lens) / len(output_lens)
        stats.min_output_len = min(output_lens)
        stats.max_output_len = max(output_lens)
    return stats


def _parse_record(record: dict[str, Any], index: int, source: str) -> DatasetEntry:
    """Parse a single record dict into a DatasetEntry."""
    prompt = record.get("prompt")
    if prompt is None:
        # Try alternate field names
        prompt = record.get("text") or record.get("input") or record.get("question")
    if prompt is None:
        raise ValueError(
            f"Record at index {index} in {source} has no 'prompt', 'text', "
            f"'input', or 'question' field"
        )
    output_len = record.get("output_len") or record.get("max_tokens")
    if output_len is not None:
        output_len = int(output_len)
    priority = record.get("priority")
    if priority is not None:
        priority = int(priority)
    return DatasetEntry(prompt=str(prompt), output_len=output_len, priority=priority)


def load_jsonl(path: Path) -> list[DatasetEntry]:
    """Load dataset from a JSONL file (one JSON object per line)."""
    entries: list[DatasetEntry] = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {i + 1} in {path}: {exc}") from exc
            entries.append(_parse_record(record, i, str(path)))
    return entries


def load_json(path: Path) -> list[DatasetEntry]:
    """Load dataset from a JSON file (array of objects)."""
    with open(path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}, got {type(data).__name__}")
    return [_parse_record(record, i, str(path)) for i, record in enumerate(data)]


def load_csv(path: Path) -> list[DatasetEntry]:
    """Load dataset from a CSV file with 'prompt' column."""
    entries: list[DatasetEntry] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {path} has no header row")
        # Find prompt column (case-insensitive)
        prompt_col = None
        for col in reader.fieldnames:
            if col.lower() in ("prompt", "text", "input", "question"):
                prompt_col = col
                break
        if prompt_col is None:
            raise ValueError(
                f"CSV file {path} has no 'prompt', 'text', 'input', or 'question' column. "
                f"Found columns: {reader.fieldnames}"
            )
        output_col = None
        for col in reader.fieldnames:
            if col.lower() in ("output_len", "max_tokens"):
                output_col = col
                break
        for i, row in enumerate(reader):
            prompt = row[prompt_col]
            if not prompt or not prompt.strip():
                raise ValueError(f"Empty prompt at row {i + 1} in {path}")
            output_len = None
            if output_col and row.get(output_col):
                output_len = int(row[output_col])
            entries.append(DatasetEntry(prompt=prompt, output_len=output_len))
    return entries


# ---------------------------------------------------------------------------
# Synthetic generation
# ---------------------------------------------------------------------------

_VOCAB = [
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
    "a", "an", "is", "was", "hello", "world", "benchmark", "test",
    "data", "model", "request", "response", "token", "stream",
    "latency", "throughput", "server", "client", "batch", "queue",
    "memory", "compute", "inference", "training", "weight", "layer",
    "attention", "transformer", "embedding", "gradient", "optimizer", "loss",
]


def generate_synthetic(
    num: int,
    input_len: int = 256,
    output_len: int = 128,
    input_len_dist: str = "fixed",
    output_len_dist: str = "fixed",
    seed: int = 0,
    tokenizer: str | None = None,
) -> list[DatasetEntry]:
    """Generate synthetic dataset with configurable length distributions.

    Supported distributions: fixed, uniform, normal, zipf.
    For uniform/normal/zipf, the given length is used as the mean/center.

    When *tokenizer* is provided and tiktoken is available, prompts are
    generated with exact token counts using BPE tokenization.
    """
    rng = random.Random(seed)
    entries: list[DatasetEntry] = []

    input_lens = _sample_lengths(num, input_len, input_len_dist, seed)
    output_lens = _sample_lengths(num, output_len, output_len_dist, seed + 1)

    if tokenizer is not None:
        from xpyd_bench.tokenizer import tiktoken_available, tokens_to_text

        if tiktoken_available():
            for i in range(num):
                prompt = tokens_to_text(
                    input_lens[i], tokenizer=tokenizer, seed=seed + i + 2
                )
                entries.append(
                    DatasetEntry(prompt=prompt, output_len=output_lens[i])
                )
            return entries

    for i in range(num):
        prompt_words = [rng.choice(_VOCAB) for _ in range(input_lens[i])]
        entries.append(DatasetEntry(prompt=" ".join(prompt_words), output_len=output_lens[i]))

    return entries


def _sample_lengths(num: int, center: int, dist: str, seed: int) -> list[int]:
    """Sample lengths from a distribution centered around *center*."""
    rng = random.Random(seed)

    if dist == "fixed":
        return [center] * num

    if dist == "uniform":
        lo = max(1, center // 2)
        hi = center * 2
        return [rng.randint(lo, hi) for _ in range(num)]

    if dist == "normal":
        std = max(1, center // 4)
        return [max(1, int(rng.gauss(center, std))) for _ in range(num)]

    if dist == "zipf":
        # Approximate Zipf-like: sample from a power-law around center
        results = []
        for _ in range(num):
            # Zipf(a=1.5) mapped to center
            x = rng.paretovariate(1.5)
            length = max(1, int(center / x))
            results.append(length)
        return results

    raise ValueError(f"Unknown distribution: {dist!r}. Use: fixed, uniform, normal, zipf.")


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def load_dataset(
    path: str | None = None,
    name: str = "random",
    num_prompts: int = 1000,
    input_len: int = 256,
    output_len: int = 128,
    input_len_dist: str = "fixed",
    output_len_dist: str = "fixed",
    seed: int = 0,
    tokenizer: str | None = None,
) -> list[DatasetEntry]:
    """Load or generate a dataset.

    If *path* is given, detect format by extension (.jsonl, .json, .csv).
    Otherwise generate a synthetic dataset named *name*.

    Parameters
    ----------
    tokenizer:
        Optional tiktoken model/encoding for token-accurate synthetic
        generation.
    """
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        ext = p.suffix.lower()
        if ext == ".jsonl":
            entries = load_jsonl(p)
        elif ext == ".json":
            entries = load_json(p)
        elif ext == ".csv":
            entries = load_csv(p)
        else:
            raise ValueError(f"Unsupported dataset format: {ext}. Use .jsonl, .json, or .csv.")
        _validate_entries(entries, str(p))
        return entries

    # Synthetic
    return generate_synthetic(
        num=num_prompts,
        input_len=input_len,
        output_len=output_len,
        input_len_dist=input_len_dist,
        output_len_dist=output_len_dist,
        seed=seed,
        tokenizer=tokenizer,
    )


def validate_and_report(
    entries: list[DatasetEntry],
    source: str = "dataset",
    tokenizer: str | None = None,
) -> DatasetStats:
    """Validate entries and print summary stats. Returns stats."""
    _validate_entries(entries, source)
    stats = compute_stats(entries, tokenizer=tokenizer)
    counting_method = "tiktoken" if tokenizer is not None else "word-split"
    print(f"Dataset: {source}")
    print(f"  Entries:          {stats.count}")
    print(f"  Avg prompt len:   {stats.avg_prompt_len:.1f} tokens ({counting_method})")
    print(f"  Min prompt len:   {stats.min_prompt_len}")
    print(f"  Max prompt len:   {stats.max_prompt_len}")
    if stats.avg_output_len is not None:
        print(f"  Avg output len:   {stats.avg_output_len:.1f}")
        print(f"  Min output len:   {stats.min_output_len}")
        print(f"  Max output len:   {stats.max_output_len}")
    return stats

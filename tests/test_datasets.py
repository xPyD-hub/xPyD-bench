"""Tests for dataset loading and generation (M5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpyd_bench.datasets.loader import (
    DatasetEntry,
    compute_stats,
    generate_synthetic,
    load_csv,
    load_dataset,
    load_json,
    load_jsonl,
    validate_and_report,
)

# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------


class TestLoadJsonl:
    def test_basic(self, tmp_path: Path) -> None:
        p = tmp_path / "data.jsonl"
        p.write_text(
            '{"prompt": "hello world"}\n'
            '{"prompt": "foo bar", "output_len": 64}\n'
        )
        entries = load_jsonl(p)
        assert len(entries) == 2
        assert entries[0].prompt == "hello world"
        assert entries[0].output_len is None
        assert entries[1].output_len == 64

    def test_alternate_fields(self, tmp_path: Path) -> None:
        p = tmp_path / "data.jsonl"
        p.write_text('{"text": "alt field"}\n')
        entries = load_jsonl(p)
        assert entries[0].prompt == "alt field"

    def test_invalid_json(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.jsonl"
        p.write_text("not json\n")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_jsonl(p)

    def test_missing_prompt(self, tmp_path: Path) -> None:
        p = tmp_path / "noprompt.jsonl"
        p.write_text('{"other": "value"}\n')
        with pytest.raises(ValueError, match="no 'prompt'"):
            load_jsonl(p)

    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "blanks.jsonl"
        p.write_text('{"prompt": "a"}\n\n{"prompt": "b"}\n')
        assert len(load_jsonl(p)) == 2


# ---------------------------------------------------------------------------
# JSON array
# ---------------------------------------------------------------------------


class TestLoadJson:
    def test_basic(self, tmp_path: Path) -> None:
        p = tmp_path / "data.json"
        p.write_text(json.dumps([{"prompt": "p1"}, {"prompt": "p2"}]))
        entries = load_json(p)
        assert len(entries) == 2

    def test_not_array(self, tmp_path: Path) -> None:
        p = tmp_path / "obj.json"
        p.write_text('{"prompt": "single"}')
        with pytest.raises(ValueError, match="Expected JSON array"):
            load_json(p)

    def test_invalid_json(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("{broken")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_json(p)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


class TestLoadCsv:
    def test_basic(self, tmp_path: Path) -> None:
        p = tmp_path / "data.csv"
        p.write_text("prompt,output_len\nhello world,64\nfoo bar,128\n")
        entries = load_csv(p)
        assert len(entries) == 2
        assert entries[0].prompt == "hello world"
        assert entries[0].output_len == 64

    def test_alternate_column(self, tmp_path: Path) -> None:
        p = tmp_path / "data.csv"
        p.write_text("text\nsome prompt\n")
        entries = load_csv(p)
        assert entries[0].prompt == "some prompt"

    def test_no_prompt_column(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.csv"
        p.write_text("id,value\n1,abc\n")
        with pytest.raises(ValueError, match="no 'prompt'"):
            load_csv(p)

    def test_empty_prompt_row(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.csv"
        # A row with an explicitly empty prompt value
        p.write_text('prompt\nhello\n""\n')
        with pytest.raises(ValueError, match="Empty prompt"):
            load_csv(p)


# ---------------------------------------------------------------------------
# Synthetic generation
# ---------------------------------------------------------------------------


class TestSynthetic:
    def test_fixed(self) -> None:
        entries = generate_synthetic(10, input_len=50, output_len=25, seed=42)
        assert len(entries) == 10
        for e in entries:
            assert len(e.prompt.split()) == 50
            assert e.output_len == 25

    def test_uniform(self) -> None:
        entries = generate_synthetic(
            100, input_len=100, input_len_dist="uniform", seed=42
        )
        lens = [len(e.prompt.split()) for e in entries]
        assert min(lens) >= 1
        # Should have variety
        assert max(lens) > min(lens)

    def test_normal(self) -> None:
        entries = generate_synthetic(
            100, input_len=100, input_len_dist="normal", seed=42
        )
        lens = [len(e.prompt.split()) for e in entries]
        assert max(lens) > min(lens)

    def test_zipf(self) -> None:
        entries = generate_synthetic(
            100, input_len=100, input_len_dist="zipf", seed=42
        )
        lens = [len(e.prompt.split()) for e in entries]
        assert max(lens) > min(lens)

    def test_unknown_dist(self) -> None:
        with pytest.raises(ValueError, match="Unknown distribution"):
            generate_synthetic(10, input_len_dist="bogus")

    def test_output_dist(self) -> None:
        entries = generate_synthetic(
            50, output_len=100, output_len_dist="uniform", seed=7
        )
        out_lens = [e.output_len for e in entries]
        assert max(out_lens) > min(out_lens)


# ---------------------------------------------------------------------------
# load_dataset (unified API)
# ---------------------------------------------------------------------------


class TestLoadDataset:
    def test_from_jsonl(self, tmp_path: Path) -> None:
        p = tmp_path / "d.jsonl"
        p.write_text('{"prompt": "x"}\n')
        entries = load_dataset(path=str(p))
        assert len(entries) == 1

    def test_from_json(self, tmp_path: Path) -> None:
        p = tmp_path / "d.json"
        p.write_text(json.dumps([{"prompt": "x"}]))
        entries = load_dataset(path=str(p))
        assert len(entries) == 1

    def test_from_csv(self, tmp_path: Path) -> None:
        p = tmp_path / "d.csv"
        p.write_text("prompt\nx\n")
        entries = load_dataset(path=str(p))
        assert len(entries) == 1

    def test_unsupported_ext(self, tmp_path: Path) -> None:
        p = tmp_path / "d.xml"
        p.write_text("<data/>")
        with pytest.raises(ValueError, match="Unsupported"):
            load_dataset(path=str(p))

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_dataset(path="/nonexistent/file.jsonl")

    def test_synthetic_default(self) -> None:
        entries = load_dataset(num_prompts=5, input_len=10, seed=1)
        assert len(entries) == 5

    def test_empty_dataset(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        with pytest.raises(ValueError, match="empty"):
            load_dataset(path=str(p))


# ---------------------------------------------------------------------------
# Stats & validation
# ---------------------------------------------------------------------------


class TestStatsAndValidation:
    def test_compute_stats(self) -> None:
        entries = [
            DatasetEntry(prompt="one two three", output_len=10),
            DatasetEntry(prompt="a b c d e", output_len=20),
        ]
        stats = compute_stats(entries)
        assert stats.count == 2
        assert stats.min_prompt_len == 3
        assert stats.max_prompt_len == 5
        assert stats.avg_output_len == 15.0

    def test_validate_and_report(self, capsys) -> None:
        entries = [DatasetEntry(prompt="hello world")]
        stats = validate_and_report(entries, "test")
        assert stats.count == 1
        captured = capsys.readouterr()
        assert "Dataset: test" in captured.out
        assert "Entries:" in captured.out

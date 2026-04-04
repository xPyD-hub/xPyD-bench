"""Tests for M18: Profile & Replay Mode."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from xpyd_bench.profile import (
    TraceData,
    TraceEntry,
    TraceRecorder,
    compute_delays,
    load_trace,
    save_trace,
)

# ---------------------------------------------------------------------------
# TraceEntry & TraceData
# ---------------------------------------------------------------------------


class TestTraceData:
    def test_to_dict_roundtrip(self):
        entry = TraceEntry(
            offset_s=1.5,
            prompt_len=100,
            output_len=50,
            endpoint="/v1/completions",
            model="gpt-4",
            prompt="hello world",
            temperature=0.7,
            max_tokens=256,
            stream=True,
        )
        trace = TraceData(
            version=1,
            base_url="http://localhost:8000",
            total_duration_s=10.0,
            num_entries=1,
            entries=[entry],
        )
        d = trace.to_dict()
        restored = TraceData.from_dict(d)
        assert restored.version == 1
        assert restored.base_url == "http://localhost:8000"
        assert restored.total_duration_s == 10.0
        assert len(restored.entries) == 1
        assert restored.entries[0].offset_s == 1.5
        assert restored.entries[0].prompt == "hello world"
        assert restored.entries[0].model == "gpt-4"

    def test_from_dict_defaults(self):
        trace = TraceData.from_dict({})
        assert trace.version == 1
        assert trace.entries == []
        assert trace.num_entries == 0

    def test_from_dict_computes_num_entries(self):
        data = {
            "entries": [
                {"offset_s": 0.0, "prompt_len": 10, "output_len": 5},
                {"offset_s": 1.0, "prompt_len": 20, "output_len": 10},
            ]
        }
        trace = TraceData.from_dict(data)
        assert trace.num_entries == 2


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_and_load(self, tmp_path: Path):
        entry = TraceEntry(offset_s=0.5, prompt_len=50, output_len=30)
        trace = TraceData(
            base_url="http://example.com",
            total_duration_s=5.0,
            num_entries=1,
            entries=[entry],
        )
        out = tmp_path / "trace.json"
        save_trace(trace, out)

        loaded = load_trace(out)
        assert loaded.base_url == "http://example.com"
        assert loaded.total_duration_s == 5.0
        assert len(loaded.entries) == 1
        assert loaded.entries[0].offset_s == 0.5

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        out = tmp_path / "sub" / "dir" / "trace.json"
        trace = TraceData(num_entries=0)
        save_trace(trace, out)
        assert out.exists()

    def test_save_produces_valid_json(self, tmp_path: Path):
        trace = TraceData(
            entries=[TraceEntry(offset_s=0, prompt_len=10, output_len=5)]
        )
        out = tmp_path / "t.json"
        save_trace(trace, out)
        data = json.loads(out.read_text())
        assert data["version"] == 1
        assert len(data["entries"]) == 1


# ---------------------------------------------------------------------------
# TraceRecorder
# ---------------------------------------------------------------------------


class TestTraceRecorder:
    def test_basic_recording(self):
        rec = TraceRecorder(base_url="http://localhost:8000")
        rec.start()
        time.sleep(0.01)
        rec.record(prompt_len=100, output_len=50)
        time.sleep(0.01)
        rec.record(prompt_len=200, output_len=100)
        trace = rec.finish()

        assert trace.base_url == "http://localhost:8000"
        assert trace.num_entries == 2
        assert trace.total_duration_s > 0
        assert trace.entries[0].offset_s < trace.entries[1].offset_s

    def test_auto_start(self):
        rec = TraceRecorder()
        rec.record(prompt_len=10, output_len=5)
        trace = rec.finish()
        assert trace.num_entries == 1
        assert trace.entries[0].offset_s >= 0

    def test_record_all_fields(self):
        rec = TraceRecorder()
        rec.record(
            prompt_len=100,
            output_len=50,
            endpoint="/v1/chat/completions",
            model="gpt-4",
            prompt="test prompt",
            temperature=0.5,
            max_tokens=512,
            stream=False,
        )
        trace = rec.finish()
        e = trace.entries[0]
        assert e.endpoint == "/v1/chat/completions"
        assert e.model == "gpt-4"
        assert e.temperature == 0.5
        assert e.max_tokens == 512
        assert e.stream is False


# ---------------------------------------------------------------------------
# compute_delays
# ---------------------------------------------------------------------------


class TestComputeDelays:
    def test_empty_trace(self):
        trace = TraceData(entries=[])
        assert compute_delays(trace) == []

    def test_single_entry(self):
        trace = TraceData(entries=[
            TraceEntry(offset_s=2.0, prompt_len=10, output_len=5)
        ])
        delays = compute_delays(trace)
        assert len(delays) == 1
        assert delays[0] == pytest.approx(2.0)

    def test_multiple_entries(self):
        trace = TraceData(entries=[
            TraceEntry(offset_s=0.0, prompt_len=10, output_len=5),
            TraceEntry(offset_s=1.0, prompt_len=20, output_len=10),
            TraceEntry(offset_s=3.0, prompt_len=30, output_len=15),
        ])
        delays = compute_delays(trace)
        assert delays == [pytest.approx(0.0), pytest.approx(1.0), pytest.approx(2.0)]

    def test_speed_multiplier(self):
        trace = TraceData(entries=[
            TraceEntry(offset_s=0.0, prompt_len=10, output_len=5),
            TraceEntry(offset_s=2.0, prompt_len=20, output_len=10),
        ])
        delays = compute_delays(trace, speed=2.0)
        assert delays == [pytest.approx(0.0), pytest.approx(1.0)]

    def test_speed_slow(self):
        trace = TraceData(entries=[
            TraceEntry(offset_s=0.0, prompt_len=10, output_len=5),
            TraceEntry(offset_s=1.0, prompt_len=20, output_len=10),
        ])
        delays = compute_delays(trace, speed=0.5)
        assert delays == [pytest.approx(0.0), pytest.approx(2.0)]

    def test_invalid_speed(self):
        trace = TraceData(entries=[
            TraceEntry(offset_s=0.0, prompt_len=10, output_len=5)
        ])
        with pytest.raises(ValueError, match="speed must be positive"):
            compute_delays(trace, speed=0)
        with pytest.raises(ValueError, match="speed must be positive"):
            compute_delays(trace, speed=-1)

    def test_negative_offset_clamped(self):
        """If offsets go backwards (shouldn't happen), delays clamp to 0."""
        trace = TraceData(entries=[
            TraceEntry(offset_s=5.0, prompt_len=10, output_len=5),
            TraceEntry(offset_s=3.0, prompt_len=20, output_len=10),
        ])
        delays = compute_delays(trace)
        assert delays[1] == 0.0


# ---------------------------------------------------------------------------
# CLI integration (argument parsing only)
# ---------------------------------------------------------------------------


class TestCLIParsing:
    def test_profile_help(self):
        """profile_main accepts --output."""
        from xpyd_bench.cli import profile_main

        with pytest.raises(SystemExit):
            profile_main(["--help"])

    def test_replay_help(self):
        """replay_main accepts --trace and --base-url."""
        from xpyd_bench.cli import replay_main

        with pytest.raises(SystemExit):
            replay_main(["--help"])

    def test_replay_requires_trace(self):
        from xpyd_bench.cli import replay_main

        with pytest.raises(SystemExit):
            replay_main(["--base-url", "http://localhost:8000"])

    def test_replay_requires_base_url(self):
        from xpyd_bench.cli import replay_main

        with pytest.raises(SystemExit):
            replay_main(["--trace", "trace.json"])

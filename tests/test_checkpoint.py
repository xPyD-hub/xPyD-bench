"""Tests for benchmark checkpointing and resume (M74)."""

from __future__ import annotations

import json

import pytest

from xpyd_bench.bench.models import RequestResult
from xpyd_bench.checkpoint import (
    CheckpointManager,
    load_checkpoint,
    restore_requests,
    resume_main,
    validate_config_match,
)


def _make_result(latency: float = 100.0, success: bool = True) -> RequestResult:
    return RequestResult(
        prompt_tokens=10,
        completion_tokens=20,
        latency_ms=latency,
        success=success,
    )


class TestCheckpointManager:
    def test_checkpoint_creation(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ckpt", interval=3, config_snapshot={"model": "test"})
        # Add 3 results → should auto-save
        for i in range(3):
            mgr.record(_make_result(latency=float(i)))

        ckpt_file = tmp_path / "ckpt" / "checkpoint.json"
        assert ckpt_file.exists()
        data = json.loads(ckpt_file.read_text())
        assert data["completed_count"] == 3
        assert data["config_snapshot"]["model"] == "test"
        assert len(data["requests"]) == 3
        assert data["version"] == 1

    def test_checkpoint_interval(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ckpt", interval=5)
        # Add 4 results → no checkpoint yet
        for i in range(4):
            mgr.record(_make_result())
        ckpt_file = tmp_path / "ckpt" / "checkpoint.json"
        assert not ckpt_file.exists()

        # 5th result triggers checkpoint
        mgr.record(_make_result())
        assert ckpt_file.exists()
        data = json.loads(ckpt_file.read_text())
        assert data["completed_count"] == 5

    def test_manual_save(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ckpt", interval=100)
        mgr.record(_make_result())
        mgr.save()
        ckpt_file = tmp_path / "ckpt" / "checkpoint.json"
        assert ckpt_file.exists()
        data = json.loads(ckpt_file.read_text())
        assert data["completed_count"] == 1

    def test_results_property(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ckpt", interval=100)
        r = _make_result(latency=42.0)
        mgr.record(r)
        assert len(mgr.results) == 1
        assert mgr.results[0].latency_ms == 42.0

    def test_creates_directory(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        mgr = CheckpointManager(nested, interval=1)
        mgr.record(_make_result())
        assert (nested / "checkpoint.json").exists()


class TestLoadCheckpoint:
    def test_load_from_file(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ckpt", interval=1, config_snapshot={"k": "v"})
        mgr.record(_make_result())
        data = load_checkpoint(tmp_path / "ckpt" / "checkpoint.json")
        assert data["completed_count"] == 1
        assert data["config_snapshot"] == {"k": "v"}

    def test_load_from_dir(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ckpt", interval=1)
        mgr.record(_make_result())
        data = load_checkpoint(tmp_path / "ckpt")
        assert data["completed_count"] == 1

    def test_load_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "nonexistent")


class TestRestoreRequests:
    def test_restore(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ckpt", interval=1)
        mgr.record(_make_result(latency=99.0))
        mgr.record(_make_result(latency=200.0))
        data = load_checkpoint(tmp_path / "ckpt")
        results = restore_requests(data)
        assert len(results) == 2
        assert results[0].latency_ms == 99.0
        assert results[1].latency_ms == 200.0
        assert isinstance(results[0], RequestResult)


class TestValidateConfigMatch:
    def test_matching_configs(self):
        cfg = {"base_url": "http://x", "model": "m1"}
        assert validate_config_match(cfg, cfg) == []

    def test_mismatched_configs(self):
        old = {"base_url": "http://x", "model": "m1"}
        new = {"base_url": "http://y", "model": "m1"}
        mismatches = validate_config_match(old, new)
        assert len(mismatches) == 1
        assert "base_url" in mismatches[0]

    def test_missing_keys_ignored(self):
        old = {"model": "m1"}
        new = {"endpoint": "/v1/chat/completions"}
        assert validate_config_match(old, new) == []


class TestResumeMain:
    def test_resume_text_output(self, tmp_path, capsys):
        mgr = CheckpointManager(tmp_path / "ckpt", interval=1, config_snapshot={"model": "test"})
        mgr.record(_make_result())
        resume_main(["--checkpoint", str(tmp_path / "ckpt")])
        captured = capsys.readouterr()
        assert "Completed requests: 1" in captured.out

    def test_resume_json_output(self, tmp_path, capsys):
        mgr = CheckpointManager(tmp_path / "ckpt", interval=1, config_snapshot={"model": "test"})
        mgr.record(_make_result())
        resume_main(["--checkpoint", str(tmp_path / "ckpt"), "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["completed_count"] == 1

    def test_resume_not_found(self, tmp_path):
        with pytest.raises(SystemExit):
            resume_main(["--checkpoint", str(tmp_path / "nonexistent")])


class TestYAMLConfigKeys:
    def test_checkpoint_keys_in_known(self):
        from xpyd_bench.config_cmd import _KNOWN_KEYS
        assert "checkpoint_dir" in _KNOWN_KEYS
        assert "checkpoint_interval" in _KNOWN_KEYS

"""Tests for benchmark result archival & cloud storage (M58)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from xpyd_bench.archive import (
    ArchiveBackend,
    GCSArchiveBackend,
    LocalArchiveBackend,
    S3ArchiveBackend,
    archive_main,
    create_backend,
    discover_backends,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result_file(tmp_path: Path, name: str = "result.json", **overrides) -> Path:
    """Create a minimal benchmark result JSON file."""
    data = {
        "model": "test-model",
        "num_prompts": 10,
        "completed": 9,
        "failed": 1,
        "request_throughput": 5.0,
        "output_throughput": 100.0,
        "mean_ttft_ms": 50.0,
        "partial": False,
        "tags": {"env": "test"},
        "environment": {"python": "3.10"},
    }
    data.update(overrides)
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return p


# ---------------------------------------------------------------------------
# LocalArchiveBackend
# ---------------------------------------------------------------------------

class TestLocalArchiveBackend:
    def test_push_creates_manifest(self, tmp_path):
        archive_dir = tmp_path / "archive"
        result = _make_result_file(tmp_path)
        backend = LocalArchiveBackend(archive_dir)

        run_id = backend.push(result)

        assert len(run_id) == 12
        manifest = json.loads((archive_dir / "manifest.json").read_text())
        assert len(manifest) == 1
        assert manifest[0]["run_id"] == run_id
        assert manifest[0]["model"] == "test-model"
        assert manifest[0]["num_prompts"] == 10

    def test_push_copies_file(self, tmp_path):
        archive_dir = tmp_path / "archive"
        result = _make_result_file(tmp_path)
        backend = LocalArchiveBackend(archive_dir)

        run_id = backend.push(result)

        archived_files = [f for f in archive_dir.iterdir() if f.name != "manifest.json"]
        assert len(archived_files) == 1
        assert run_id in archived_files[0].name

    def test_push_multiple_runs(self, tmp_path):
        archive_dir = tmp_path / "archive"
        backend = LocalArchiveBackend(archive_dir)

        r1 = _make_result_file(tmp_path, "r1.json", model="m1")
        r2 = _make_result_file(tmp_path, "r2.json", model="m2")

        id1 = backend.push(r1)
        id2 = backend.push(r2)

        assert id1 != id2
        manifest = json.loads((archive_dir / "manifest.json").read_text())
        assert len(manifest) == 2

    def test_push_with_metadata(self, tmp_path):
        archive_dir = tmp_path / "archive"
        result = _make_result_file(tmp_path)
        backend = LocalArchiveBackend(archive_dir)

        backend.push(result, metadata={"note": "baseline run"})

        manifest = json.loads((archive_dir / "manifest.json").read_text())
        assert manifest[0]["user_metadata"]["note"] == "baseline run"

    def test_push_missing_file(self, tmp_path):
        archive_dir = tmp_path / "archive"
        backend = LocalArchiveBackend(archive_dir)

        with pytest.raises(FileNotFoundError):
            backend.push(tmp_path / "nonexistent.json")

    def test_list_runs_empty(self, tmp_path):
        backend = LocalArchiveBackend(tmp_path / "archive")
        assert backend.list_runs() == []

    def test_list_runs(self, tmp_path):
        archive_dir = tmp_path / "archive"
        backend = LocalArchiveBackend(archive_dir)
        r1 = _make_result_file(tmp_path, "r1.json")
        r2 = _make_result_file(tmp_path, "r2.json")
        backend.push(r1)
        backend.push(r2)

        runs = backend.list_runs()
        assert len(runs) == 2

    def test_list_runs_with_limit(self, tmp_path):
        archive_dir = tmp_path / "archive"
        backend = LocalArchiveBackend(archive_dir)
        for i in range(5):
            r = _make_result_file(tmp_path, f"r{i}.json")
            backend.push(r)

        runs = backend.list_runs(limit=2)
        assert len(runs) == 2

    def test_fetch_run(self, tmp_path):
        archive_dir = tmp_path / "archive"
        result = _make_result_file(tmp_path)
        backend = LocalArchiveBackend(archive_dir)
        run_id = backend.push(result)

        output = tmp_path / "fetched.json"
        fetched = backend.fetch(run_id, output)

        assert fetched == output
        assert output.exists()
        data = json.loads(output.read_text())
        assert data["model"] == "test-model"

    def test_fetch_unknown_id(self, tmp_path):
        backend = LocalArchiveBackend(tmp_path / "archive")
        with pytest.raises(KeyError, match="not found"):
            backend.fetch("nonexistent", tmp_path / "out.json")

    def test_manifest_survives_reload(self, tmp_path):
        archive_dir = tmp_path / "archive"
        result = _make_result_file(tmp_path)

        b1 = LocalArchiveBackend(archive_dir)
        run_id = b1.push(result)

        b2 = LocalArchiveBackend(archive_dir)
        runs = b2.list_runs()
        assert len(runs) == 1
        assert runs[0]["run_id"] == run_id


# ---------------------------------------------------------------------------
# S3/GCS backends (instantiation check — actual calls mocked)
# ---------------------------------------------------------------------------

class TestS3Backend:
    def test_requires_boto3(self):
        with patch.dict(sys.modules, {"boto3": None}):
            with pytest.raises(ImportError, match="boto3"):
                S3ArchiveBackend(bucket="test-bucket")

    def test_instantiation_with_boto3(self):
        mock_boto3 = MagicMock()
        mock_session = MagicMock()
        mock_boto3.Session.return_value = mock_session
        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            backend = S3ArchiveBackend(bucket="my-bucket", prefix="results/")
            assert backend.bucket == "my-bucket"
            assert backend.prefix == "results/"


class TestGCSBackend:
    def test_requires_google_cloud(self):
        with patch.dict(sys.modules, {"google.cloud": None, "google.cloud.storage": None}):
            with pytest.raises(ImportError, match="google-cloud-storage"):
                GCSArchiveBackend(bucket="test-bucket")

    def test_instantiation_with_gcs(self):
        mock_storage = MagicMock()
        mock_client = MagicMock()
        mock_storage.Client.return_value = mock_client
        with patch.dict(
            sys.modules,
            {
                "google": MagicMock(),
                "google.cloud": MagicMock(),
                "google.cloud.storage": mock_storage,
            },
        ):
            backend = GCSArchiveBackend(bucket="my-bucket")
            assert backend.prefix == "xpyd-bench/"


# ---------------------------------------------------------------------------
# Backend discovery
# ---------------------------------------------------------------------------

class TestDiscovery:
    def test_builtin_backends(self):
        backends = discover_backends()
        assert "local" in backends
        assert "s3" in backends
        assert "gcs" in backends

    def test_create_local_backend(self, tmp_path):
        backend = create_backend("local", archive_path=str(tmp_path / "archive"))
        assert isinstance(backend, LocalArchiveBackend)

    def test_create_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown archive backend"):
            create_backend("foobar")

    def test_create_local_without_path(self):
        with pytest.raises(ValueError, match="archive-path"):
            create_backend("local")

    def test_create_s3_without_bucket(self):
        # Mock boto3 to avoid import error
        mock_boto3 = MagicMock()
        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            with pytest.raises(ValueError, match="archive-bucket"):
                create_backend("s3")


# ---------------------------------------------------------------------------
# Plugin interface
# ---------------------------------------------------------------------------

class TestPluginInterface:
    def test_abstract_methods(self):
        """ArchiveBackend cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ArchiveBackend()

    def test_custom_backend_subclass(self, tmp_path):
        class MyBackend(ArchiveBackend):
            def push(self, result_path, metadata=None):
                return "custom-id"

            def list_runs(self, limit=None):
                return [{"run_id": "custom-id"}]

            def fetch(self, run_id, output_path):
                return Path(output_path)

        b = MyBackend()
        assert b.push("dummy") == "custom-id"
        assert b.list_runs() == [{"run_id": "custom-id"}]


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

class TestArchiveCLI:
    def test_archive_list_local(self, tmp_path, capsys):
        archive_dir = tmp_path / "archive"
        backend = LocalArchiveBackend(archive_dir)
        r = _make_result_file(tmp_path)
        backend.push(r)

        archive_main(["list", "--archive", "local", "--archive-path", str(archive_dir)])
        out = capsys.readouterr().out
        assert "test-model" in out

    def test_archive_list_json(self, tmp_path, capsys):
        archive_dir = tmp_path / "archive"
        backend = LocalArchiveBackend(archive_dir)
        r = _make_result_file(tmp_path)
        backend.push(r)

        archive_main([
            "list", "--archive", "local",
            "--archive-path", str(archive_dir), "--json",
        ])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert len(data) == 1
        assert data[0]["model"] == "test-model"

    def test_archive_fetch_local(self, tmp_path, capsys):
        archive_dir = tmp_path / "archive"
        backend = LocalArchiveBackend(archive_dir)
        r = _make_result_file(tmp_path)
        run_id = backend.push(r)

        output = tmp_path / "fetched.json"
        archive_main([
            "fetch", run_id,
            "--archive", "local",
            "--archive-path", str(archive_dir),
            "--output", str(output),
        ])
        assert output.exists()
        out = capsys.readouterr().out
        assert run_id in out

    def test_archive_fetch_unknown_id(self, tmp_path):
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()
        (archive_dir / "manifest.json").write_text("[]")

        with pytest.raises(SystemExit):
            archive_main([
                "fetch", "nonexistent",
                "--archive", "local",
                "--archive-path", str(archive_dir),
                "--output", str(tmp_path / "out.json"),
            ])

    def test_archive_no_action(self, capsys):
        with pytest.raises(SystemExit):
            archive_main([])

    def test_archive_list_empty(self, tmp_path, capsys):
        archive_dir = tmp_path / "archive"
        archive_main(["list", "--archive", "local", "--archive-path", str(archive_dir)])
        out = capsys.readouterr().out
        assert "No archived runs" in out

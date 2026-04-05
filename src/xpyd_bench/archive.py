"""Benchmark result archival & cloud storage (M58).

Provides:
- ``ArchiveBackend`` base class for custom storage backends
- ``LocalArchiveBackend`` for local filesystem archival
- ``S3ArchiveBackend`` for Amazon S3
- ``GCSArchiveBackend`` for Google Cloud Storage
- CLI subcommands: ``xpyd-bench archive list`` and ``xpyd-bench archive fetch``
- Plugin discovery via ``xpyd.archives`` entry point group
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path


class ArchiveBackend(ABC):
    """Base class for archive storage backends."""

    @abstractmethod
    def push(self, result_path: str | Path, metadata: dict | None = None) -> str:
        """Archive a result file. Returns the run ID."""

    @abstractmethod
    def list_runs(self, limit: int | None = None) -> list[dict]:
        """List archived runs. Returns list of manifest entries."""

    @abstractmethod
    def fetch(self, run_id: str, output_path: str | Path) -> Path:
        """Fetch an archived run by ID. Returns path to the fetched file."""


# ---------------------------------------------------------------------------
# Local filesystem backend
# ---------------------------------------------------------------------------

class LocalArchiveBackend(ArchiveBackend):
    """Archive results to a local directory with a JSON manifest."""

    def __init__(self, archive_path: str | Path) -> None:
        self.archive_dir = Path(archive_path)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.archive_dir / "manifest.json"

    def _load_manifest(self) -> list[dict]:
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _save_manifest(self, manifest: list[dict]) -> None:
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

    def push(self, result_path: str | Path, metadata: dict | None = None) -> str:
        result_path = Path(result_path)
        if not result_path.exists():
            raise FileNotFoundError(f"Result file not found: {result_path}")

        run_id = uuid.uuid4().hex[:12]
        archived_name = f"{run_id}_{result_path.name}"
        dest = self.archive_dir / archived_name
        shutil.copy2(result_path, dest)

        # Load result to extract summary metadata
        try:
            with open(result_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}

        entry = {
            "run_id": run_id,
            "filename": archived_name,
            "original_name": result_path.name,
            "archived_at": datetime.now(timezone.utc).isoformat(),
            "model": data.get("model", ""),
            "num_prompts": data.get("num_prompts", 0),
            "completed": data.get("completed", 0),
            "failed": data.get("failed", 0),
            "request_throughput": data.get("request_throughput"),
            "output_throughput": data.get("output_throughput"),
            "mean_ttft_ms": data.get("mean_ttft_ms"),
            "partial": data.get("partial", False),
            "tags": data.get("tags", {}),
            "environment": data.get("environment", {}),
        }
        if metadata:
            entry["user_metadata"] = metadata

        manifest = self._load_manifest()
        manifest.append(entry)
        self._save_manifest(manifest)
        return run_id

    def list_runs(self, limit: int | None = None) -> list[dict]:
        manifest = self._load_manifest()
        manifest.sort(key=lambda e: e.get("archived_at", ""))
        if limit is not None and limit > 0:
            manifest = manifest[-limit:]
        return manifest

    def fetch(self, run_id: str, output_path: str | Path) -> Path:
        manifest = self._load_manifest()
        entry = None
        for e in manifest:
            if e["run_id"] == run_id:
                entry = e
                break
        if entry is None:
            raise KeyError(f"Run ID not found in archive: {run_id}")

        src = self.archive_dir / entry["filename"]
        if not src.exists():
            raise FileNotFoundError(f"Archived file missing: {src}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, output_path)
        return output_path


# ---------------------------------------------------------------------------
# S3 backend (requires boto3)
# ---------------------------------------------------------------------------

class S3ArchiveBackend(ArchiveBackend):
    """Archive results to Amazon S3."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "xpyd-bench/",
        region: str | None = None,
    ) -> None:
        try:
            import boto3  # noqa: F401
        except ImportError:
            raise ImportError(
                "boto3 is required for S3 archival. Install with: pip install boto3"
            )
        self.bucket = bucket
        self.prefix = prefix.rstrip("/") + "/"
        session = boto3.Session(region_name=region)
        self.s3 = session.client("s3")
        self._manifest_key = f"{self.prefix}manifest.json"

    def _load_manifest(self) -> list[dict]:
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=self._manifest_key)
            return json.loads(resp["Body"].read())
        except self.s3.exceptions.NoSuchKey:
            return []
        except Exception:
            return []

    def _save_manifest(self, manifest: list[dict]) -> None:
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self._manifest_key,
            Body=json.dumps(manifest, indent=2, default=str).encode(),
            ContentType="application/json",
        )

    def push(self, result_path: str | Path, metadata: dict | None = None) -> str:
        result_path = Path(result_path)
        if not result_path.exists():
            raise FileNotFoundError(f"Result file not found: {result_path}")

        run_id = uuid.uuid4().hex[:12]
        key = f"{self.prefix}{run_id}_{result_path.name}"
        self.s3.upload_file(str(result_path), self.bucket, key)

        try:
            with open(result_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}

        entry = {
            "run_id": run_id,
            "s3_key": key,
            "original_name": result_path.name,
            "archived_at": datetime.now(timezone.utc).isoformat(),
            "model": data.get("model", ""),
            "num_prompts": data.get("num_prompts", 0),
            "completed": data.get("completed", 0),
            "tags": data.get("tags", {}),
        }
        if metadata:
            entry["user_metadata"] = metadata

        manifest = self._load_manifest()
        manifest.append(entry)
        self._save_manifest(manifest)
        return run_id

    def list_runs(self, limit: int | None = None) -> list[dict]:
        manifest = self._load_manifest()
        manifest.sort(key=lambda e: e.get("archived_at", ""))
        if limit is not None and limit > 0:
            manifest = manifest[-limit:]
        return manifest

    def fetch(self, run_id: str, output_path: str | Path) -> Path:
        manifest = self._load_manifest()
        entry = None
        for e in manifest:
            if e["run_id"] == run_id:
                entry = e
                break
        if entry is None:
            raise KeyError(f"Run ID not found in archive: {run_id}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file(self.bucket, entry["s3_key"], str(output_path))
        return output_path


# ---------------------------------------------------------------------------
# GCS backend (requires google-cloud-storage)
# ---------------------------------------------------------------------------

class GCSArchiveBackend(ArchiveBackend):
    """Archive results to Google Cloud Storage."""

    def __init__(self, bucket: str, prefix: str = "xpyd-bench/") -> None:
        try:
            from google.cloud import storage as gcs_storage  # noqa: F401
        except ImportError:
            raise ImportError(
                "google-cloud-storage is required for GCS archival. "
                "Install with: pip install google-cloud-storage"
            )
        client = gcs_storage.Client()
        self.bucket_obj = client.bucket(bucket)
        self.prefix = prefix.rstrip("/") + "/"
        self._manifest_key = f"{self.prefix}manifest.json"

    def _load_manifest(self) -> list[dict]:
        blob = self.bucket_obj.blob(self._manifest_key)
        try:
            data = blob.download_as_text()
            return json.loads(data)
        except Exception:
            return []

    def _save_manifest(self, manifest: list[dict]) -> None:
        blob = self.bucket_obj.blob(self._manifest_key)
        blob.upload_from_string(
            json.dumps(manifest, indent=2, default=str),
            content_type="application/json",
        )

    def push(self, result_path: str | Path, metadata: dict | None = None) -> str:
        result_path = Path(result_path)
        if not result_path.exists():
            raise FileNotFoundError(f"Result file not found: {result_path}")

        run_id = uuid.uuid4().hex[:12]
        key = f"{self.prefix}{run_id}_{result_path.name}"
        blob = self.bucket_obj.blob(key)
        blob.upload_from_filename(str(result_path))

        try:
            with open(result_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}

        entry = {
            "run_id": run_id,
            "gcs_key": key,
            "original_name": result_path.name,
            "archived_at": datetime.now(timezone.utc).isoformat(),
            "model": data.get("model", ""),
            "num_prompts": data.get("num_prompts", 0),
            "completed": data.get("completed", 0),
            "tags": data.get("tags", {}),
        }
        if metadata:
            entry["user_metadata"] = metadata

        manifest = self._load_manifest()
        manifest.append(entry)
        self._save_manifest(manifest)
        return run_id

    def list_runs(self, limit: int | None = None) -> list[dict]:
        manifest = self._load_manifest()
        manifest.sort(key=lambda e: e.get("archived_at", ""))
        if limit is not None and limit > 0:
            manifest = manifest[-limit:]
        return manifest

    def fetch(self, run_id: str, output_path: str | Path) -> Path:
        manifest = self._load_manifest()
        entry = None
        for e in manifest:
            if e["run_id"] == run_id:
                entry = e
                break
        if entry is None:
            raise KeyError(f"Run ID not found in archive: {run_id}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        blob = self.bucket_obj.blob(entry["gcs_key"])
        blob.download_to_filename(str(output_path))
        return output_path


# ---------------------------------------------------------------------------
# Backend discovery & factory
# ---------------------------------------------------------------------------

def discover_backends() -> dict[str, type[ArchiveBackend]]:
    """Discover archive backends from entry points and built-ins."""
    backends: dict[str, type[ArchiveBackend]] = {
        "local": LocalArchiveBackend,
        "s3": S3ArchiveBackend,
        "gcs": GCSArchiveBackend,
    }
    try:
        eps = importlib.metadata.entry_points()
        group = eps.get("xpyd.archives", []) if isinstance(eps, dict) else eps
        if not isinstance(eps, dict):
            group = [ep for ep in eps if ep.group == "xpyd.archives"]
        for ep in group:
            try:
                backends[ep.name] = ep.load()
            except Exception:
                pass
    except Exception:
        pass
    return backends


def create_backend(
    backend_type: str,
    archive_path: str | None = None,
    archive_bucket: str | None = None,
    archive_prefix: str = "xpyd-bench/",
    archive_region: str | None = None,
) -> ArchiveBackend:
    """Create an archive backend instance by type name."""
    backends = discover_backends()
    if backend_type not in backends:
        raise ValueError(
            f"Unknown archive backend '{backend_type}'. "
            f"Available: {', '.join(sorted(backends))}"
        )

    cls = backends[backend_type]
    if backend_type == "local":
        if not archive_path:
            raise ValueError("--archive-path is required for local archive backend")
        return cls(archive_path=archive_path)
    elif backend_type == "s3":
        if not archive_bucket:
            raise ValueError("--archive-bucket is required for S3 archive backend")
        return cls(bucket=archive_bucket, prefix=archive_prefix, region=archive_region)
    elif backend_type == "gcs":
        if not archive_bucket:
            raise ValueError("--archive-bucket is required for GCS archive backend")
        return cls(bucket=archive_bucket, prefix=archive_prefix)
    else:
        # Custom plugin — try common constructor signatures
        try:
            return cls(path=archive_path, bucket=archive_bucket)
        except TypeError:
            return cls()


# ---------------------------------------------------------------------------
# CLI: archive list / archive fetch
# ---------------------------------------------------------------------------

def _format_archive_table(entries: list[dict]) -> str:
    """Format archive manifest entries as a human-readable table."""
    if not entries:
        return "No archived runs found."

    lines: list[str] = []
    lines.append(
        f"{'Run ID':<14} {'Archived At':<22} {'Model':<16} "
        f"{'Prompts':>7} {'OK':>5} {'Partial':>7} {'Original Name'}"
    )
    lines.append("-" * 100)

    for e in entries:
        ts = e.get("archived_at", "?")[:19]
        model = (e.get("model") or "?")[:15]
        partial = "yes" if e.get("partial") else ""
        lines.append(
            f"{e['run_id']:<14} {ts:<22} {model:<16} "
            f"{e.get('num_prompts', 0):>7} {e.get('completed', 0):>5} "
            f"{partial:>7} {e.get('original_name', '?')}"
        )

    return "\n".join(lines)


def archive_main(argv: list[str] | None = None) -> None:
    """Entry point for ``xpyd-bench archive <list|fetch>`` subcommand."""
    parser = argparse.ArgumentParser(
        prog="xpyd-bench archive",
        description="Manage archived benchmark results",
    )
    sub = parser.add_subparsers(dest="action", help="Archive action")

    # list
    list_parser = sub.add_parser("list", help="List archived runs")
    list_parser.add_argument("--archive-path", type=str, help="Local archive directory")
    list_parser.add_argument("--archive-bucket", type=str, help="S3/GCS bucket name")
    list_parser.add_argument(
        "--archive", type=str, default="local", help="Backend type (local/s3/gcs)"
    )
    list_parser.add_argument("--archive-prefix", type=str, default="xpyd-bench/")
    list_parser.add_argument("--archive-region", type=str, default=None)
    list_parser.add_argument("--last", type=int, default=None, metavar="N")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    # fetch
    fetch_parser = sub.add_parser("fetch", help="Fetch an archived run by ID")
    fetch_parser.add_argument("run_id", help="Run ID to fetch")
    fetch_parser.add_argument("--archive-path", type=str, help="Local archive directory")
    fetch_parser.add_argument("--archive-bucket", type=str, help="S3/GCS bucket name")
    fetch_parser.add_argument(
        "--archive", type=str, default="local", help="Backend type (local/s3/gcs)"
    )
    fetch_parser.add_argument("--archive-prefix", type=str, default="xpyd-bench/")
    fetch_parser.add_argument("--archive-region", type=str, default=None)
    fetch_parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output file path"
    )

    args = parser.parse_args(argv)

    if not args.action:
        parser.print_help()
        sys.exit(1)

    try:
        backend = create_backend(
            backend_type=args.archive,
            archive_path=getattr(args, "archive_path", None),
            archive_bucket=getattr(args, "archive_bucket", None),
            archive_prefix=getattr(args, "archive_prefix", "xpyd-bench/"),
            archive_region=getattr(args, "archive_region", None),
        )
    except (ValueError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.action == "list":
        entries = backend.list_runs(limit=args.last)
        if args.json:
            print(json.dumps(entries, indent=2, default=str))
        else:
            print(_format_archive_table(entries))

    elif args.action == "fetch":
        try:
            out = backend.fetch(args.run_id, args.output)
            print(f"Fetched run {args.run_id} → {out}")
        except (KeyError, FileNotFoundError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

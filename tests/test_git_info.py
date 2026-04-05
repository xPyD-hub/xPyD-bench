"""Tests for git metadata capture (M79)."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

from xpyd_bench.bench.env import collect_git_info
from xpyd_bench.bench.models import BenchmarkResult


class TestCollectGitInfo:
    """Tests for collect_git_info()."""

    def test_returns_dict_in_git_repo(self) -> None:
        """When running inside a git repo, returns metadata dict."""
        info = collect_git_info()
        # This test suite runs inside xPyD-bench which is a git repo
        assert info is not None
        assert "commit" in info
        assert "commit_short" in info
        assert "branch" in info
        assert "dirty" in info
        assert "remote_url" in info
        # commit should be a 40-char hex string
        assert len(info["commit"]) == 40
        assert len(info["commit_short"]) >= 7
        # dirty is "True" or "False"
        assert info["dirty"] in ("True", "False")

    def test_returns_none_when_not_in_repo(self) -> None:
        """When git rev-parse fails, returns None."""
        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(128, "git")):
            assert collect_git_info() is None

    def test_returns_none_when_git_not_installed(self) -> None:
        """When git binary is not found, returns None."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert collect_git_info() is None

    def test_returns_none_on_timeout(self) -> None:
        """When git command times out, returns None."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("git", 5)):
            assert collect_git_info() is None


class TestBenchmarkResultGitInfo:
    """Tests for git_info field on BenchmarkResult."""

    def test_default_none(self) -> None:
        """git_info defaults to None."""
        r = BenchmarkResult()
        assert r.git_info is None

    def test_set_git_info(self) -> None:
        """Can set git_info on BenchmarkResult."""
        info = {"commit": "abc123", "branch": "main", "dirty": "False"}
        r = BenchmarkResult(git_info=info)
        assert r.git_info == info
        assert r.git_info["branch"] == "main"


class TestGitInfoSerialization:
    """Tests for git_info in JSON output."""

    def test_git_info_in_result_dict(self) -> None:
        """git_info is included in serialized result dict."""
        from xpyd_bench.bench.runner import _to_dict as result_to_dict

        info = {"commit": "abc123", "branch": "main", "dirty": "False", "remote_url": "https://example.com"}
        r = BenchmarkResult(git_info=info)
        d = result_to_dict(r)
        assert d["git_info"] == info

    def test_git_info_omitted_when_none(self) -> None:
        """git_info is not in serialized dict when None."""
        from xpyd_bench.bench.runner import _to_dict as result_to_dict

        r = BenchmarkResult()
        d = result_to_dict(r)
        assert "git_info" not in d

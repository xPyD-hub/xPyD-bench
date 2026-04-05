"""Tests for benchmark scheduling & cron integration (M63)."""

from __future__ import annotations

from datetime import datetime

import pytest

from xpyd_bench.schedule import (
    generate_crontab_entry,
    next_cron_occurrences,
    run_on_complete,
    schedule_main,
    validate_cron_expression,
)


class TestValidateCronExpression:
    def test_valid_simple(self):
        assert validate_cron_expression("0 */6 * * *") == []

    def test_valid_complex(self):
        assert validate_cron_expression("30 2 1,15 * 0-5") == []

    def test_valid_all_stars(self):
        assert validate_cron_expression("* * * * *") == []

    def test_invalid_field_count(self):
        errors = validate_cron_expression("0 * *")
        assert len(errors) == 1
        assert "Expected 5 fields" in errors[0]

    def test_invalid_range(self):
        errors = validate_cron_expression("60 * * * *")
        assert len(errors) == 1
        assert "out of bounds" in errors[0]

    def test_invalid_non_numeric(self):
        errors = validate_cron_expression("abc * * * *")
        assert len(errors) == 1
        assert "Non-numeric" in errors[0]

    def test_invalid_step(self):
        errors = validate_cron_expression("*/0 * * * *")
        assert len(errors) == 1

    def test_invalid_range_reversed(self):
        errors = validate_cron_expression("* 10-5 * * *")
        assert len(errors) == 1
        assert "start > end" in errors[0]


class TestNextCronOccurrences:
    def test_every_6_hours(self):
        after = datetime(2025, 1, 1, 0, 0, 0)
        results = next_cron_occurrences("0 */6 * * *", count=4, after=after)
        assert len(results) == 4
        assert results[0] == datetime(2025, 1, 1, 6, 0)
        assert results[1] == datetime(2025, 1, 1, 12, 0)
        assert results[2] == datetime(2025, 1, 1, 18, 0)
        assert results[3] == datetime(2025, 1, 2, 0, 0)

    def test_specific_time(self):
        after = datetime(2025, 6, 15, 8, 0, 0)
        results = next_cron_occurrences("30 9 * * *", count=2, after=after)
        assert len(results) == 2
        assert results[0] == datetime(2025, 6, 15, 9, 30)
        assert results[1] == datetime(2025, 6, 16, 9, 30)

    def test_invalid_expression(self):
        assert next_cron_occurrences("bad") == []

    def test_count_parameter(self):
        results = next_cron_occurrences("* * * * *", count=3, after=datetime(2025, 1, 1, 0, 0))
        assert len(results) == 3


class TestGenerateCrontabEntry:
    def test_basic(self):
        entry = generate_crontab_entry("0 */6 * * *")
        assert "0 */6 * * *" in entry
        assert "xpyd-bench" in entry
        assert "run" in entry

    def test_with_config(self):
        entry = generate_crontab_entry("0 0 * * *", config_path="/etc/bench.yaml")
        assert "--config /etc/bench.yaml" in entry

    def test_with_extra_args(self):
        entry = generate_crontab_entry("0 0 * * *", extra_args="--model gpt-4")
        assert "--model gpt-4" in entry


class TestRunOnComplete:
    def test_success(self):
        result = run_on_complete("echo hello")
        assert result.returncode == 0
        assert "hello" in result.stdout

    def test_failure(self):
        result = run_on_complete("exit 1")
        assert result.returncode == 1

    def test_output_capture(self):
        result = run_on_complete("echo stdout_test && echo stderr_test >&2")
        assert "stdout_test" in result.stdout
        assert "stderr_test" in result.stderr


class TestScheduleMain:
    def test_valid_schedule(self, capsys):
        schedule_main(["--cron", "0 */6 * * *"])
        captured = capsys.readouterr()
        assert "Crontab entry:" in captured.out
        assert "Next" in captured.out

    def test_invalid_cron_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            schedule_main(["--cron", "60 * * * *"])
        assert exc_info.value.code == 1

    def test_with_config(self, capsys):
        schedule_main(["--cron", "0 0 * * *", "--config", "bench.yaml"])
        captured = capsys.readouterr()
        assert "--config bench.yaml" in captured.out

    def test_preview_count(self, capsys):
        schedule_main(["--cron", "0 0 * * *", "--preview", "3"])
        captured = capsys.readouterr()
        assert "Next 3 occurrences:" in captured.out


class TestOnCompleteCliArg:
    def test_on_complete_in_known_keys(self):
        from xpyd_bench.config_cmd import _KNOWN_KEYS

        assert "on_complete" in _KNOWN_KEYS

    def test_on_complete_cli_arg(self):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--on-complete", "echo done"])
        assert args.on_complete == "echo done"


class TestScheduleSubcommand:
    def test_schedule_in_subcommands(self):
        from xpyd_bench.main import _SUBCOMMANDS

        assert "schedule" in _SUBCOMMANDS

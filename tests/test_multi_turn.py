"""Tests for multi-turn conversation benchmarking (M45)."""

from __future__ import annotations

import json
from pathlib import Path

from xpyd_bench.multi_turn import (
    ConversationResult,
    TurnResult,
    compute_multi_turn_stats,
    generate_synthetic_conversations,
    load_multi_turn_dataset,
)


class TestLoadMultiTurnDataset:
    """Tests for JSONL dataset loading."""

    def test_load_valid_jsonl(self, tmp_path: Path) -> None:
        data = tmp_path / "conversations.jsonl"
        conv1 = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ]
        }
        conv2 = {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
            ]
        }
        data.write_text(json.dumps(conv1) + "\n" + json.dumps(conv2) + "\n")

        result = load_multi_turn_dataset(str(data))
        assert len(result) == 2
        assert len(result[0]) == 4
        assert len(result[1]) == 1
        assert result[0][0]["role"] == "system"

    def test_load_empty_messages_skipped(self, tmp_path: Path) -> None:
        data = tmp_path / "empty.jsonl"
        data.write_text(json.dumps({"messages": []}) + "\n")
        result = load_multi_turn_dataset(str(data))
        assert len(result) == 0


class TestGenerateSyntheticConversations:
    """Tests for synthetic conversation generation."""

    def test_basic_generation(self) -> None:
        convs = generate_synthetic_conversations(
            num_conversations=3, turns=4, input_len=100, seed=42
        )
        assert len(convs) == 3
        for conv in convs:
            # Should have system + alternating user/assistant
            assert conv[0]["role"] == "system"
            user_turns = [m for m in conv if m["role"] == "user"]
            assert len(user_turns) == 4

    def test_deterministic_with_seed(self) -> None:
        a = generate_synthetic_conversations(num_conversations=2, turns=3, seed=123)
        b = generate_synthetic_conversations(num_conversations=2, turns=3, seed=123)
        assert a == b

    def test_different_seeds_differ(self) -> None:
        a = generate_synthetic_conversations(num_conversations=2, turns=3, seed=1)
        b = generate_synthetic_conversations(num_conversations=2, turns=3, seed=2)
        assert a != b


class TestTurnResult:
    """Tests for TurnResult serialization."""

    def test_to_dict(self) -> None:
        tr = TurnResult(
            turn_index=0,
            ttft_ms=15.5,
            total_latency_ms=120.3,
            prompt_tokens=50,
            completion_tokens=30,
            context_tokens=50,
        )
        d = tr.to_dict()
        assert d["turn_index"] == 0
        assert d["ttft_ms"] == 15.5
        assert "error" not in d

    def test_to_dict_with_error(self) -> None:
        tr = TurnResult(
            turn_index=1,
            ttft_ms=0,
            total_latency_ms=0,
            prompt_tokens=100,
            completion_tokens=0,
            context_tokens=100,
            error="Connection refused",
        )
        d = tr.to_dict()
        assert d["error"] == "Connection refused"


class TestConversationResult:
    """Tests for ConversationResult serialization."""

    def test_to_dict(self) -> None:
        turn = TurnResult(0, 10.0, 100.0, 50, 30, 50)
        conv = ConversationResult(
            conversation_id=0,
            turns=[turn],
            total_turns=1,
            total_latency_ms=100.0,
        )
        d = conv.to_dict()
        assert d["conversation_id"] == 0
        assert d["total_turns"] == 1
        assert len(d["turns"]) == 1


class TestComputeMultiTurnStats:
    """Tests for aggregate statistics computation."""

    def _make_conv(
        self, conv_id: int, turns: list[tuple[float, float, int]]
    ) -> ConversationResult:
        """Helper: create a ConversationResult from (ttft, latency, ctx_tokens) tuples."""
        result = ConversationResult(conversation_id=conv_id)
        for i, (ttft, lat, ctx) in enumerate(turns):
            result.turns.append(
                TurnResult(
                    turn_index=i,
                    ttft_ms=ttft,
                    total_latency_ms=lat,
                    prompt_tokens=ctx,
                    completion_tokens=30,
                    context_tokens=ctx,
                )
            )
        result.total_turns = len(turns)
        result.total_latency_ms = sum(t[1] for t in turns)
        return result

    def test_aggregate_stats(self) -> None:
        conv1 = self._make_conv(0, [(10, 100, 50), (15, 150, 100)])
        conv2 = self._make_conv(1, [(12, 110, 55), (18, 160, 110)])
        mt = compute_multi_turn_stats([conv1, conv2])

        assert mt.aggregate_stats["total_conversations"] == 2
        assert mt.aggregate_stats["total_turns"] == 4
        assert mt.aggregate_stats["total_errors"] == 0
        assert mt.aggregate_stats["mean_ttft_ms"] > 0
        assert mt.aggregate_stats["mean_latency_ms"] > 0

    def test_per_turn_stats(self) -> None:
        conv1 = self._make_conv(0, [(10, 100, 50), (20, 200, 120)])
        conv2 = self._make_conv(1, [(12, 110, 55), (22, 210, 125)])
        mt = compute_multi_turn_stats([conv1, conv2])

        assert 0 in mt.per_turn_stats
        assert 1 in mt.per_turn_stats
        # Turn 0 mean TTFT = (10 + 12) / 2 = 11
        assert mt.per_turn_stats[0]["mean_ttft_ms"] == 11.0
        assert mt.per_turn_stats[0]["count"] == 2
        # Turn 1 mean context should be (120 + 125) / 2 = 122.5
        assert mt.per_turn_stats[1]["mean_context_tokens"] == 122.5

    def test_error_turns_excluded(self) -> None:
        conv = ConversationResult(conversation_id=0)
        conv.turns.append(
            TurnResult(0, 10.0, 100.0, 50, 30, 50)
        )
        conv.turns.append(
            TurnResult(1, 0, 0, 100, 0, 100, error="timeout")
        )
        conv.total_turns = 2
        conv.total_latency_ms = 100.0

        mt = compute_multi_turn_stats([conv])
        assert mt.aggregate_stats["total_errors"] == 1
        assert mt.aggregate_stats["total_turns"] == 2
        # Only turn 0 in per_turn_stats
        assert 0 in mt.per_turn_stats
        assert 1 not in mt.per_turn_stats

    def test_to_dict_structure(self) -> None:
        conv = self._make_conv(0, [(10, 100, 50)])
        mt = compute_multi_turn_stats([conv])
        d = mt.to_dict()
        assert "conversations" in d
        assert "per_turn_stats" in d
        assert "aggregate_stats" in d

    def test_percentiles(self) -> None:
        # Create enough data points for percentiles
        turns = [(float(i), float(i * 10), i * 5) for i in range(1, 21)]
        conv = self._make_conv(0, turns)
        mt = compute_multi_turn_stats([conv])
        assert "p50_latency_ms" in mt.aggregate_stats
        assert "p99_latency_ms" in mt.aggregate_stats

    def test_context_growth_visible(self) -> None:
        """Per-turn stats should show increasing context tokens."""
        conv = self._make_conv(0, [
            (10, 100, 50),
            (15, 150, 120),
            (20, 200, 200),
        ])
        mt = compute_multi_turn_stats([conv])
        ctx_0 = mt.per_turn_stats[0]["mean_context_tokens"]
        ctx_1 = mt.per_turn_stats[1]["mean_context_tokens"]
        ctx_2 = mt.per_turn_stats[2]["mean_context_tokens"]
        assert ctx_0 < ctx_1 < ctx_2


class TestMultiTurnCLI:
    """Tests for multi-turn CLI integration."""

    def test_multi_turn_arg_added(self) -> None:
        """--multi-turn argument exists in CLI parser."""
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--multi-turn", "test.jsonl", "--max-turns", "3"])
        assert args.multi_turn == "test.jsonl"
        assert args.max_turns == 3

    def test_turns_arg_default(self) -> None:
        """--turns defaults to 5."""
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args([])
        assert args.turns == 5

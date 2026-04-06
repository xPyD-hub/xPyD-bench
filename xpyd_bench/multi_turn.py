"""Multi-turn conversation benchmarking (M45).

Supports sequential multi-turn conversations where each turn builds
on the previous context, measuring per-turn and aggregate metrics.

Usage:
    xpyd-bench run --multi-turn conversations.jsonl
    xpyd-bench run --multi-turn synthetic --turns 5 --input-len 256
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx


@dataclass
class TurnResult:
    """Metrics for a single conversation turn."""

    turn_index: int
    ttft_ms: float
    total_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    context_tokens: int  # total tokens in context at this turn
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "turn_index": self.turn_index,
            "ttft_ms": round(self.ttft_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "context_tokens": self.context_tokens,
        }
        if self.error:
            d["error"] = self.error
        return d


@dataclass
class ConversationResult:
    """Result for a full multi-turn conversation."""

    conversation_id: int
    turns: list[TurnResult] = field(default_factory=list)
    total_turns: int = 0
    total_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "total_turns": self.total_turns,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "turns": [t.to_dict() for t in self.turns],
        }


@dataclass
class MultiTurnResult:
    """Aggregated multi-turn benchmark results."""

    conversations: list[ConversationResult] = field(default_factory=list)
    per_turn_stats: dict[int, dict[str, float]] = field(default_factory=dict)
    aggregate_stats: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversations": [c.to_dict() for c in self.conversations],
            "per_turn_stats": self.per_turn_stats,
            "aggregate_stats": self.aggregate_stats,
        }


def load_multi_turn_dataset(path: str) -> list[list[dict[str, str]]]:
    """Load multi-turn conversations from JSONL file.

    Each line is a JSON object with a ``messages`` array in OpenAI chat
    format (list of ``{"role": ..., "content": ...}`` dicts).

    Returns:
        List of conversations, each a list of message dicts.
    """
    conversations: list[list[dict[str, str]]] = []
    p = Path(path)
    for line in p.read_text().strip().splitlines():
        obj = json.loads(line)
        messages = obj.get("messages", [])
        if messages:
            conversations.append(messages)
    return conversations


def generate_synthetic_conversations(
    num_conversations: int = 10,
    turns: int = 5,
    input_len: int = 256,
    seed: int | None = None,
) -> list[list[dict[str, str]]]:
    """Generate synthetic multi-turn conversations.

    Args:
        num_conversations: Number of conversations to generate.
        turns: Number of user turns per conversation.
        input_len: Approximate token count per user message.
        seed: Random seed for reproducibility.

    Returns:
        List of conversations in OpenAI chat format.
    """
    rng = random.Random(seed)
    words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her",
        "she", "or", "an", "will", "my", "one", "all", "would", "there",
        "their", "what", "so", "up", "out", "if", "about", "who", "get",
        "which", "go", "me", "when", "make", "can", "like", "time", "no",
    ]
    conversations: list[list[dict[str, str]]] = []
    for _ in range(num_conversations):
        msgs: list[dict[str, str]] = []
        msgs.append({
            "role": "system",
            "content": "You are a helpful assistant.",
        })
        for t in range(turns):
            # Generate user message
            word_count = max(1, input_len // 4)  # rough tokens-to-words
            user_text = " ".join(rng.choices(words, k=word_count))
            msgs.append({"role": "user", "content": user_text})
            # Placeholder for assistant response (will be filled by server)
            if t < turns - 1:
                msgs.append({
                    "role": "assistant",
                    "content": " ".join(rng.choices(words, k=word_count // 2)),
                })
        conversations.append(msgs)
    return conversations


def _estimate_tokens(text: str) -> int:
    """Rough token estimation (word count * 1.3)."""
    return max(1, int(len(text.split()) * 1.3))


async def run_conversation(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    conversation_id: int,
    max_turns: int | None = None,
    endpoint: str = "/v1/chat/completions",
    api_key: str | None = None,
    timeout: float = 300.0,
) -> ConversationResult:
    """Execute a single multi-turn conversation.

    Sends user messages sequentially, appending assistant responses to
    context before sending the next turn.

    Args:
        client: HTTP client.
        base_url: Server base URL.
        model: Model name.
        messages: Full conversation messages (user + assistant placeholders).
        conversation_id: ID for tracking.
        max_turns: Maximum turns to execute.
        endpoint: API endpoint.
        api_key: Optional API key.
        timeout: Per-request timeout.

    Returns:
        ConversationResult with per-turn metrics.
    """
    result = ConversationResult(conversation_id=conversation_id)
    url = f"{base_url.rstrip('/')}{endpoint}"

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Build conversation context incrementally
    context: list[dict[str, str]] = []
    turn_index = 0
    conv_start = time.monotonic()

    for msg in messages:
        if msg["role"] == "system":
            context.append(msg)
            continue
        if msg["role"] == "assistant":
            # Skip pre-filled assistant messages; we use server responses
            continue
        if msg["role"] == "user":
            if max_turns is not None and turn_index >= max_turns:
                break

            context.append(msg)
            context_tokens = sum(_estimate_tokens(m["content"]) for m in context)

            payload = {
                "model": model,
                "messages": context,
                "max_tokens": 128,
                "stream": True,
            }

            ttft_ms = 0.0
            total_latency_ms = 0.0
            completion_tokens = 0
            prompt_tokens = context_tokens
            error: str | None = None
            assistant_text = ""

            req_start = time.monotonic()
            first_token_time: float | None = None

            try:
                async with client.stream(
                    "POST",
                    url,
                    json=payload,
                    headers=headers,
                    timeout=timeout,
                ) as resp:
                    resp.raise_for_status()
                    async for raw_line in resp.aiter_lines():
                        line = raw_line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[len("data:"):].strip()
                        if data_str == "[DONE]":
                            break
                        if first_token_time is None:
                            first_token_time = time.monotonic()
                        try:
                            chunk = json.loads(data_str)
                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    assistant_text += content
                                    completion_tokens += 1
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                error = str(e)

            req_end = time.monotonic()
            total_latency_ms = (req_end - req_start) * 1000
            if first_token_time is not None:
                ttft_ms = (first_token_time - req_start) * 1000

            # Append assistant response to context
            if assistant_text:
                context.append({"role": "assistant", "content": assistant_text})

            turn_result = TurnResult(
                turn_index=turn_index,
                ttft_ms=ttft_ms,
                total_latency_ms=total_latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                context_tokens=context_tokens,
                error=error,
            )
            result.turns.append(turn_result)
            turn_index += 1

    result.total_turns = turn_index
    result.total_latency_ms = (time.monotonic() - conv_start) * 1000
    return result


def compute_multi_turn_stats(
    conversations: list[ConversationResult],
) -> MultiTurnResult:
    """Compute aggregate and per-turn statistics.

    Args:
        conversations: List of completed conversation results.

    Returns:
        MultiTurnResult with per-turn and aggregate statistics.
    """
    mt = MultiTurnResult(conversations=conversations)

    # Collect per-turn metrics
    turn_metrics: dict[int, list[TurnResult]] = {}
    all_ttft: list[float] = []
    all_latency: list[float] = []
    total_errors = 0
    total_turns_count = 0

    for conv in conversations:
        for turn in conv.turns:
            total_turns_count += 1
            if turn.error:
                total_errors += 1
                continue
            turn_metrics.setdefault(turn.turn_index, []).append(turn)
            all_ttft.append(turn.ttft_ms)
            all_latency.append(turn.total_latency_ms)

    # Per-turn stats
    for idx in sorted(turn_metrics.keys()):
        turns = turn_metrics[idx]
        ttfts = [t.ttft_ms for t in turns]
        lats = [t.total_latency_ms for t in turns]
        ctx_tokens = [t.context_tokens for t in turns]
        mt.per_turn_stats[idx] = {
            "count": len(turns),
            "mean_ttft_ms": round(sum(ttfts) / len(ttfts), 2) if ttfts else 0,
            "mean_latency_ms": round(sum(lats) / len(lats), 2) if lats else 0,
            "mean_context_tokens": round(
                sum(ctx_tokens) / len(ctx_tokens), 2
            ) if ctx_tokens else 0,
        }

    # Aggregate stats
    mt.aggregate_stats = {
        "total_conversations": len(conversations),
        "total_turns": total_turns_count,
        "total_errors": total_errors,
        "mean_ttft_ms": round(sum(all_ttft) / len(all_ttft), 2)
        if all_ttft
        else 0,
        "mean_latency_ms": round(sum(all_latency) / len(all_latency), 2)
        if all_latency
        else 0,
    }

    # Sort per-turn latencies for percentiles
    if all_latency:
        all_latency.sort()
        n = len(all_latency)
        mt.aggregate_stats["p50_latency_ms"] = round(
            all_latency[int(n * 0.5)], 2
        )
        mt.aggregate_stats["p99_latency_ms"] = round(
            all_latency[min(int(n * 0.99), n - 1)], 2
        )

    return mt

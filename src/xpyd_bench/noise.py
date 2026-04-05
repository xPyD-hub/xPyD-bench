"""Noise injection & chaos testing (M60).

Provides client-side fault injection to measure server resilience and
benchmark behavior under adverse network conditions.

Injection modes:
- **delay**: artificial latency added before sending each request
- **error**: randomly abort requests client-side (simulates client crashes)
- **payload corruption**: randomly mangle request payloads before sending
"""

from __future__ import annotations

import asyncio
import copy
import random
from dataclasses import dataclass
from typing import Any


@dataclass
class NoiseConfig:
    """Configuration for noise injection."""

    inject_delay_ms: float = 0.0
    inject_error_rate: float = 0.0
    inject_payload_corruption: float = 0.0
    seed: int | None = None

    @property
    def enabled(self) -> bool:
        return (
            self.inject_delay_ms > 0
            or self.inject_error_rate > 0
            or self.inject_payload_corruption > 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "inject_delay_ms": self.inject_delay_ms,
            "inject_error_rate": self.inject_error_rate,
            "inject_payload_corruption": self.inject_payload_corruption,
        }


@dataclass
class NoiseStats:
    """Tracks noise injection statistics."""

    delays_injected: int = 0
    total_delay_ms: float = 0.0
    errors_injected: int = 0
    payloads_corrupted: int = 0
    total_requests: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "delays_injected": self.delays_injected,
            "total_delay_ms": round(self.total_delay_ms, 2),
            "errors_injected": self.errors_injected,
            "payloads_corrupted": self.payloads_corrupted,
            "total_requests": self.total_requests,
        }


class NoiseInjector:
    """Applies noise injection to benchmark requests.

    Parameters
    ----------
    config : NoiseConfig
        Noise injection parameters.
    """

    def __init__(self, config: NoiseConfig) -> None:
        self.config = config
        self.stats = NoiseStats()
        self._rng = random.Random(config.seed)

    async def maybe_delay(self) -> float:
        """Inject artificial delay. Returns actual delay in ms."""
        if self.config.inject_delay_ms <= 0:
            return 0.0
        delay_s = self.config.inject_delay_ms / 1000.0
        await asyncio.sleep(delay_s)
        actual_ms = self.config.inject_delay_ms
        self.stats.delays_injected += 1
        self.stats.total_delay_ms += actual_ms
        self.stats.total_requests += 1
        return actual_ms

    def should_inject_error(self) -> bool:
        """Decide whether to inject a client-side error for this request."""
        if self.config.inject_error_rate <= 0:
            return False
        return self._rng.random() < self.config.inject_error_rate

    def corrupt_payload(self, payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Maybe corrupt the payload. Returns (payload, was_corrupted)."""
        if self.config.inject_payload_corruption <= 0:
            return payload, False
        if self._rng.random() >= self.config.inject_payload_corruption:
            return payload, False

        corrupted = copy.deepcopy(payload)
        # Corruption strategy: mangle the prompt/messages field
        if "prompt" in corrupted:
            corrupted["prompt"] = _corrupt_string(corrupted["prompt"], self._rng)
        elif "messages" in corrupted and corrupted["messages"]:
            # Corrupt the last message content
            msgs = corrupted["messages"]
            if msgs and isinstance(msgs[-1], dict) and "content" in msgs[-1]:
                msgs[-1]["content"] = _corrupt_string(
                    msgs[-1]["content"], self._rng
                )
        elif "input" in corrupted:
            corrupted["input"] = _corrupt_string(str(corrupted["input"]), self._rng)

        self.stats.payloads_corrupted += 1
        return corrupted, True


def _corrupt_string(s: str, rng: random.Random) -> str:
    """Corrupt a string by shuffling, truncating, or inserting garbage."""
    if not s:
        return s
    strategy = rng.choice(["shuffle", "truncate", "garbage"])
    if strategy == "shuffle":
        chars = list(s)
        rng.shuffle(chars)
        return "".join(chars)
    elif strategy == "truncate":
        cut = max(1, len(s) // 4)
        return s[:cut]
    else:  # garbage
        garbage = "".join(
            rng.choice("!@#$%^&*(){}[]<>?/\\|~`") for _ in range(len(s) // 2)
        )
        pos = rng.randint(0, len(s))
        return s[:pos] + garbage + s[pos:]


def build_noise_config_from_args(args: Any) -> NoiseConfig:
    """Build NoiseConfig from CLI args namespace."""
    return NoiseConfig(
        inject_delay_ms=float(getattr(args, "inject_delay", 0) or 0),
        inject_error_rate=float(getattr(args, "inject_error_rate", 0) or 0),
        inject_payload_corruption=float(
            getattr(args, "inject_payload_corruption", 0) or 0
        ),
        seed=getattr(args, "seed", None),
    )

"""Built-in benchmark scenario presets.

Each preset defines a set of CLI-compatible defaults for common workload
patterns.  Users select a preset via ``--scenario <name>`` and can still
override individual flags on the command line.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ScenarioConfig:
    """A named benchmark scenario preset."""

    name: str
    description: str

    # Workload
    num_prompts: int = 1000
    input_len: int = 256
    output_len: int = 128
    request_rate: float = float("inf")
    max_concurrency: int | None = None
    burstiness: float = 1.0

    # Dataset
    dataset_name: str = "random"
    synthetic_input_len_dist: str = "fixed"
    synthetic_output_len_dist: str = "fixed"

    # Sampling
    temperature: float | None = None

    # Rate pattern (YAML-style dict, None = use request_rate)
    rate_pattern: dict[str, Any] | None = None

    def to_overrides(self) -> dict[str, Any]:
        """Return a dict of non-None fields suitable for applying to args."""
        d = asdict(self)
        d.pop("name")
        d.pop("description")
        return {k: v for k, v in d.items() if v is not None}


# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------

SHORT = ScenarioConfig(
    name="short",
    description="Short prompts, short outputs — simulates interactive chat workload",
    num_prompts=2000,
    input_len=32,
    output_len=64,
    request_rate=50.0,
    max_concurrency=64,
    temperature=0.8,
)

LONG_CONTEXT = ScenarioConfig(
    name="long_context",
    description="Long prompts, moderate outputs — simulates RAG / document QA workload",
    num_prompts=200,
    input_len=2048,
    output_len=256,
    request_rate=5.0,
    max_concurrency=16,
    temperature=0.0,
)

MIXED = ScenarioConfig(
    name="mixed",
    description="Realistic mix of short and long prompts with varied output lengths",
    num_prompts=1000,
    input_len=256,
    output_len=128,
    request_rate=20.0,
    max_concurrency=32,
    dataset_name="synthetic",
    synthetic_input_len_dist="zipf",
    synthetic_output_len_dist="uniform",
)

STRESS = ScenarioConfig(
    name="stress",
    description="High-concurrency burst test — designed to stress-test the server",
    num_prompts=5000,
    input_len=64,
    output_len=32,
    request_rate=float("inf"),
    max_concurrency=256,
    burstiness=1.0,
    temperature=0.0,
)

# Registry
SCENARIOS: dict[str, ScenarioConfig] = {
    s.name: s for s in [SHORT, LONG_CONTEXT, MIXED, STRESS]
}


def get_scenario(name: str) -> ScenarioConfig:
    """Look up a scenario by name. Raises ``KeyError`` if not found."""
    if name not in SCENARIOS:
        available = ", ".join(sorted(SCENARIOS.keys()))
        raise KeyError(f"Unknown scenario '{name}'. Available: {available}")
    return SCENARIOS[name]


def list_scenarios() -> list[ScenarioConfig]:
    """Return all built-in scenarios."""
    return list(SCENARIOS.values())

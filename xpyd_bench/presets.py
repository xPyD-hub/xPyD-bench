"""Benchmark presets library (M38).

Provides built-in and user-defined benchmark configuration presets.
User presets are loaded from ``~/.xpyd-bench/presets/*.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PresetConfig:
    """A named benchmark preset with full configuration."""

    name: str
    description: str
    source: str = "built-in"  # "built-in" or "user"

    # All fields map directly to CLI arguments / YAML config keys
    config: dict[str, Any] = field(default_factory=dict)

    def to_overrides(self) -> dict[str, Any]:
        """Return config dict for applying to args."""
        return dict(self.config)


# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------

THROUGHPUT_MAX = PresetConfig(
    name="throughput-max",
    description="Maximize request throughput — high concurrency, short prompts, no rate limit",
    config={
        "num_prompts": 10000,
        "input_len": 64,
        "output_len": 32,
        "request_rate": float("inf"),
        "max_concurrency": 512,
        "temperature": 0.0,
        "stream": False,
    },
)

LATENCY_OPTIMAL = PresetConfig(
    name="latency-optimal",
    description="Measure optimal latency — low concurrency, sequential requests",
    config={
        "num_prompts": 100,
        "input_len": 128,
        "output_len": 64,
        "request_rate": 1.0,
        "max_concurrency": 1,
        "temperature": 0.0,
        "stream": True,
    },
)

SOAK_TEST = PresetConfig(
    name="soak-test",
    description="Long-running stability test — moderate rate over many requests",
    config={
        "num_prompts": 50000,
        "input_len": 256,
        "output_len": 128,
        "request_rate": 10.0,
        "max_concurrency": 32,
        "dataset_name": "synthetic",
        "synthetic_input_len_dist": "uniform",
        "synthetic_output_len_dist": "uniform",
    },
)

COLD_START = PresetConfig(
    name="cold-start",
    description="Cold-start measurement — few requests with warmup to isolate first-request",
    config={
        "num_prompts": 10,
        "input_len": 128,
        "output_len": 64,
        "request_rate": 1.0,
        "max_concurrency": 1,
        "warmup": 0,
        "stream": True,
        "temperature": 0.0,
    },
)

_BUILTIN_PRESETS: dict[str, PresetConfig] = {
    p.name: p for p in [THROUGHPUT_MAX, LATENCY_OPTIMAL, SOAK_TEST, COLD_START]
}


def _user_presets_dir() -> Path:
    """Return the user presets directory."""
    return Path.home() / ".xpyd-bench" / "presets"


def load_user_presets(presets_dir: Path | None = None) -> dict[str, PresetConfig]:
    """Load user-defined presets from YAML files in the presets directory."""
    d = presets_dir or _user_presets_dir()
    presets: dict[str, PresetConfig] = {}
    if not d.is_dir():
        return presets
    for f in sorted(d.glob("*.yaml")):
        try:
            with open(f) as fh:
                data = yaml.safe_load(fh) or {}
            name = data.get("name", f.stem)
            desc = data.get("description", "")
            # Everything except name/description is config
            config = {k: v for k, v in data.items() if k not in ("name", "description")}
            presets[name] = PresetConfig(
                name=name,
                description=desc,
                source="user",
                config=config,
            )
        except Exception:
            # Skip malformed files silently
            continue
    return presets


def list_presets(presets_dir: Path | None = None) -> list[PresetConfig]:
    """Return all available presets (built-in + user). User presets override built-in."""
    merged = dict(_BUILTIN_PRESETS)
    merged.update(load_user_presets(presets_dir))
    return list(merged.values())


def get_preset(name: str, presets_dir: Path | None = None) -> PresetConfig:
    """Look up a preset by name. Raises ``KeyError`` if not found."""
    # Check user presets first (override built-in)
    user = load_user_presets(presets_dir)
    if name in user:
        return user[name]
    if name in _BUILTIN_PRESETS:
        return _BUILTIN_PRESETS[name]
    available = sorted(set(list(_BUILTIN_PRESETS.keys()) + list(user.keys())))
    raise KeyError(f"Unknown preset '{name}'. Available: {', '.join(available)}")


def format_preset_detail(preset: PresetConfig) -> str:
    """Format a preset for display."""
    lines = [
        f"Preset: {preset.name}",
        f"Source: {preset.source}",
        f"Description: {preset.description}",
        "",
        "Configuration:",
    ]
    for k, v in sorted(preset.config.items()):
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def format_preset_list(presets: list[PresetConfig]) -> str:
    """Format a list of presets for display."""
    lines = ["Available presets:", ""]
    for p in presets:
        tag = f" [{p.source}]" if p.source != "built-in" else ""
        lines.append(f"  {p.name:20s} {p.description}{tag}")
    return "\n".join(lines)


def presets_list_main(argv: list[str] | None = None) -> None:
    """Entry point for ``xpyd-bench presets list``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="xpyd-bench presets list",
        description="List available benchmark presets",
    )
    parser.add_argument(
        "--presets-dir",
        type=str,
        default=None,
        help="Path to user presets directory (default: ~/.xpyd-bench/presets/)",
    )
    args = parser.parse_args(argv)
    pdir = Path(args.presets_dir) if args.presets_dir else None
    presets = list_presets(pdir)
    print(format_preset_list(presets))


def presets_show_main(argv: list[str] | None = None) -> None:
    """Entry point for ``xpyd-bench presets show <name>``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="xpyd-bench presets show",
        description="Show details of a benchmark preset",
    )
    parser.add_argument("name", help="Preset name")
    parser.add_argument(
        "--presets-dir",
        type=str,
        default=None,
        help="Path to user presets directory (default: ~/.xpyd-bench/presets/)",
    )
    args = parser.parse_args(argv)
    pdir = Path(args.presets_dir) if args.presets_dir else None
    try:
        preset = get_preset(args.name, pdir)
    except KeyError as e:
        print(str(e))
        raise SystemExit(1) from e
    print(format_preset_detail(preset))

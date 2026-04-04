"""M39: Cost Estimation — compute estimated costs based on token usage and pricing models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from xpyd_bench.bench.models import BenchmarkResult


@dataclass
class CostModel:
    """Pricing model mapping model names to per-1K-token costs."""

    # model_name -> {"input": float, "output": float} ($/1K tokens)
    models: dict[str, dict[str, float]] = field(default_factory=dict)
    # Optional default pricing when model name not found
    default: dict[str, float] | None = None


@dataclass
class CostEstimate:
    """Computed cost estimate for a benchmark run."""

    model: str
    input_tokens: int
    output_tokens: int
    input_cost_per_1k: float
    output_cost_per_1k: float
    input_cost: float
    output_cost: float
    total_cost: float
    currency: str = "USD"
    matched: bool = True  # False if fell back to default pricing


def load_cost_model(path: str | Path) -> CostModel:
    """Load a cost model from a YAML file.

    Expected format::

        currency: USD  # optional, default USD
        default:       # optional fallback
          input: 0.01
          output: 0.03
        models:
          gpt-4:
            input: 0.03
            output: 0.06
          gpt-3.5-turbo:
            input: 0.0005
            output: 0.0015
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Cost model file not found: {p}")

    data = yaml.safe_load(p.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Cost model must be a YAML mapping, got {type(data).__name__}")

    models_raw = data.get("models", {})
    if not isinstance(models_raw, dict):
        raise ValueError("Cost model 'models' must be a mapping")

    models: dict[str, dict[str, float]] = {}
    for name, pricing in models_raw.items():
        if not isinstance(pricing, dict):
            raise ValueError(f"Pricing for model '{name}' must be a mapping with input/output")
        models[str(name)] = {
            "input": float(pricing.get("input", 0)),
            "output": float(pricing.get("output", 0)),
        }

    default_raw = data.get("default")
    default = None
    if isinstance(default_raw, dict):
        default = {
            "input": float(default_raw.get("input", 0)),
            "output": float(default_raw.get("output", 0)),
        }

    cm = CostModel(models=models, default=default)
    # Attach currency for downstream use
    cm._currency = data.get("currency", "USD")  # type: ignore[attr-defined]
    return cm


def estimate_cost(
    result: BenchmarkResult,
    cost_model: CostModel,
    model_override: str | None = None,
) -> CostEstimate:
    """Compute cost estimate from benchmark result and cost model."""
    model_name = model_override or result.model or ""
    pricing = cost_model.models.get(model_name)
    matched = True
    if pricing is None:
        pricing = cost_model.default
        matched = False
    if pricing is None:
        raise ValueError(
            f"No pricing found for model '{model_name}' and no default pricing configured"
        )

    input_per_1k = pricing["input"]
    output_per_1k = pricing["output"]
    input_cost = result.total_input_tokens / 1000.0 * input_per_1k
    output_cost = result.total_output_tokens / 1000.0 * output_per_1k

    currency = getattr(cost_model, "_currency", "USD")

    return CostEstimate(
        model=model_name,
        input_tokens=result.total_input_tokens,
        output_tokens=result.total_output_tokens,
        input_cost_per_1k=input_per_1k,
        output_cost_per_1k=output_per_1k,
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=input_cost + output_cost,
        currency=currency,
        matched=matched,
    )


def estimate_cost_from_counts(
    input_tokens: int,
    output_tokens: int,
    cost_model: CostModel,
    model_name: str = "",
) -> CostEstimate:
    """Compute cost estimate from raw token counts (for dry-run)."""
    pricing = cost_model.models.get(model_name)
    matched = True
    if pricing is None:
        pricing = cost_model.default
        matched = False
    if pricing is None:
        raise ValueError(
            f"No pricing found for model '{model_name}' and no default pricing configured"
        )

    input_per_1k = pricing["input"]
    output_per_1k = pricing["output"]
    input_cost = input_tokens / 1000.0 * input_per_1k
    output_cost = output_tokens / 1000.0 * output_per_1k

    currency = getattr(cost_model, "_currency", "USD")

    return CostEstimate(
        model=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost_per_1k=input_per_1k,
        output_cost_per_1k=output_per_1k,
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=input_cost + output_cost,
        currency=currency,
        matched=matched,
    )


def format_cost_summary(est: CostEstimate) -> str:
    """Format a human-readable cost summary."""
    lines = [
        "Cost Estimation:",
        f"  Model:              {est.model or '(unknown)'}",
    ]
    if not est.matched:
        lines.append("  Pricing:            (default — model not found in cost model)")
    lines.extend([
        f"  Input tokens:       {est.input_tokens:,}",
        f"  Output tokens:      {est.output_tokens:,}",
        f"  Input rate:         ${est.input_cost_per_1k:.4f} / 1K tokens",
        f"  Output rate:        ${est.output_cost_per_1k:.4f} / 1K tokens",
        f"  Input cost:         ${est.input_cost:.6f}",
        f"  Output cost:        ${est.output_cost:.6f}",
        f"  Total cost:         ${est.total_cost:.6f} {est.currency}",
    ])
    return "\n".join(lines)


def cost_to_dict(est: CostEstimate) -> dict[str, Any]:
    """Serialize cost estimate to a JSON-compatible dict."""
    return {
        "model": est.model,
        "input_tokens": est.input_tokens,
        "output_tokens": est.output_tokens,
        "input_cost_per_1k": est.input_cost_per_1k,
        "output_cost_per_1k": est.output_cost_per_1k,
        "input_cost": est.input_cost,
        "output_cost": est.output_cost,
        "total_cost": est.total_cost,
        "currency": est.currency,
        "model_matched": est.matched,
    }

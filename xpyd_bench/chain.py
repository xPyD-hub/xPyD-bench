"""Request dependency chains (M59).

Define request sequences where output of request N feeds into request N+1.
Useful for benchmarking RAG pipelines and agent workflows.

Usage:
    xpyd-bench chain --chain chain.jsonl --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ChainStep:
    """A single step in a request dependency chain."""

    step_index: int
    prompt_template: str
    endpoint: str = "/v1/completions"
    model: str | None = None
    max_tokens: int = 128
    extract: dict[str, str] = field(default_factory=dict)
    # extract maps variable_name -> extraction expression
    # Supported formats:
    #   "jsonpath:choices.0.text"   – dot-notation JSONPath
    #   "regex:<pattern>"           – first regex group match on response body


@dataclass
class StepResult:
    """Metrics for a single chain step."""

    step_index: int
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    success: bool = True
    error: str | None = None
    extracted: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "step_index": self.step_index,
            "latency_ms": round(self.latency_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "success": self.success,
        }
        if self.error:
            d["error"] = self.error
        if self.extracted:
            d["extracted"] = self.extracted
        return d


@dataclass
class ChainResult:
    """Result for a full dependency chain execution."""

    chain_id: int
    steps: list[StepResult] = field(default_factory=list)
    total_latency_ms: float = 0.0
    total_steps: int = 0
    successful_steps: int = 0
    failed_step: int | None = None  # index of first failure, None if all ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "total_steps": self.total_steps,
            "successful_steps": self.successful_steps,
            "failed_step": self.failed_step,
            "steps": [s.to_dict() for s in self.steps],
        }


@dataclass
class ChainSummary:
    """Aggregated results across multiple chain executions."""

    chains: list[ChainResult] = field(default_factory=list)
    total_chains: int = 0
    completed_chains: int = 0
    mean_chain_latency_ms: float = 0.0
    per_step_stats: dict[int, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_chains": self.total_chains,
            "completed_chains": self.completed_chains,
            "mean_chain_latency_ms": round(self.mean_chain_latency_ms, 2),
            "per_step_stats": self.per_step_stats,
            "chains": [c.to_dict() for c in self.chains],
        }


# ---------------------------------------------------------------------------
# Chain loading
# ---------------------------------------------------------------------------


def load_chain(path: str) -> list[ChainStep]:
    """Load chain definition from a JSONL file.

    Each line is a JSON object with:
        - prompt_template (str): prompt with ``{{ var }}`` placeholders
        - endpoint (str, optional): API endpoint (default /v1/completions)
        - model (str, optional): model name
        - max_tokens (int, optional): max tokens (default 128)
        - extract (dict, optional): variable_name -> extraction expression

    Returns:
        Ordered list of ChainStep objects.
    """
    steps: list[ChainStep] = []
    p = Path(path)
    for idx, line in enumerate(p.read_text().strip().splitlines()):
        obj = json.loads(line)
        steps.append(
            ChainStep(
                step_index=idx,
                prompt_template=obj["prompt_template"],
                endpoint=obj.get("endpoint", "/v1/completions"),
                model=obj.get("model"),
                max_tokens=obj.get("max_tokens", 128),
                extract=obj.get("extract", {}),
            )
        )
    return steps


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _extract_jsonpath(data: Any, path: str) -> str:
    """Extract a value from a parsed JSON object using dot notation.

    Supports array indexing, e.g. ``choices.0.text``.
    """
    current = data
    for part in path.split("."):
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return ""
        elif isinstance(current, dict):
            current = current.get(part, "")
        else:
            return ""
    return str(current) if current is not None else ""


def _extract_regex(text: str, pattern: str) -> str:
    """Extract the first capturing group from *text* using *pattern*."""
    m = re.search(pattern, text)
    if m and m.groups():
        return m.group(1)
    if m:
        return m.group(0)
    return ""


def extract_value(response_body: str, expression: str) -> str:
    """Extract a value from a response body using an extraction expression.

    Supported formats:
        ``jsonpath:<dot.notation.path>``
        ``regex:<pattern>``

    Returns:
        Extracted string value, or empty string on failure.
    """
    if expression.startswith("jsonpath:"):
        jpath = expression[len("jsonpath:"):]
        try:
            data = json.loads(response_body)
        except json.JSONDecodeError:
            return ""
        return _extract_jsonpath(data, jpath)
    elif expression.startswith("regex:"):
        pattern = expression[len("regex:"):]
        return _extract_regex(response_body, pattern)
    else:
        # Default: treat as jsonpath
        try:
            data = json.loads(response_body)
        except json.JSONDecodeError:
            return ""
        return _extract_jsonpath(data, expression)


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


def render_template(template: str, variables: dict[str, str]) -> str:
    """Render ``{{ var }}`` placeholders in *template* with *variables*."""
    result = template
    for key, value in variables.items():
        result = result.replace("{{ " + key + " }}", value)
        result = result.replace("{{" + key + "}}", value)
    return result


# ---------------------------------------------------------------------------
# Rough token estimation
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """Rough token estimation."""
    return max(1, int(len(text.split()) * 1.3))


# ---------------------------------------------------------------------------
# Chain runner
# ---------------------------------------------------------------------------


async def run_chain(
    client: httpx.AsyncClient,
    base_url: str,
    steps: list[ChainStep],
    chain_id: int = 0,
    default_model: str = "default",
    api_key: str | None = None,
    timeout: float = 300.0,
    custom_headers: dict[str, str] | None = None,
) -> ChainResult:
    """Execute a dependency chain sequentially.

    Each step's response is parsed for extraction rules, and extracted
    variables are injected into the next step's prompt template.

    Args:
        client: Async HTTP client.
        base_url: Server base URL.
        steps: Ordered chain steps.
        chain_id: Identifier for this chain execution.
        default_model: Model name when step doesn't specify one.
        api_key: Optional API key.
        timeout: Per-request timeout in seconds.
        custom_headers: Additional HTTP headers.

    Returns:
        ChainResult with per-step metrics and extracted variables.
    """
    result = ChainResult(chain_id=chain_id)
    variables: dict[str, str] = {}
    chain_start = time.monotonic()

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if custom_headers:
        headers.update(custom_headers)

    for step in steps:
        prompt = render_template(step.prompt_template, variables)
        model = step.model or default_model
        endpoint = step.endpoint
        url = f"{base_url.rstrip('/')}{endpoint}"

        # Build payload based on endpoint type
        if "/chat/" in endpoint:
            payload: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": step.max_tokens,
            }
        else:
            payload = {
                "model": model,
                "prompt": prompt,
                "max_tokens": step.max_tokens,
            }

        step_result = StepResult(
            step_index=step.step_index,
            prompt_tokens=_estimate_tokens(prompt),
        )

        req_start = time.monotonic()
        try:
            resp = await client.post(url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            body = resp.text
            req_end = time.monotonic()
            step_result.latency_ms = (req_end - req_start) * 1000

            # Parse response for token counts
            try:
                resp_json = resp.json()
                usage = resp_json.get("usage", {})
                step_result.completion_tokens = usage.get("completion_tokens", 0)
                step_result.prompt_tokens = usage.get("prompt_tokens", step_result.prompt_tokens)
            except (json.JSONDecodeError, AttributeError):
                pass

            # Extract variables for next step
            for var_name, expr in step.extract.items():
                extracted = extract_value(body, expr)
                variables[var_name] = extracted
                step_result.extracted[var_name] = extracted

        except Exception as e:
            req_end = time.monotonic()
            step_result.latency_ms = (req_end - req_start) * 1000
            step_result.success = False
            step_result.error = str(e)
            result.steps.append(step_result)
            result.failed_step = step.step_index
            break

        result.steps.append(step_result)

    result.total_steps = len(result.steps)
    result.successful_steps = sum(1 for s in result.steps if s.success)
    result.total_latency_ms = (time.monotonic() - chain_start) * 1000
    return result


async def run_chains(
    base_url: str,
    chains: list[list[ChainStep]],
    default_model: str = "default",
    api_key: str | None = None,
    timeout: float = 300.0,
    max_concurrency: int = 1,
    custom_headers: dict[str, str] | None = None,
) -> ChainSummary:
    """Execute multiple chains with optional concurrency.

    Args:
        base_url: Server base URL.
        chains: List of chain definitions (each a list of steps).
        default_model: Default model name.
        api_key: Optional API key.
        timeout: Per-request timeout.
        max_concurrency: Max concurrent chain executions.
        custom_headers: Additional HTTP headers.

    Returns:
        ChainSummary with aggregate statistics.
    """
    sem = asyncio.Semaphore(max_concurrency)
    results: list[ChainResult] = []

    async def _run_one(chain_id: int, steps: list[ChainStep]) -> ChainResult:
        async with sem:
            async with httpx.AsyncClient() as client:
                return await run_chain(
                    client=client,
                    base_url=base_url,
                    steps=steps,
                    chain_id=chain_id,
                    default_model=default_model,
                    api_key=api_key,
                    timeout=timeout,
                    custom_headers=custom_headers,
                )

    tasks = [_run_one(i, steps) for i, steps in enumerate(chains)]
    results = await asyncio.gather(*tasks)

    return compute_chain_summary(list(results))


def compute_chain_summary(chains: list[ChainResult]) -> ChainSummary:
    """Compute aggregate statistics across chain executions."""
    summary = ChainSummary(chains=chains, total_chains=len(chains))

    completed = [c for c in chains if c.failed_step is None]
    summary.completed_chains = len(completed)

    latencies = [c.total_latency_ms for c in chains]
    if latencies:
        summary.mean_chain_latency_ms = sum(latencies) / len(latencies)

    # Per-step stats
    step_latencies: dict[int, list[float]] = {}
    for chain in chains:
        for step in chain.steps:
            if step.success:
                step_latencies.setdefault(step.step_index, []).append(step.latency_ms)

    for idx in sorted(step_latencies.keys()):
        lats = step_latencies[idx]
        n = len(lats)
        sorted_lats = sorted(lats)
        summary.per_step_stats[idx] = {
            "count": n,
            "mean_latency_ms": round(sum(lats) / n, 2),
            "p50_latency_ms": round(sorted_lats[int(n * 0.5)], 2) if n else 0,
            "p99_latency_ms": round(sorted_lats[min(int(n * 0.99), n - 1)], 2) if n else 0,
        }

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def chain_main(argv: list[str] | None = None) -> None:
    """CLI entry point for ``xpyd-bench chain``."""
    parser = argparse.ArgumentParser(
        description="Run request dependency chains (M59).",
    )
    parser.add_argument(
        "--chain",
        type=str,
        required=True,
        help="Path to chain definition JSONL file.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Server base URL.",
    )
    parser.add_argument("--model", type=str, default=None, help="Model name.")
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat the chain (default: 1).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Max concurrent chain executions (default: 1).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file.",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        dest="headers",
        help="Custom HTTP header in 'Key: Value' format (repeatable).",
    )

    args = parser.parse_args(argv)

    # Load chain
    steps = load_chain(args.chain)

    # Parse custom headers
    custom_headers: dict[str, str] = {}
    for h in args.headers:
        if ":" in h:
            k, v = h.split(":", 1)
            custom_headers[k.strip()] = v.strip()

    # Resolve model
    model = args.model or "default"

    # Resolve API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")

    # Build chain list (repeat)
    chains = [steps] * args.repeat

    # Run
    summary = asyncio.run(
        run_chains(
            base_url=args.base_url,
            chains=chains,
            default_model=model,
            api_key=api_key,
            timeout=args.timeout,
            max_concurrency=args.concurrency,
            custom_headers=custom_headers or None,
        )
    )

    # Print results
    _print_summary(summary)

    # Save
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(summary.to_dict(), indent=2))
        print(f"\nResults saved to {output_path}")


def _print_summary(summary: ChainSummary) -> None:
    """Print chain benchmark summary to terminal."""
    print("=" * 60)
    print("Chain Benchmark Summary")
    print("=" * 60)
    print(f"Total chains:     {summary.total_chains}")
    print(f"Completed chains: {summary.completed_chains}")
    print(f"Mean chain latency: {summary.mean_chain_latency_ms:.2f} ms")
    print()

    if summary.per_step_stats:
        print("Per-step statistics:")
        print(f"  {'Step':>6}  {'Count':>6}  {'Mean (ms)':>10}  {'P50 (ms)':>10}  {'P99 (ms)':>10}")
        print(f"  {'----':>6}  {'-----':>6}  {'---------':>10}  {'--------':>10}  {'--------':>10}")
        for idx in sorted(summary.per_step_stats.keys()):
            stats = summary.per_step_stats[idx]
            print(
                f"  {idx:>6}  {stats['count']:>6}  "
                f"{stats['mean_latency_ms']:>10.2f}  "
                f"{stats['p50_latency_ms']:>10.2f}  "
                f"{stats['p99_latency_ms']:>10.2f}"
            )

    # Per-chain details
    print()
    for chain in summary.chains:
        status = "✓" if chain.failed_step is None else f"✗ (failed at step {chain.failed_step})"
        print(
            f"Chain {chain.chain_id}: {status}  "
            f"steps={chain.successful_steps}/{chain.total_steps}  "
            f"latency={chain.total_latency_ms:.2f}ms"
        )
        for step in chain.steps:
            extracted_str = ""
            if step.extracted:
                pairs = [
                    f"{k}={v[:30]}..." if len(v) > 30 else f"{k}={v}"
                    for k, v in step.extracted.items()
                ]
                extracted_str = f"  extracted: {', '.join(pairs)}"
            err_str = f"  error: {step.error}" if step.error else ""
            print(
                f"  Step {step.step_index}: {step.latency_ms:.2f}ms"
                f"{extracted_str}{err_str}"
            )

    print("=" * 60)

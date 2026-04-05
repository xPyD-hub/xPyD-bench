"""Endpoint health check utility (M48).

Verify an OpenAI-compatible endpoint is alive and responding before
running a full benchmark.

Usage:
    xpyd-bench healthcheck --base-url http://localhost:8000

Checks performed:
1. TCP connectivity to the target host/port
2. GET /v1/models — list available models
3. POST /v1/completions — single short request sanity check
4. POST /v1/chat/completions — single short request sanity check
5. Report supported endpoints, latency, and model list
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class EndpointCheck:
    """Result of checking a single endpoint."""

    path: str
    available: bool = False
    latency_ms: float | None = None
    status_code: int | None = None
    error: str | None = None
    detail: dict | None = None


@dataclass
class HealthCheckResult:
    """Overall health check result for a base URL."""

    base_url: str
    reachable: bool = False
    connect_latency_ms: float | None = None
    models: list[str] = field(default_factory=list)
    endpoints: list[EndpointCheck] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "base_url": self.base_url,
            "reachable": self.reachable,
            "connect_latency_ms": self.connect_latency_ms,
            "models": self.models,
            "endpoints": [
                {
                    "path": e.path,
                    "available": e.available,
                    "latency_ms": e.latency_ms,
                    "status_code": e.status_code,
                    "error": e.error,
                }
                for e in self.endpoints
            ],
            "errors": self.errors,
        }

    @property
    def healthy(self) -> bool:
        """True when the endpoint is reachable and at least one endpoint works."""
        return self.reachable and any(e.available for e in self.endpoints)


async def _check_models(
    client: httpx.AsyncClient,
    base_url: str,
    headers: dict[str, str],
) -> EndpointCheck:
    """GET /v1/models."""
    path = "/v1/models"
    url = base_url.rstrip("/") + path
    t0 = time.perf_counter()
    try:
        resp = await client.get(url, headers=headers, timeout=10.0)
        latency = (time.perf_counter() - t0) * 1000
        check = EndpointCheck(
            path=path,
            status_code=resp.status_code,
            latency_ms=round(latency, 2),
        )
        if resp.status_code == 200:
            check.available = True
            try:
                body = resp.json()
                models = [m.get("id", "") for m in body.get("data", [])]
                check.detail = {"models": models}
            except Exception:
                check.detail = {"raw": resp.text[:500]}
        else:
            check.error = f"HTTP {resp.status_code}"
        return check
    except Exception as exc:
        return EndpointCheck(path=path, error=str(exc))


async def _check_completions(
    client: httpx.AsyncClient,
    base_url: str,
    model: str | None,
    headers: dict[str, str],
) -> EndpointCheck:
    """POST /v1/completions with a tiny prompt."""
    path = "/v1/completions"
    url = base_url.rstrip("/") + path
    payload = {
        "model": model or "default",
        "prompt": "Hello",
        "max_tokens": 1,
    }
    t0 = time.perf_counter()
    try:
        resp = await client.post(url, json=payload, headers=headers, timeout=30.0)
        latency = (time.perf_counter() - t0) * 1000
        check = EndpointCheck(
            path=path,
            status_code=resp.status_code,
            latency_ms=round(latency, 2),
        )
        if resp.status_code == 200:
            check.available = True
        else:
            check.error = f"HTTP {resp.status_code}"
        return check
    except Exception as exc:
        return EndpointCheck(path=path, error=str(exc))


async def _check_chat(
    client: httpx.AsyncClient,
    base_url: str,
    model: str | None,
    headers: dict[str, str],
) -> EndpointCheck:
    """POST /v1/chat/completions with a tiny message."""
    path = "/v1/chat/completions"
    url = base_url.rstrip("/") + path
    payload = {
        "model": model or "default",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1,
    }
    t0 = time.perf_counter()
    try:
        resp = await client.post(url, json=payload, headers=headers, timeout=30.0)
        latency = (time.perf_counter() - t0) * 1000
        check = EndpointCheck(
            path=path,
            status_code=resp.status_code,
            latency_ms=round(latency, 2),
        )
        if resp.status_code == 200:
            check.available = True
        else:
            check.error = f"HTTP {resp.status_code}"
        return check
    except Exception as exc:
        return EndpointCheck(path=path, error=str(exc))


async def _check_embeddings(
    client: httpx.AsyncClient,
    base_url: str,
    model: str | None,
    headers: dict[str, str],
) -> EndpointCheck:
    """POST /v1/embeddings with a tiny input."""
    path = "/v1/embeddings"
    url = base_url.rstrip("/") + path
    payload = {
        "model": model or "default",
        "input": "test",
    }
    t0 = time.perf_counter()
    try:
        resp = await client.post(url, json=payload, headers=headers, timeout=30.0)
        latency = (time.perf_counter() - t0) * 1000
        check = EndpointCheck(
            path=path,
            status_code=resp.status_code,
            latency_ms=round(latency, 2),
        )
        if resp.status_code == 200:
            check.available = True
        else:
            check.error = f"HTTP {resp.status_code}"
        return check
    except Exception as exc:
        return EndpointCheck(path=path, error=str(exc))


async def run_healthcheck(
    base_url: str,
    model: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> HealthCheckResult:
    """Run all health checks against *base_url* and return results."""
    result = HealthCheckResult(base_url=base_url)
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient() as client:
        # Step 1: connectivity check
        t0 = time.perf_counter()
        try:
            await client.get(
                base_url.rstrip("/") + "/v1/models",
                headers=headers,
                timeout=timeout,
            )
            result.connect_latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            result.reachable = True
        except Exception as exc:
            result.connect_latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            result.errors.append(f"Connection failed: {exc}")
            return result

        # Step 2: /v1/models
        models_check = await _check_models(client, base_url, headers)
        result.endpoints.append(models_check)
        if models_check.detail and "models" in models_check.detail:
            result.models = models_check.detail["models"]

        # Auto-select first model if not specified
        effective_model = model
        if not effective_model and result.models:
            effective_model = result.models[0]

        # Step 3: /v1/completions
        comp_check = await _check_completions(client, base_url, effective_model, headers)
        result.endpoints.append(comp_check)

        # Step 4: /v1/chat/completions
        chat_check = await _check_chat(client, base_url, effective_model, headers)
        result.endpoints.append(chat_check)

        # Step 5: /v1/embeddings
        emb_check = await _check_embeddings(client, base_url, effective_model, headers)
        result.endpoints.append(emb_check)

    return result


def print_healthcheck(result: HealthCheckResult) -> None:
    """Pretty-print health check results to stdout."""
    print(f"\n{'='*60}")
    print("  Endpoint Health Check")
    print(f"{'='*60}")
    print(f"  Base URL:   {result.base_url}")
    status = "✓ Reachable" if result.reachable else "✗ Unreachable"
    print(f"  Status:     {status}")
    if result.connect_latency_ms is not None:
        print(f"  Connect:    {result.connect_latency_ms:.1f} ms")
    if result.models:
        print(f"  Models:     {', '.join(result.models)}")
    print()

    for ep in result.endpoints:
        icon = "✓" if ep.available else "✗"
        lat = f"  ({ep.latency_ms:.1f} ms)" if ep.latency_ms is not None else ""
        err = f"  [{ep.error}]" if ep.error else ""
        print(f"  {icon} {ep.path}{lat}{err}")

    if result.errors:
        print()
        for e in result.errors:
            print(f"  ⚠ {e}")
    print(f"\n{'='*60}")
    overall = "HEALTHY" if result.healthy else "UNHEALTHY"
    print(f"  Result: {overall}")
    print(f"{'='*60}\n")


def healthcheck_main(argv: list[str] | None = None) -> None:
    """CLI entry point for ``xpyd-bench healthcheck``."""
    parser = argparse.ArgumentParser(
        prog="xpyd-bench healthcheck",
        description="Check if an OpenAI-compatible endpoint is alive and responsive",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        required=True,
        help="Base URL of the endpoint to check (e.g. http://localhost:8000)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for test requests (auto-detected from /v1/models if omitted)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Connection timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON instead of human-readable format",
    )
    args = parser.parse_args(argv)

    import os

    if args.api_key is None:
        args.api_key = os.environ.get("OPENAI_API_KEY")

    hc_result = asyncio.run(
        run_healthcheck(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            timeout=args.timeout,
        )
    )

    if args.json_output:
        print(json.dumps(hc_result.to_dict(), indent=2))
    else:
        print_healthcheck(hc_result)

    sys.exit(0 if hc_result.healthy else 1)

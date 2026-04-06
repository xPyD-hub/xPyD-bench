"""Endpoint capability discovery (M64).

Probes an LLM endpoint to auto-detect supported features and generates
recommended benchmark configurations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field

import httpx
import yaml


@dataclass
class EndpointCapabilities:
    """Discovered capabilities of an LLM endpoint."""

    base_url: str
    reachable: bool = False
    models: list[str] = field(default_factory=list)
    completions: bool = False
    chat_completions: bool = False
    embeddings: bool = False
    streaming: bool = False
    function_calling: bool = False
    batch: bool = False
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return dict representation."""
        return asdict(self)


def _probe_models(
    client: httpx.Client, base_url: str, caps: EndpointCapabilities
) -> None:
    """Probe /v1/models endpoint."""
    try:
        resp = client.get(f"{base_url}/v1/models")
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("data", [])
            caps.models = [m.get("id", "") for m in models if m.get("id")]
    except Exception as exc:
        caps.errors.append(f"models probe failed: {exc}")


def _probe_completions(
    client: httpx.Client, base_url: str, model: str, caps: EndpointCapabilities
) -> None:
    """Probe /v1/completions endpoint."""
    try:
        resp = client.post(
            f"{base_url}/v1/completions",
            json={"model": model, "prompt": "test", "max_tokens": 1},
        )
        if resp.status_code == 200:
            caps.completions = True
    except Exception as exc:
        caps.errors.append(f"completions probe failed: {exc}")


def _probe_chat(
    client: httpx.Client, base_url: str, model: str, caps: EndpointCapabilities
) -> None:
    """Probe /v1/chat/completions endpoint."""
    try:
        resp = client.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1,
            },
        )
        if resp.status_code == 200:
            caps.chat_completions = True
    except Exception as exc:
        caps.errors.append(f"chat completions probe failed: {exc}")


def _probe_streaming(
    client: httpx.Client, base_url: str, model: str, caps: EndpointCapabilities
) -> None:
    """Probe streaming support via /v1/chat/completions."""
    try:
        resp = client.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1,
                "stream": True,
            },
        )
        if resp.status_code == 200 and "text/event-stream" in resp.headers.get(
            "content-type", ""
        ):
            caps.streaming = True
    except Exception as exc:
        caps.errors.append(f"streaming probe failed: {exc}")


def _probe_embeddings(
    client: httpx.Client, base_url: str, model: str, caps: EndpointCapabilities
) -> None:
    """Probe /v1/embeddings endpoint."""
    try:
        resp = client.post(
            f"{base_url}/v1/embeddings",
            json={"model": model, "input": "test"},
        )
        if resp.status_code == 200:
            caps.embeddings = True
    except Exception as exc:
        caps.errors.append(f"embeddings probe failed: {exc}")


def _probe_function_calling(
    client: httpx.Client, base_url: str, model: str, caps: EndpointCapabilities
) -> None:
    """Probe function calling / tool support."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "test_fn",
                "description": "test",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    try:
        resp = client.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1,
                "tools": tools,
            },
        )
        if resp.status_code == 200:
            caps.function_calling = True
    except Exception as exc:
        caps.errors.append(f"function calling probe failed: {exc}")


def _probe_batch(
    client: httpx.Client, base_url: str, caps: EndpointCapabilities
) -> None:
    """Probe /v1/batch endpoint availability."""
    try:
        # Just check if the endpoint responds (even with 400 = exists)
        resp = client.post(f"{base_url}/v1/batch", json={})
        if resp.status_code in (200, 400, 422):
            caps.batch = True
    except Exception as exc:
        caps.errors.append(f"batch probe failed: {exc}")


def discover_endpoint(
    base_url: str,
    api_key: str | None = None,
    timeout: float = 10.0,
) -> EndpointCapabilities:
    """Probe an endpoint and return discovered capabilities."""
    base_url = base_url.rstrip("/")
    caps = EndpointCapabilities(base_url=base_url)

    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        client = httpx.Client(
            headers=headers, timeout=timeout, follow_redirects=True
        )
    except Exception as exc:
        caps.errors.append(f"client creation failed: {exc}")
        return caps

    # Check reachability
    try:
        resp = client.get(f"{base_url}/v1/models")
        caps.reachable = resp.status_code < 500
    except Exception:
        caps.reachable = False
        client.close()
        return caps

    _probe_models(client, base_url, caps)

    model = caps.models[0] if caps.models else "default"

    _probe_completions(client, base_url, model, caps)
    _probe_chat(client, base_url, model, caps)
    _probe_streaming(client, base_url, model, caps)
    _probe_embeddings(client, base_url, model, caps)
    _probe_function_calling(client, base_url, model, caps)
    _probe_batch(client, base_url, caps)

    client.close()
    return caps


def generate_config(caps: EndpointCapabilities) -> dict:
    """Generate a recommended YAML config from discovered capabilities."""
    config: dict = {"base_url": caps.base_url}

    if caps.models:
        config["model"] = caps.models[0]

    if caps.chat_completions:
        config["endpoint"] = "/v1/chat/completions"
    elif caps.completions:
        config["endpoint"] = "/v1/completions"

    if caps.streaming:
        config["stream"] = True

    config["num_prompts"] = 100
    config["request_rate"] = 10.0

    return config


def format_summary(caps: EndpointCapabilities) -> str:
    """Format capabilities as a human-readable summary."""
    lines = [f"Endpoint Discovery: {caps.base_url}", ""]

    def _indicator(val: bool) -> str:
        return "✓" if val else "✗"

    lines.append(f"  Reachable:           {_indicator(caps.reachable)}")
    if caps.models:
        lines.append(f"  Models:              {', '.join(caps.models)}")
    else:
        lines.append("  Models:              (none detected)")
    lines.append(f"  /v1/completions:     {_indicator(caps.completions)}")
    lines.append(f"  /v1/chat/completions:{_indicator(caps.chat_completions)}")
    lines.append(f"  /v1/embeddings:      {_indicator(caps.embeddings)}")
    lines.append(f"  Streaming:           {_indicator(caps.streaming)}")
    lines.append(f"  Function calling:    {_indicator(caps.function_calling)}")
    lines.append(f"  /v1/batch:           {_indicator(caps.batch)}")

    if caps.errors:
        lines.append("")
        lines.append("  Probe errors:")
        for err in caps.errors:
            lines.append(f"    - {err}")

    return "\n".join(lines)


def discover_main(argv: list[str] | None = None) -> None:
    """CLI entry point for discover subcommand."""
    parser = argparse.ArgumentParser(
        description="Discover endpoint capabilities"
    )
    parser.add_argument(
        "--base-url", required=True, help="Base URL of the LLM endpoint"
    )
    parser.add_argument("--api-key", default=None, help="API key for authentication")
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="Probe timeout in seconds"
    )
    parser.add_argument(
        "--json", dest="json_output", action="store_true", help="JSON output"
    )
    parser.add_argument(
        "--generate-config", default=None, help="Write recommended YAML config to path"
    )

    args = parser.parse_args(argv)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")

    caps = discover_endpoint(args.base_url, api_key=api_key, timeout=args.timeout)

    if args.json_output:
        print(json.dumps(caps.to_dict(), indent=2))
    else:
        print(format_summary(caps))

    if args.generate_config:
        config = generate_config(caps)
        with open(args.generate_config, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"\nRecommended config written to {args.generate_config}")

    if not caps.reachable:
        sys.exit(1)

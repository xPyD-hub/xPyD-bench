"""CLI entry points for xpyd-bench and xpyd-plan."""

from __future__ import annotations

import argparse
import sys


def bench_main(argv: list[str] | None = None) -> None:
    """Entry point for `xpyd-bench` command."""
    parser = argparse.ArgumentParser(
        prog="xpyd-bench",
        description="Benchmark an xPyD proxy instance",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Base URL of the xPyD proxy (e.g. http://localhost:8080)",
    )
    parser.add_argument(
        "--endpoint",
        choices=["completion", "chat"],
        default="chat",
        help="API endpoint to benchmark (default: chat)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent in-flight requests (default: 1)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=100,
        help="Total number of requests to send (default: 100)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens per request (default: 256)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset file (JSON/JSONL) for prompts",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Named scenario from scenarios/ directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for results (JSON)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use in requests",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help="Use streaming responses (required for TTFT measurement)",
    )

    args = parser.parse_args(argv)

    # TODO: implement benchmark runner
    print(f"xpyd-bench v{_get_version()}")
    print(f"  Target:      {args.target}")
    print(f"  Endpoint:    /v1/{'chat/completions' if args.endpoint == 'chat' else 'completions'}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Requests:    {args.num_requests}")
    print(f"  Max tokens:  {args.max_tokens}")
    print(f"  Stream:      {args.stream}")
    print()
    print("Benchmark runner not yet implemented.")


def plan_main(argv: list[str] | None = None) -> None:
    """Entry point for `xpyd-plan` command."""
    parser = argparse.ArgumentParser(
        prog="xpyd-plan",
        description="Recommend optimal Prefill:Decode node ratio",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to benchmark results (JSON) from xpyd-bench",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset file for offline estimation (without running benchmark)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        required=True,
        help="Total number of available nodes/GPUs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for recommendation (JSON)",
    )

    args = parser.parse_args(argv)

    if not args.data and not args.dataset:
        parser.error("Either --data or --dataset is required")

    # TODO: implement planner
    print(f"xpyd-plan v{_get_version()}")
    print(f"  Data:    {args.data}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Budget:  {args.budget} nodes")
    print()
    print("Planner not yet implemented.")


def _get_version() -> str:
    from xpyd_bench import __version__

    return __version__

"""CLI entry point for distributed benchmark (M32)."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path


def distributed_main(argv: list[str] | None = None) -> None:
    """Entry point for ``xpyd-bench distributed`` subcommand."""
    parser = argparse.ArgumentParser(
        prog="xpyd-bench distributed",
        description="Run a distributed benchmark across multiple worker machines",
    )
    parser.add_argument(
        "--workers",
        type=str,
        required=True,
        help="Comma-separated list of worker URLs (e.g. worker1:8089,worker2:8089).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        required=True,
        help="Target LLM server base URL.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint path (default: /v1/completions).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model name for requests.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=100,
        help="Total number of prompts to distribute (default: 100).",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=256,
        help="Input prompt length in tokens (default: 256).",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="Max output tokens per request (default: 128).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=None,
        dest="stream",
        help="Enable streaming.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_false",
        dest="stream",
        help="Disable streaming.",
    )
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
        help="Per-request timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--task-timeout",
        type=float,
        default=600.0,
        help="Total timeout per worker task in seconds (default: 600).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to dataset file (.jsonl, .json, or .csv).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help="Save aggregated result JSON to this path.",
    )
    args = parser.parse_args(argv)

    import os

    if args.api_key is None:
        args.api_key = os.environ.get("OPENAI_API_KEY")

    # Parse worker URLs
    raw_workers = [w.strip() for w in args.workers.split(",") if w.strip()]
    worker_urls: list[str] = []
    for w in raw_workers:
        if not w.startswith("http://") and not w.startswith("https://"):
            w = f"http://{w}"
        worker_urls.append(w.rstrip("/"))

    if len(worker_urls) < 1:
        parser.error("--workers requires at least 1 worker URL")

    # Generate prompts
    prompts = _build_prompts(args)

    from xpyd_bench import __version__
    from xpyd_bench.distributed.coordinator import run_distributed

    print(f"xpyd-bench distributed v{__version__}")
    print(f"  Workers:     {', '.join(worker_urls)}")
    print(f"  Target:      {args.base_url}")
    print(f"  Endpoint:    {args.endpoint}")
    print(f"  Num prompts: {len(prompts)}")
    print()

    result = asyncio.run(
        run_distributed(
            worker_urls=worker_urls,
            prompts=prompts,
            base_url=args.base_url,
            endpoint=args.endpoint,
            model=args.model,
            output_len=args.output_len,
            stream=args.stream,
            api_key=args.api_key,
            timeout=args.timeout,
            task_timeout=args.task_timeout,
        )
    )

    # Print summary
    print("=" * 60)
    print("  Distributed Benchmark Results")
    print("=" * 60)
    print(f"  Workers:        {len(worker_urls)}")
    print(f"  Healthy:        {len(worker_urls) - len(result.failed_workers)}")
    print(f"  Failed workers: {len(result.failed_workers)}")
    if result.failed_workers:
        for fw in result.failed_workers:
            print(f"    ✗ {fw}")
    print()

    for wr in result.worker_results:
        status = "✓" if not wr.error else "✗"
        print(f"  {status} {wr.worker_url}: {wr.completed} completed, {wr.failed} failed")
        if wr.error:
            print(f"      Error: {wr.error}")

    if result.benchmark_result:
        br = result.benchmark_result
        print()
        print("  Aggregated Metrics:")
        print(f"    Total completed:    {br.completed}")
        print(f"    Total failed:       {br.failed}")
        print(f"    Duration:           {br.total_duration_s:.2f}s")
        print(f"    Request throughput: {br.request_throughput:.2f} req/s")
        print(f"    Output throughput:  {br.output_throughput:.2f} tok/s")
        if br.mean_ttft_ms is not None:
            print(f"    Mean TTFT:          {br.mean_ttft_ms:.2f}ms")
        if br.mean_e2el_ms is not None:
            print(f"    Mean E2E latency:   {br.mean_e2el_ms:.2f}ms")
        if br.p99_e2el_ms is not None:
            print(f"    P99 E2E latency:    {br.p99_e2el_ms:.2f}ms")

    print("=" * 60)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        print(f"\nResults saved to {out_path}")


def _build_prompts(args) -> list[dict]:
    """Build prompt list from dataset or synthetic generation."""
    if args.dataset_path:
        from xpyd_bench.datasets.loader import load_dataset

        return load_dataset(
            path=args.dataset_path,
            name="random",
            num_prompts=args.num_prompts,
            input_len=args.input_len,
            output_len=args.output_len,
            seed=args.seed,
        )

    # Generate simple random prompts
    import random

    rng = random.Random(args.seed)
    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "benchmark", "test", "data", "model", "request", "response",
    ]
    prompts = []
    for _ in range(args.num_prompts):
        tokens = args.input_len
        text = " ".join(rng.choice(words) for _ in range(tokens))
        prompts.append({
            "prompt": text,
            "prompt_len": tokens,
            "output_len": args.output_len,
        })
    return prompts

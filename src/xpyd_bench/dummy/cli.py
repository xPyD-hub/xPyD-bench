"""CLI entry point for the dummy server."""

from __future__ import annotations

import argparse


def dummy_main(argv: list[str] | None = None) -> None:
    """Entry point for ``xpyd-dummy`` command."""
    parser = argparse.ArgumentParser(
        prog="xpyd-dummy",
        description="Dummy OpenAI-compatible server for xpyd-bench validation",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Bind host (default: 127.0.0.1)."
    )
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    parser.add_argument(
        "--prefill-ms",
        type=float,
        default=50.0,
        help="Simulated prefill latency in ms (default: 50).",
    )
    parser.add_argument(
        "--decode-ms",
        type=float,
        default=10.0,
        help="Simulated per-token decode latency in ms (default: 10).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="dummy-model",
        help="Model name to report (default: dummy-model).",
    )
    parser.add_argument(
        "--max-tokens-default",
        type=int,
        default=128,
        help="Default max_tokens when not specified in request (default: 128).",
    )
    parser.add_argument(
        "--eos-min-ratio",
        type=float,
        default=0.5,
        help="Minimum fraction of max_tokens before EOS can fire (default: 0.5).",
    )
    parser.add_argument(
        "--require-api-key",
        type=str,
        default=None,
        help="Require this API key for authentication. Returns 401 for missing/wrong key.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=1536,
        help="Dimensionality of embedding vectors (default: 1536).",
    )
    parser.add_argument(
        "--max-rps",
        type=float,
        default=None,
        help="Maximum requests per second before rejecting with 429 (default: unlimited).",
    )
    parser.add_argument(
        "--ratelimit-rpm",
        type=int,
        default=None,
        help="Include rate-limit headers in responses with this RPM limit.",
    )
    parser.add_argument(
        "--speculative-draft-size",
        type=int,
        default=0,
        dest="speculative_draft_size",
        help="Emit x_spec data in SSE chunks with this draft batch size (0 disables).",
    )
    parser.add_argument(
        "--speculative-acceptance-rate",
        type=float,
        default=0.8,
        dest="speculative_acceptance_rate",
        help="Simulated draft token acceptance rate (default 0.8).",
    )
    args = parser.parse_args(argv)

    import uvicorn

    from xpyd_bench.dummy.server import ServerConfig, create_app

    config = ServerConfig(
        prefill_ms=args.prefill_ms,
        decode_ms=args.decode_ms,
        model_name=args.model_name,
        max_tokens_default=args.max_tokens_default,
        eos_min_ratio=args.eos_min_ratio,
        require_api_key=args.require_api_key,
        embedding_dim=args.embedding_dim,
        max_rps=args.max_rps,
        ratelimit_rpm=args.ratelimit_rpm,
        speculative_draft_size=args.speculative_draft_size,
        speculative_acceptance_rate=args.speculative_acceptance_rate,
    )
    app = create_app(config)

    print(f"xpyd-dummy starting on {args.host}:{args.port}")
    print(f"  Prefill: {args.prefill_ms}ms, Decode: {args.decode_ms}ms/token")
    print(f"  Model: {args.model_name}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

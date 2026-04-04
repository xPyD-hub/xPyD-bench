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
    args = parser.parse_args(argv)

    import uvicorn

    from xpyd_bench.dummy.server import ServerConfig, create_app

    config = ServerConfig(
        prefill_ms=args.prefill_ms,
        decode_ms=args.decode_ms,
        model_name=args.model_name,
        max_tokens_default=args.max_tokens_default,
    )
    app = create_app(config)

    print(f"xpyd-dummy starting on {args.host}:{args.port}")
    print(f"  Prefill: {args.prefill_ms}ms, Decode: {args.decode_ms}ms/token")
    print(f"  Model: {args.model_name}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

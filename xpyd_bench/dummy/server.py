"""Backward-compatible shim — delegates to xpyd-sim.

This module is DEPRECATED. Use xpyd_bench.sim_adapter or xpyd_sim directly.
Kept only so existing test imports don't break during migration.
"""

from __future__ import annotations

from xpyd_sim.common.helpers import count_prompt_tokens as _estimate_prompt_tokens  # noqa: F401
from xpyd_sim.common.tools import build_tool_calls as _build_tool_calls  # noqa: F401
from xpyd_sim.common.tools import generate_dummy_from_schema as _generate_dummy_args  # noqa: F401
from xpyd_sim.server import ServerConfig as _SimServerConfig
from xpyd_sim.server import _generate_response_content  # noqa: F401
from xpyd_sim.server import create_app as _sim_create_app  # noqa: F401


def _build_json_response(response_format: dict, max_tokens: int) -> str:
    """Thin wrapper around sim's _generate_response_content."""
    result = _generate_response_content(response_format, max_tokens)
    if result is not None:
        return result
    import json
    return json.dumps({"result": " ".join(["token"] * min(max_tokens, 5))})


class ServerConfig(_SimServerConfig):
    """Backward-compatible ServerConfig accepting old field names.

    Maps:
      prefill_ms -> prefill_delay_ms
      decode_ms  -> decode_delay_per_token_ms
    """

    def __init__(self, **kwargs):
        if "prefill_ms" in kwargs and "prefill_delay_ms" not in kwargs:
            kwargs["prefill_delay_ms"] = kwargs.pop("prefill_ms")
        elif "prefill_ms" in kwargs:
            kwargs.pop("prefill_ms")

        if "decode_ms" in kwargs and "decode_delay_per_token_ms" not in kwargs:
            kwargs["decode_delay_per_token_ms"] = kwargs.pop("decode_ms")
        elif "decode_ms" in kwargs:
            kwargs.pop("decode_ms")

        kwargs.pop("max_tokens_default", None)
        super().__init__(**kwargs)


def create_app(config=None):
    """Create app, accepting both old and new ServerConfig."""
    return _sim_create_app(config)

"""PD ratio planner — analyzes data to recommend Prefill:Decode ratios."""

from __future__ import annotations

# TODO: implement
# - load benchmark results (from xpyd-bench output JSON)
# - load dataset for offline estimation (prompt length distribution)
# - model prefill cost: f(input_tokens)
# - model decode cost: g(output_tokens)
# - given budget N nodes, recommend P:D split
# - output recommendation with reasoning

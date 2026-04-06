"""Webhook notification support (M61).

POST benchmark results to one or more webhook URLs after completion.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time


def compute_signature(payload: bytes, secret: str) -> str:
    """Compute HMAC-SHA256 signature for webhook payload."""
    return hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


def send_webhook(
    url: str,
    result: dict,
    secret: str | None = None,
    timeout: float = 30.0,
    max_retries: int = 3,
) -> dict:
    """POST result JSON to a webhook URL.

    Returns a dict with delivery status:
        {"url": ..., "status_code": ..., "success": ..., "attempts": ..., "error": ...}
    """
    import httpx

    payload = json.dumps(result, default=str).encode()
    headers = {"Content-Type": "application/json"}
    if secret:
        sig = compute_signature(payload, secret)
        headers["X-Webhook-Signature"] = f"sha256={sig}"

    last_error: str | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = httpx.post(url, content=payload, headers=headers, timeout=timeout)
            if resp.status_code < 400:
                return {
                    "url": url,
                    "status_code": resp.status_code,
                    "success": True,
                    "attempts": attempt,
                    "error": None,
                }
            last_error = f"HTTP {resp.status_code}"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)

        if attempt < max_retries:
            time.sleep(1.0 * attempt)  # simple backoff

    return {
        "url": url,
        "status_code": None,
        "success": False,
        "attempts": max_retries,
        "error": last_error,
    }


def send_webhooks(
    urls: list[str],
    result: dict,
    secret: str | None = None,
    timeout: float = 30.0,
    max_retries: int = 3,
) -> list[dict]:
    """Send webhook to multiple URLs. Returns list of delivery results."""
    return [
        send_webhook(url, result, secret=secret, timeout=timeout, max_retries=max_retries)
        for url in urls
    ]


def format_webhook_summary(deliveries: list[dict]) -> str:
    """Format webhook delivery results for terminal output."""
    lines = ["Webhook notifications:"]
    for d in deliveries:
        status = "✓" if d["success"] else "✗"
        detail = f"HTTP {d['status_code']}" if d["status_code"] else d["error"]
        lines.append(f"  {status} {d['url']} ({detail}, {d['attempts']} attempt(s))")
    return "\n".join(lines)

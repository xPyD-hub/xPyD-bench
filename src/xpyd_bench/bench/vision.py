"""Multimodal (vision) input utilities for benchmarking VLMs (M77)."""

from __future__ import annotations

import base64
import random
from pathlib import Path
from typing import Any

# Common MIME types by extension
_EXT_TO_MIME: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
}


def _guess_mime(path: str) -> str:
    """Guess MIME type from file extension."""
    ext = Path(path).suffix.lower()
    return _EXT_TO_MIME.get(ext, "image/png")


def encode_image_base64(path: str) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def build_vision_content(
    text: str,
    *,
    image_urls: list[str] | None = None,
    image_files: list[str] | None = None,
    image_detail: str = "auto",
) -> list[dict[str, Any]]:
    """Build a multimodal ``content`` array for an OpenAI chat message.

    Parameters
    ----------
    text:
        The text portion of the user message.
    image_urls:
        HTTP(S) URLs to include as ``image_url`` content parts.
    image_files:
        Local file paths to include as base64-encoded ``image_url`` parts.
    image_detail:
        Image detail level: ``"auto"``, ``"low"``, or ``"high"``.

    Returns
    -------
    list[dict]:
        A ``content`` array suitable for OpenAI chat completions.
    """
    parts: list[dict[str, Any]] = []

    # Add image parts first (URLs)
    for url in image_urls or []:
        parts.append({
            "type": "image_url",
            "image_url": {"url": url, "detail": image_detail},
        })

    # Add image parts (local files as base64 data URIs)
    for fpath in image_files or []:
        mime = _guess_mime(fpath)
        b64 = encode_image_base64(fpath)
        data_uri = f"data:{mime};base64,{b64}"
        parts.append({
            "type": "image_url",
            "image_url": {"url": data_uri, "detail": image_detail},
        })

    # Add text part
    parts.append({"type": "text", "text": text})

    return parts


def generate_synthetic_image(
    width: int = 64,
    height: int = 64,
    seed: int | None = None,
) -> bytes:
    """Generate a minimal synthetic PNG image (random pixels).

    Returns raw PNG bytes. Uses a simple uncompressed PNG to avoid
    numpy/PIL dependencies.
    """
    import struct
    import zlib

    rng = random.Random(seed)

    # Build raw pixel data (RGB)
    raw_rows = []
    for _ in range(height):
        row = b"\x00"  # filter byte: None
        row += bytes(rng.randint(0, 255) for _ in range(width * 3))
        raw_rows.append(row)
    raw_data = b"".join(raw_rows)

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    # PNG signature
    sig = b"\x89PNG\r\n\x1a\n"
    # IHDR
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = _chunk(b"IHDR", ihdr_data)
    # IDAT
    compressed = zlib.compress(raw_data)
    idat = _chunk(b"IDAT", compressed)
    # IEND
    iend = _chunk(b"IEND", b"")

    return sig + ihdr + idat + iend


def load_image_sources(
    image_dir: str | None = None,
    image_url: str | None = None,
    synthetic_images: int = 0,
    synthetic_width: int = 64,
    synthetic_height: int = 64,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Load or generate image sources for vision benchmarking.

    Returns a list of dicts, each with either ``"url"`` or ``"data_uri"`` key.
    """
    sources: list[dict[str, Any]] = []

    # Load from directory
    if image_dir:
        p = Path(image_dir)
        if not p.is_dir():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        exts = set(_EXT_TO_MIME.keys())
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in exts:
                mime = _guess_mime(str(f))
                b64 = encode_image_base64(str(f))
                sources.append({"data_uri": f"data:{mime};base64,{b64}"})

    # Single URL
    if image_url:
        sources.append({"url": image_url})

    # Synthetic images
    if synthetic_images > 0:
        for i in range(synthetic_images):
            img_bytes = generate_synthetic_image(
                width=synthetic_width,
                height=synthetic_height,
                seed=seed + i,
            )
            b64 = base64.b64encode(img_bytes).decode("ascii")
            sources.append({"data_uri": f"data:image/png;base64,{b64}"})

    return sources


def build_vision_payload_content(
    text: str,
    image_source: dict[str, Any],
    image_detail: str = "auto",
) -> list[dict[str, Any]]:
    """Build multimodal content array from a text prompt and one image source."""
    url = image_source.get("url") or image_source.get("data_uri", "")
    return [
        {
            "type": "image_url",
            "image_url": {"url": url, "detail": image_detail},
        },
        {"type": "text", "text": text},
    ]

"""Token-level streaming latency CDF analysis (M91).

Computes cumulative distribution function of inter-token latencies from
streaming responses, with fine-grained percentiles and bimodal distribution
detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TokenCDFResult:
    """CDF analysis result for inter-token latencies."""

    # CDF data points: list of (latency_ms, cumulative_fraction) tuples
    cdf_points: list[tuple[float, float]] = field(default_factory=list)
    # Fine-grained percentiles
    percentiles: dict[str, float] = field(default_factory=dict)
    # Total token count used for CDF
    total_tokens: int = 0
    # Bimodal detection
    is_bimodal: bool = False
    bimodal_details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cdf_points": [
                {"latency_ms": round(lat, 3), "cumulative_fraction": round(frac, 6)}
                for lat, frac in self.cdf_points
            ],
            "percentiles": {k: round(v, 3) for k, v in self.percentiles.items()},
            "total_tokens": self.total_tokens,
            "is_bimodal": self.is_bimodal,
            "bimodal_details": self.bimodal_details,
        }


_PERCENTILE_LEVELS = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]


def compute_token_cdf(
    itl_values: list[float],
    max_cdf_points: int = 200,
) -> TokenCDFResult:
    """Compute CDF of inter-token latencies.

    Args:
        itl_values: List of inter-token latency values in milliseconds.
        max_cdf_points: Maximum number of CDF data points to emit (downsampled).

    Returns:
        TokenCDFResult with CDF data, percentiles, and bimodal detection.
    """
    if not itl_values:
        return TokenCDFResult()

    arr = np.array(itl_values, dtype=np.float64)
    arr_sorted = np.sort(arr)
    n = len(arr_sorted)

    # Compute fine-grained percentiles
    percentiles: dict[str, float] = {}
    for p in _PERCENTILE_LEVELS:
        percentiles[f"p{p}"] = float(np.percentile(arr_sorted, p))

    # Build CDF points (downsample if too many)
    if n <= max_cdf_points:
        cdf_points = [
            (float(arr_sorted[i]), (i + 1) / n)
            for i in range(n)
        ]
    else:
        indices = np.linspace(0, n - 1, max_cdf_points, dtype=int)
        cdf_points = [
            (float(arr_sorted[idx]), (idx + 1) / n)
            for idx in indices
        ]

    # Bimodal detection using histogram-based dip test
    is_bimodal, bimodal_details = _detect_bimodal(arr)

    return TokenCDFResult(
        cdf_points=cdf_points,
        percentiles=percentiles,
        total_tokens=n,
        is_bimodal=is_bimodal,
        bimodal_details=bimodal_details,
    )


def _detect_bimodal(arr: np.ndarray) -> tuple[bool, dict[str, Any]]:
    """Detect bimodal distribution using histogram valley detection.

    Uses a simple approach: bin the data, smooth the histogram, look for
    a valley (local minimum) between two peaks (local maxima).

    Returns:
        (is_bimodal, details_dict)
    """
    if len(arr) < 20:
        return False, {"reason": "insufficient_data", "sample_count": len(arr)}

    # Use sqrt rule for bin count to better capture bimodal structure
    n_bins = max(20, int(np.sqrt(len(arr))))
    counts, bin_edges = np.histogram(arr, bins=n_bins)

    # Smooth with moving average (window=3)
    if len(counts) >= 3:
        smoothed = np.convolve(counts, np.ones(3) / 3, mode="same")
    else:
        smoothed = counts.astype(float)

    # Find peaks (local maxima) and valleys (local minima)
    peaks = []
    valleys = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
            peaks.append(i)
        elif smoothed[i] < smoothed[i - 1] and smoothed[i] < smoothed[i + 1]:
            valleys.append(i)
        elif (
            smoothed[i] <= smoothed[i - 1]
            and smoothed[i] <= smoothed[i + 1]
            and smoothed[i] == 0
            and (smoothed[i - 1] > 0 or smoothed[i + 1] > 0)
        ):
            # Zero-gap valley: flat zero region between non-zero bins
            valleys.append(i)

    if len(peaks) >= 2 and len(valleys) >= 1:
        # Check if valley is deep enough (at least 50% drop from lower peak)
        peak_heights = [smoothed[p] for p in peaks]
        peak_heights_sorted = sorted(peak_heights, reverse=True)
        min_peak = peak_heights_sorted[1]  # Second highest peak

        # Find deepest valley between the two highest peaks
        top_two_peaks = sorted(
            [peaks[i] for i in np.argsort([-smoothed[p] for p in peaks])[:2]]
        )
        valleys_between = [v for v in valleys if top_two_peaks[0] < v < top_two_peaks[1]]

        if valleys_between:
            deepest_valley = min(valleys_between, key=lambda v: smoothed[v])
            valley_depth = smoothed[deepest_valley]
            drop_ratio = 1.0 - (valley_depth / min_peak) if min_peak > 0 else 0.0

            if drop_ratio >= 0.3:  # Valley is at least 30% below the smaller peak
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                return True, {
                    "peak_count": len(peaks),
                    "valley_drop_ratio": round(float(drop_ratio), 3),
                    "mode1_ms": round(float(bin_centers[top_two_peaks[0]]), 3),
                    "mode2_ms": round(float(bin_centers[top_two_peaks[1]]), 3),
                    "valley_ms": round(float(bin_centers[deepest_valley]), 3),
                }

    return False, {"peak_count": len(peaks)}


def collect_itl_from_requests(requests: list) -> list[float]:
    """Collect all inter-token latency values from request results.

    Args:
        requests: List of RequestResult objects.

    Returns:
        Flat list of inter-token latency values in milliseconds.
    """
    itl_values: list[float] = []
    for req in requests:
        if hasattr(req, "itl_ms") and req.itl_ms:
            itl_values.extend(req.itl_ms)
    return itl_values


def generate_cdf_html_chart(cdf_result: TokenCDFResult) -> str:
    """Generate an HTML/JS snippet for an interactive CDF chart.

    Uses inline SVG with embedded data for self-contained HTML.

    Args:
        cdf_result: TokenCDFResult with CDF data.

    Returns:
        HTML string containing the CDF chart section.
    """
    if not cdf_result.cdf_points:
        return ""

    import json

    chart_data = json.dumps(
        [{"x": p[0], "y": p[1]} for p in cdf_result.cdf_points]
    )

    percentile_rows = ""
    for key, val in sorted(cdf_result.percentiles.items(), key=lambda kv: float(kv[0][1:])):
        percentile_rows += f"<tr><td>{key}</td><td>{val:.3f} ms</td></tr>\n"

    bimodal_info = ""
    if cdf_result.is_bimodal:
        d = cdf_result.bimodal_details
        bimodal_info = f"""
<div style="background:#fff3cd;padding:10px;border-radius:4px;margin:10px 0;">
  <strong>⚠ Bimodal Distribution Detected</strong><br>
  Mode 1: {d.get('mode1_ms', 'N/A'):.1f} ms | Mode 2: {d.get('mode2_ms', 'N/A'):.1f} ms |
  Valley: {d.get('valley_ms', 'N/A'):.1f} ms (drop ratio: {d.get('valley_drop_ratio', 0):.1%})
</div>"""

    return f"""
<h2>Token-Level Streaming Latency CDF</h2>
{bimodal_info}
<table>
<thead><tr><th>Percentile</th><th>Latency</th></tr></thead>
<tbody>
{percentile_rows}
</tbody>
</table>
<div id="token-cdf-chart" style="width:100%;height:400px;position:relative;margin:20px 0;">
  <canvas id="tokenCdfCanvas" width="800" height="400"></canvas>
</div>
<script>
(function() {{
  var data = {chart_data};
  var canvas = document.getElementById('tokenCdfCanvas');
  if (!canvas) return;
  var ctx = canvas.getContext('2d');
  var W = canvas.width, H = canvas.height;
  var pad = {{l:60, r:20, t:20, b:40}};
  var pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
  if (data.length === 0) return;
  var maxX = data[data.length-1].x * 1.05;
  ctx.strokeStyle = '#2196F3'; ctx.lineWidth = 2;
  ctx.beginPath();
  for (var i = 0; i < data.length; i++) {{
    var px = pad.l + (data[i].x / maxX) * pw;
    var py = pad.t + (1 - data[i].y) * ph;
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }}
  ctx.stroke();
  // Axes
  ctx.strokeStyle = '#333'; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, H-pad.b);
  ctx.lineTo(W-pad.r, H-pad.b); ctx.stroke();
  ctx.fillStyle = '#333'; ctx.font = '12px sans-serif';
  ctx.fillText('Latency (ms)', W/2-30, H-5);
  ctx.save(); ctx.rotate(-Math.PI/2);
  ctx.fillText('Cumulative Fraction', -H/2-40, 15);
  ctx.restore();
  // Y ticks
  for (var y = 0; y <= 1; y += 0.25) {{
    var yy = pad.t + (1-y)*ph;
    ctx.fillText(y.toFixed(2), 5, yy+4);
    ctx.strokeStyle='#eee'; ctx.beginPath();
    ctx.moveTo(pad.l,yy); ctx.lineTo(W-pad.r,yy); ctx.stroke();
  }}
  // X ticks
  for (var xi = 0; xi <= 4; xi++) {{
    var xv = maxX * xi / 4;
    var xx = pad.l + (xv/maxX)*pw;
    ctx.fillStyle='#333'; ctx.fillText(xv.toFixed(1), xx-15, H-pad.b+15);
  }}
}})();
</script>
"""

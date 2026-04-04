"""Request latency heatmap visualization.

Time-bucketed latency heatmap for terminal (rich) and HTML report.
X-axis: benchmark elapsed time, Y-axis: latency buckets.
Color intensity: request count per bucket.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from xpyd_bench.bench.models import BenchmarkResult


@dataclass
class HeatmapData:
    """Pre-computed heatmap grid data."""

    # 2D grid: grid[time_bucket][latency_bucket] = count
    grid: list[list[int]] = field(default_factory=list)
    time_edges: list[float] = field(default_factory=list)  # seconds
    latency_edges: list[float] = field(default_factory=list)  # ms
    max_count: int = 0
    total_requests: int = 0


def compute_heatmap(
    result: BenchmarkResult,
    *,
    time_bins: int = 20,
    latency_bins: int = 15,
) -> HeatmapData:
    """Build a time × latency heatmap from benchmark results.

    Parameters
    ----------
    result:
        Completed benchmark result with per-request data.
    time_bins:
        Number of time buckets along the X-axis.
    latency_bins:
        Number of latency buckets along the Y-axis.

    Returns
    -------
    HeatmapData with the computed grid and edge arrays.
    """
    successful = [
        r for r in result.requests
        if r.success and r.start_time is not None
    ]
    if not successful:
        return HeatmapData()

    bench_start = result.bench_start_time
    elapsed = [r.start_time - bench_start for r in successful]
    latencies = [r.latency_ms for r in successful]

    t_min, t_max = min(elapsed), max(elapsed)
    l_min, l_max = min(latencies), max(latencies)

    # Guard against degenerate ranges
    if t_max <= t_min:
        t_max = t_min + 1.0
    if l_max <= l_min:
        l_max = l_min + 1.0

    # Compute edges
    time_step = (t_max - t_min) / time_bins
    lat_step = (l_max - l_min) / latency_bins

    time_edges = [t_min + i * time_step for i in range(time_bins + 1)]
    latency_edges = [l_min + i * lat_step for i in range(latency_bins + 1)]

    # Build grid
    grid = [[0] * latency_bins for _ in range(time_bins)]

    for t, lat in zip(elapsed, latencies):
        ti = min(int((t - t_min) / time_step), time_bins - 1)
        li = min(int((lat - l_min) / lat_step), latency_bins - 1)
        grid[ti][li] += 1

    max_count = max(max(row) for row in grid) if grid else 0

    return HeatmapData(
        grid=grid,
        time_edges=time_edges,
        latency_edges=latency_edges,
        max_count=max_count,
        total_requests=len(successful),
    )


# ── Terminal rendering (rich) ──────────────────────────────────────────

_HEAT_CHARS = " ░▒▓█"


def render_terminal_heatmap(data: HeatmapData) -> str:
    """Render a heatmap as a terminal-friendly string using block chars.

    Returns a multi-line string suitable for printing to the terminal.
    """
    if not data.grid or data.max_count == 0:
        return "  (no data for heatmap)\n"

    lines: list[str] = []
    lines.append("Latency Heatmap (time → , latency ↑)")
    lines.append("")

    num_lat = len(data.latency_edges) - 1
    num_time = len(data.time_edges) - 1

    # Render rows from top (highest latency) to bottom (lowest)
    for li in range(num_lat - 1, -1, -1):
        label = f"{data.latency_edges[li + 1]:7.0f}ms │"
        row_chars = []
        for ti in range(num_time):
            count = data.grid[ti][li]
            intensity = int(count / data.max_count * (len(_HEAT_CHARS) - 1))
            row_chars.append(_HEAT_CHARS[intensity])
        lines.append(label + "".join(row_chars))

    # X-axis
    sep = " " * 10 + "└" + "─" * num_time
    lines.append(sep)

    # Time labels (start and end)
    t_start = f"{data.time_edges[0]:.0f}s"
    t_end = f"{data.time_edges[-1]:.0f}s"
    pad = num_time - len(t_end)
    lines.append(" " * 11 + t_start + " " * max(1, pad - len(t_start)) + t_end)

    # Legend
    lines.append("")
    legend_parts = []
    for i, ch in enumerate(_HEAT_CHARS):
        if i == 0:
            legend_parts.append(f"'{ch}'=0")
        else:
            lo = math.ceil(data.max_count * (i - 1) / (len(_HEAT_CHARS) - 1)) + (1 if i > 1 else 1)
            hi = math.ceil(data.max_count * i / (len(_HEAT_CHARS) - 1))
            legend_parts.append(f"'{ch}'={lo}-{hi}")
    lines.append("Legend: " + "  ".join(legend_parts))

    return "\n".join(lines) + "\n"


# ── HTML snippet for embedding in HTML report ─────────────────────────

def heatmap_html_snippet(data: HeatmapData) -> str:
    """Return an HTML/JS snippet for an interactive heatmap chart.

    Designed to be embedded inside the existing HTML report template.
    Uses pure Canvas rendering (no external deps).
    """
    if not data.grid or data.max_count == 0:
        return "<p>No data available for heatmap.</p>"

    import json as _json

    grid_json = _json.dumps(data.grid)
    time_edges_json = _json.dumps([round(t, 2) for t in data.time_edges])
    lat_edges_json = _json.dumps([round(v, 1) for v in data.latency_edges])

    return f"""\
<div class="chart-box" style="grid-column:1/-1">
  <h3>Latency Heatmap</h3>
  <canvas id="heatmapCanvas"></canvas>
</div>
<script>
(function(){{
  var grid={grid_json};
  var timeEdges={time_edges_json};
  var latEdges={lat_edges_json};
  var maxCount={data.max_count};
  var c=document.getElementById('heatmapCanvas');
  var ctx=c.getContext('2d');
  var w=c.width=c.offsetWidth;
  var h=c.height=300;
  var pad={{t:10,r:60,b:40,l:60}};
  var pw=w-pad.l-pad.r, ph=h-pad.t-pad.b;
  var numT=grid.length, numL=grid[0].length;
  var cellW=pw/numT, cellH=ph/numL;

  // Draw cells
  for(var ti=0;ti<numT;ti++){{
    for(var li=0;li<numL;li++){{
      var count=grid[ti][li];
      var intensity=maxCount>0?count/maxCount:0;
      // Color: dark blue (0) -> yellow (0.5) -> red (1)
      var r,g,b;
      if(intensity===0){{r=13;g=17;b=23;}}
      else if(intensity<0.5){{
        var t2=intensity*2;
        r=Math.round(13+(255-13)*t2);
        g=Math.round(17+(200-17)*t2);
        b=Math.round(23+(0-23)*t2);
      }}else{{
        var t2=(intensity-0.5)*2;
        r=255;
        g=Math.round(200*(1-t2));
        b=0;
      }}
      ctx.fillStyle='rgb('+r+','+g+','+b+')';
      ctx.fillRect(pad.l+ti*cellW, pad.t+(numL-1-li)*cellH, cellW+1, cellH+1);
    }}
  }}

  // Y-axis labels (latency)
  ctx.fillStyle='#8b949e';
  ctx.font='11px sans-serif';
  ctx.textAlign='right';
  ctx.textBaseline='middle';
  var yLabelStep=Math.max(1,Math.floor(numL/5));
  for(var li=0;li<=numL;li+=yLabelStep){{
    var y=pad.t+(numL-li)*cellH;
    ctx.fillText(latEdges[li].toFixed(0)+'ms',pad.l-4,y);
  }}

  // X-axis labels (time)
  ctx.textAlign='center';
  ctx.textBaseline='top';
  var xLabelStep=Math.max(1,Math.floor(numT/5));
  for(var ti=0;ti<=numT;ti+=xLabelStep){{
    var x=pad.l+ti*cellW;
    ctx.fillText(timeEdges[ti].toFixed(1)+'s',x,pad.t+ph+4);
  }}

  // Color legend
  var lw=150,lh=12,lx=w-pad.r+8,ly=pad.t;
  ctx.font='10px sans-serif';
  ctx.textAlign='left';
  ctx.fillStyle='#8b949e';
  ctx.fillText('Count',lx,ly-2);
  for(var i=0;i<lw;i++){{
    var frac=i/lw;
    var r2,g2,b2;
    if(frac<0.5){{var t3=frac*2;r2=Math.round(13+242*t3);g2=Math.round(17+183*t3);
      b2=Math.round(23-23*t3);}}
    else{{var t3=(frac-0.5)*2;r2=255;g2=Math.round(200*(1-t3));b2=0;}}
    ctx.fillStyle='rgb('+r2+','+g2+','+b2+')';
    ctx.fillRect(lx+i*0.3,ly+10,1,lh);
  }}
  ctx.fillStyle='#8b949e';
  ctx.fillText('0',lx,ly+10+lh+2);
  ctx.fillText(maxCount.toString(),lx+lw*0.3,ly+10+lh+2);
}})();
</script>"""

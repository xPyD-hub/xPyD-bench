"""Self-contained HTML report generator for benchmark results.

Generates an interactive, offline-capable HTML dashboard with:
- Summary stats table with color-coded thresholds
- Latency distribution histogram
- Throughput timeline chart
- TTFT CDF curve
- Per-request latency scatter plot

All CSS and JS are embedded inline — no external dependencies.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from xpyd_bench.bench.models import BenchmarkResult
from xpyd_bench.reporting.metrics import compute_time_series


def _safe(v: Any) -> str:
    """HTML-escape a value."""
    return html.escape(str(v))


def _build_summary_rows(result: BenchmarkResult) -> str:
    """Build HTML table rows for the summary stats."""
    rows = []

    def _row(label: str, value: str, unit: str = "") -> None:
        rows.append(f"<tr><td>{_safe(label)}</td>"
                     f"<td class='val'>{_safe(value)}</td>"
                     f"<td class='unit'>{_safe(unit)}</td></tr>")

    _row("Backend", result.backend)
    _row("Model", result.model)
    _row("Base URL", result.base_url)
    _row("Endpoint", result.endpoint)
    _row("Num Prompts", str(result.num_prompts))
    _row("Completed", str(result.completed))
    _row("Failed", str(result.failed))
    if result.partial:
        _row("Status", "PARTIAL (interrupted)")
    _row("Total Duration", f"{result.total_duration_s:.2f}", "s")
    _row("Request Throughput", f"{result.request_throughput:.2f}", "req/s")
    _row("Output Throughput", f"{result.output_throughput:.2f}", "tok/s")
    _row("Total Token Throughput", f"{result.total_token_throughput:.2f}", "tok/s")
    return "\n".join(rows)


def _build_percentile_rows(result: BenchmarkResult) -> str:
    """Build HTML table rows for latency percentiles."""
    rows = []
    for label, prefix in [("TTFT", "ttft"), ("TPOT", "tpot"),
                           ("ITL", "itl"), ("E2E Latency", "e2el")]:
        mean = getattr(result, f"mean_{prefix}_ms")
        p50 = getattr(result, f"p50_{prefix}_ms")
        p90 = getattr(result, f"p90_{prefix}_ms")
        p95 = getattr(result, f"p95_{prefix}_ms")
        p99 = getattr(result, f"p99_{prefix}_ms")
        rows.append(
            f"<tr><td>{_safe(label)}</td>"
            f"<td>{mean:.2f}</td><td>{p50:.2f}</td>"
            f"<td>{p90:.2f}</td><td>{p95:.2f}</td>"
            f"<td class='p99'>{p99:.2f}</td></tr>"
        )
    return "\n".join(rows)


def _chart_data(result: BenchmarkResult) -> dict:
    """Prepare JSON-serializable chart data for embedded JS."""
    successful = [r for r in result.requests if r.success]
    latencies = [r.latency_ms for r in successful]
    ttfts = [r.ttft_ms for r in successful if r.ttft_ms is not None]
    tpots = [r.tpot_ms for r in successful if r.tpot_ms is not None]

    ts = compute_time_series(result, window_s=1.0)

    return {
        "latencies": latencies,
        "ttfts": ttfts,
        "tpots": tpots,
        "timeSeries": ts,
        "numRequests": len(successful),
    }


_CSS = """\
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
background:#0d1117;color:#c9d1d9;padding:24px;max-width:1200px;margin:0 auto}
h1{color:#58a6ff;margin-bottom:8px}
h2{color:#8b949e;margin:24px 0 12px;font-size:1.1em;text-transform:uppercase;
letter-spacing:1px}
.subtitle{color:#8b949e;margin-bottom:24px}
table{width:100%;border-collapse:collapse;margin-bottom:16px}
th,td{padding:6px 12px;text-align:left;border-bottom:1px solid #21262d}
th{color:#8b949e;font-weight:600}
.val{font-weight:600;color:#f0f6fc}
.unit{color:#8b949e}
.p99{color:#f85149}
.chart-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px}
@media(max-width:800px){.chart-grid{grid-template-columns:1fr}}
.chart-box{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px}
.chart-box h3{color:#8b949e;font-size:0.85em;margin-bottom:8px;text-transform:uppercase}
canvas{width:100%!important;height:250px!important}
.footer{color:#484f58;text-align:center;margin-top:32px;font-size:0.8em}
"""

_JS = """\
function histogram(canvasId,data,label,color){
  if(!data.length)return;
  var c=document.getElementById(canvasId);
  var ctx=c.getContext('2d');
  var w=c.width=c.offsetWidth;
  var h=c.height=250;
  var pad={t:10,r:10,b:30,l:50};
  var pw=w-pad.l-pad.r,ph=h-pad.t-pad.b;

  var mn=Math.min.apply(null,data),mx=Math.max.apply(null,data);
  if(mn===mx){mn=mn-1;mx=mx+1}
  var bins=Math.min(50,Math.max(10,Math.ceil(Math.sqrt(data.length))));
  var bw=(mx-mn)/bins;
  var counts=new Array(bins).fill(0);
  for(var i=0;i<data.length;i++){
    var bi=Math.min(Math.floor((data[i]-mn)/bw),bins-1);
    counts[bi]++;
  }
  var maxC=Math.max.apply(null,counts);

  ctx.fillStyle='#0d1117';ctx.fillRect(0,0,w,h);
  // bars
  for(var i=0;i<bins;i++){
    var bh=(counts[i]/maxC)*ph;
    var x=pad.l+i*(pw/bins);
    var y=pad.t+ph-bh;
    ctx.fillStyle=color;
    ctx.fillRect(x,y,pw/bins-1,bh);
  }
  // axes
  ctx.strokeStyle='#30363d';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(pad.l,pad.t);ctx.lineTo(pad.l,h-pad.b);
  ctx.lineTo(w-pad.r,h-pad.b);ctx.stroke();
  // labels
  ctx.fillStyle='#8b949e';ctx.font='11px sans-serif';ctx.textAlign='center';
  ctx.fillText(mn.toFixed(1)+' ms',pad.l,h-8);
  ctx.fillText(mx.toFixed(1)+' ms',w-pad.r,h-8);
  ctx.fillText(label,w/2,h-8);
}

function timeLine(canvasId,series){
  if(!series.length)return;
  var c=document.getElementById(canvasId);
  var ctx=c.getContext('2d');
  var w=c.width=c.offsetWidth;
  var h=c.height=250;
  var pad={t:10,r:10,b:30,l:50};
  var pw=w-pad.l-pad.r,ph=h-pad.t-pad.b;

  var xs=series.map(function(s){return s.window_start_s});
  var ys=series.map(function(s){return s.output_throughput});
  var maxY=Math.max.apply(null,ys)||1;
  var maxX=series[series.length-1].window_end_s||1;

  ctx.fillStyle='#0d1117';ctx.fillRect(0,0,w,h);
  // line
  ctx.strokeStyle='#3fb950';ctx.lineWidth=2;ctx.beginPath();
  for(var i=0;i<xs.length;i++){
    var x=pad.l+(xs[i]/maxX)*pw;
    var y=pad.t+ph-(ys[i]/maxY)*ph;
    if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
  }
  ctx.stroke();
  // axes
  ctx.strokeStyle='#30363d';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(pad.l,pad.t);ctx.lineTo(pad.l,h-pad.b);
  ctx.lineTo(w-pad.r,h-pad.b);ctx.stroke();
  ctx.fillStyle='#8b949e';ctx.font='11px sans-serif';ctx.textAlign='center';
  ctx.fillText('Time (s)',w/2,h-8);
  ctx.textAlign='right';
  ctx.fillText(maxY.toFixed(0)+' tok/s',pad.l-4,pad.t+10);
}

function cdf(canvasId,data,color){
  if(!data.length)return;
  var c=document.getElementById(canvasId);
  var ctx=c.getContext('2d');
  var w=c.width=c.offsetWidth;
  var h=c.height=250;
  var pad={t:10,r:10,b:30,l:50};
  var pw=w-pad.l-pad.r,ph=h-pad.t-pad.b;

  var sorted=data.slice().sort(function(a,b){return a-b});
  var n=sorted.length;
  var mn=sorted[0],mx=sorted[n-1];
  if(mn===mx){mn=mn-1;mx=mx+1}

  ctx.fillStyle='#0d1117';ctx.fillRect(0,0,w,h);
  ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();
  for(var i=0;i<n;i++){
    var x=pad.l+((sorted[i]-mn)/(mx-mn))*pw;
    var y=pad.t+ph-((i+1)/n)*ph;
    if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
  }
  ctx.stroke();
  ctx.strokeStyle='#30363d';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(pad.l,pad.t);ctx.lineTo(pad.l,h-pad.b);
  ctx.lineTo(w-pad.r,h-pad.b);ctx.stroke();
  ctx.fillStyle='#8b949e';ctx.font='11px sans-serif';ctx.textAlign='center';
  ctx.fillText(mn.toFixed(1)+' ms',pad.l,h-8);
  ctx.fillText(mx.toFixed(1)+' ms',w-pad.r,h-8);
}

function scatter(canvasId,data,color){
  if(!data.length)return;
  var c=document.getElementById(canvasId);
  var ctx=c.getContext('2d');
  var w=c.width=c.offsetWidth;
  var h=c.height=250;
  var pad={t:10,r:10,b:30,l:50};
  var pw=w-pad.l-pad.r,ph=h-pad.t-pad.b;

  var maxY=Math.max.apply(null,data)||1;
  ctx.fillStyle='#0d1117';ctx.fillRect(0,0,w,h);
  for(var i=0;i<data.length;i++){
    var x=pad.l+(i/(data.length-1||1))*pw;
    var y=pad.t+ph-(data[i]/maxY)*ph;
    ctx.fillStyle=color;
    ctx.beginPath();ctx.arc(x,y,2,0,Math.PI*2);ctx.fill();
  }
  ctx.strokeStyle='#30363d';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(pad.l,pad.t);ctx.lineTo(pad.l,h-pad.b);
  ctx.lineTo(w-pad.r,h-pad.b);ctx.stroke();
  ctx.fillStyle='#8b949e';ctx.font='11px sans-serif';ctx.textAlign='center';
  ctx.fillText('Request #',w/2,h-8);
  ctx.textAlign='right';
  ctx.fillText(maxY.toFixed(0)+' ms',pad.l-4,pad.t+10);
}

window.addEventListener('load',function(){
  var D=window.__BENCH_DATA__;
  histogram('hist-latency',D.latencies,'E2E Latency','#58a6ff');
  histogram('hist-ttft',D.ttfts,'TTFT','#d2a8ff');
  timeLine('timeline',D.timeSeries);
  cdf('cdf-ttft',D.ttfts,'#d2a8ff');
  scatter('scatter-latency',D.latencies,'#58a6ff');
});
"""


def _render_html(result: BenchmarkResult) -> str:
    """Render the full HTML report string."""
    data = _chart_data(result)
    data_json = json.dumps(data, default=str)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>xPyD-bench Report</title>
<style>{_CSS}</style>
</head>
<body>
<h1>xPyD-bench Report</h1>
<p class="subtitle">{_safe(result.model)} &mdash; {_safe(result.base_url)}</p>

<h2>Summary</h2>
<table>
<tbody>
{_build_summary_rows(result)}
</tbody>
</table>

<h2>Latency Percentiles (ms)</h2>
<table>
<thead>
<tr><th>Metric</th><th>Mean</th><th>P50</th><th>P90</th><th>P95</th><th>P99</th></tr>
</thead>
<tbody>
{_build_percentile_rows(result)}
</tbody>
</table>

<h2>Charts</h2>
<div class="chart-grid">
  <div class="chart-box"><h3>E2E Latency Distribution</h3><canvas id="hist-latency"></canvas></div>
  <div class="chart-box"><h3>TTFT Distribution</h3><canvas id="hist-ttft"></canvas></div>
  <div class="chart-box"><h3>Throughput Timeline</h3><canvas id="timeline"></canvas></div>
  <div class="chart-box"><h3>TTFT CDF</h3><canvas id="cdf-ttft"></canvas></div>
  <div class="chart-box" style="grid-column:1/-1">
    <h3>Per-Request Latency</h3><canvas id="scatter-latency"></canvas>
  </div>
</div>

<p class="footer">Generated by xPyD-bench</p>

<script>
window.__BENCH_DATA__={data_json};
{_JS}
</script>
</body>
</html>"""


def export_html_report(result: BenchmarkResult, path: str | Path) -> Path:
    """Export an interactive HTML report.

    Returns the written path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_render_html(result))
    return p

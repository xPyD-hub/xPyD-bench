"""Warmup Curve Analysis (M90).

Mathematical modeling of latency convergence during server warmup.
Fits an exponential decay model to initial request latencies and
detects the convergence point automatically.

Different from M51 (warmup profiling): focuses on exponential decay
curve fitting and ASCII visualization rather than simple stabilization
detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class WarmupCurveResult:
    """Results of warmup curve analysis."""

    latencies_ms: list[float] = field(default_factory=list)
    fit_a: float = 0.0  # amplitude
    fit_b: float = 0.0  # decay rate
    fit_c: float = 0.0  # steady-state offset
    convergence_index: int | None = None
    cold_start_penalty_ms: float = 0.0
    steady_state_ms: float = 0.0
    fit_r_squared: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON output."""
        return {
            "latencies_ms": [round(v, 3) for v in self.latencies_ms],
            "fit_params": {
                "a": round(self.fit_a, 6),
                "b": round(self.fit_b, 6),
                "c": round(self.fit_c, 6),
            },
            "fit_r_squared": round(self.fit_r_squared, 6),
            "convergence_index": self.convergence_index,
            "cold_start_penalty_ms": round(self.cold_start_penalty_ms, 3),
            "steady_state_ms": round(self.steady_state_ms, 3),
        }


def _exp_decay(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Exponential decay: f(x) = a * exp(-b * x) + c."""
    return a * np.exp(-b * x) + c


def fit_exponential_decay(
    latencies: list[float],
) -> tuple[float, float, float, float]:
    """Fit exponential decay model to latency data.

    Parameters
    ----------
    latencies : list[float]
        Per-request latencies in ms.

    Returns
    -------
    tuple of (a, b, c, r_squared)
        Fit parameters and R² goodness of fit.
    """
    n = len(latencies)
    if n < 3:
        # Not enough data to fit; return flat line at mean.
        mean_val = float(np.mean(latencies)) if latencies else 0.0
        return 0.0, 0.0, mean_val, 0.0

    y = np.array(latencies, dtype=np.float64)
    x = np.arange(n, dtype=np.float64)

    # Initial guesses: a = max - min, b = 0.1, c = min
    c0 = float(np.min(y))
    a0 = float(np.max(y)) - c0
    b0 = 0.1

    if a0 <= 0:
        # Flat or increasing — no decay pattern.
        mean_val = float(np.mean(y))
        return 0.0, 0.0, mean_val, 0.0

    # Simple iterative least-squares via scipy-free approach:
    # Use Gauss-Newton-like iteration with log-linearization.
    # For robustness without scipy, use grid search + refinement.
    best_a, best_b, best_c = a0, b0, c0
    best_residual = np.inf

    for b_candidate in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
        # Given b, solve for a and c via linear least squares:
        # y = a * exp(-b * x) + c  →  [exp(-b*x), 1] @ [a, c] = y
        basis = np.column_stack([np.exp(-b_candidate * x), np.ones(n)])
        try:
            coeffs, residuals, _, _ = np.linalg.lstsq(basis, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        a_fit, c_fit = float(coeffs[0]), float(coeffs[1])
        pred = a_fit * np.exp(-b_candidate * x) + c_fit
        ss_res = float(np.sum((y - pred) ** 2))
        if ss_res < best_residual:
            best_residual = ss_res
            best_a, best_b, best_c = a_fit, b_candidate, c_fit

    # Refine b around best candidate.
    for delta in [-0.04, -0.02, -0.01, 0.01, 0.02, 0.04]:
        b_try = best_b + delta
        if b_try <= 0:
            continue
        basis = np.column_stack([np.exp(-b_try * x), np.ones(n)])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(basis, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        a_fit, c_fit = float(coeffs[0]), float(coeffs[1])
        pred = a_fit * np.exp(-b_try * x) + c_fit
        ss_res = float(np.sum((y - pred) ** 2))
        if ss_res < best_residual:
            best_residual = ss_res
            best_a, best_b, best_c = a_fit, b_try, c_fit

    # R²
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - best_residual / ss_tot if ss_tot > 0 else 0.0
    r_squared = max(0.0, r_squared)  # Clamp negative R² to 0.

    return best_a, best_b, best_c, r_squared


def detect_convergence(
    latencies: list[float],
    a: float,
    b: float,
    c: float,
    threshold_pct: float = 0.05,
) -> int | None:
    """Find the request index where the curve converges to steady state.

    Convergence defined as: fitted value within *threshold_pct* of the
    steady-state value *c*.

    Parameters
    ----------
    latencies : list[float]
        Per-request latencies (used for length).
    a, b, c : float
        Exponential decay fit parameters.
    threshold_pct : float
        Fraction of *a* considered negligible (default 5%).

    Returns
    -------
    int | None
        First request index at convergence, or None.
    """
    if b <= 0 or a <= 0:
        return 0  # No decay → already converged.

    n = len(latencies)
    threshold = abs(a) * threshold_pct
    for i in range(n):
        remaining = abs(a) * np.exp(-b * i)
        if remaining <= threshold:
            return i
    return None


def render_ascii_curve(
    latencies: list[float],
    a: float,
    b: float,
    c: float,
    convergence_index: int | None,
    width: int = 60,
    height: int = 15,
) -> str:
    """Render ASCII art of the latency curve with fitted line.

    Parameters
    ----------
    latencies : list[float]
        Raw latencies.
    a, b, c : float
        Fit parameters.
    convergence_index : int | None
        Convergence point.
    width, height : int
        Character dimensions.

    Returns
    -------
    str
        Multi-line ASCII plot.
    """
    n = len(latencies)
    if n == 0:
        return "(no data)"

    y_raw = np.array(latencies, dtype=np.float64)
    x_arr = np.arange(n, dtype=np.float64)
    y_fit = _exp_decay(x_arr, a, b, c)

    y_min = min(float(np.min(y_raw)), float(np.min(y_fit)))
    y_max = max(float(np.max(y_raw)), float(np.max(y_fit)))
    if y_max <= y_min:
        y_max = y_min + 1.0

    def _scale_x(idx: int) -> int:
        return min(int(idx / max(n - 1, 1) * (width - 1)), width - 1)

    def _scale_y(val: float) -> int:
        return height - 1 - min(
            int((val - y_min) / (y_max - y_min) * (height - 1)), height - 1
        )

    # Build grid.
    grid = [[" "] * width for _ in range(height)]

    # Plot fitted curve.
    for i in range(n):
        cx = _scale_x(i)
        cy = _scale_y(float(y_fit[i]))
        if 0 <= cy < height and 0 <= cx < width:
            grid[cy][cx] = "-"

    # Plot raw data points (overwrite fit line).
    for i in range(n):
        cx = _scale_x(i)
        cy = _scale_y(float(y_raw[i]))
        if 0 <= cy < height and 0 <= cx < width:
            grid[cy][cx] = "*"

    # Mark convergence point.
    if convergence_index is not None and 0 <= convergence_index < n:
        cx = _scale_x(convergence_index)
        for row in range(height):
            if grid[row][cx] == " ":
                grid[row][cx] = "|"

    lines = []
    lines.append(f"Latency (ms)  [{y_max:.1f}]")
    for row in grid:
        lines.append("  " + "".join(row))
    lines.append(f"              [{y_min:.1f}]")
    lines.append(f"  Request index: 0 → {n - 1}")
    if convergence_index is not None:
        lines.append(f"  Convergence at request #{convergence_index} (|)")
    lines.append("  * = observed   - = fitted curve")
    return "\n".join(lines)


def build_warmup_curve(
    latencies_ms: list[float],
    threshold_pct: float = 0.05,
) -> WarmupCurveResult:
    """Build warmup curve analysis from request latencies.

    Parameters
    ----------
    latencies_ms : list[float]
        Per-request latencies in ms for the initial portion of the benchmark.
    threshold_pct : float
        Convergence threshold as fraction of amplitude (default 5%).

    Returns
    -------
    WarmupCurveResult
    """
    a, b, c, r_sq = fit_exponential_decay(latencies_ms)
    conv = detect_convergence(latencies_ms, a, b, c, threshold_pct)

    # Cold-start penalty: first fitted value minus steady state.
    if latencies_ms:
        cold_penalty = float(latencies_ms[0]) - c
    else:
        cold_penalty = 0.0

    return WarmupCurveResult(
        latencies_ms=list(latencies_ms),
        fit_a=a,
        fit_b=b,
        fit_c=c,
        convergence_index=conv,
        cold_start_penalty_ms=max(cold_penalty, 0.0),
        steady_state_ms=c,
        fit_r_squared=r_sq,
    )


def print_warmup_curve(result: WarmupCurveResult) -> None:
    """Print warmup curve analysis to terminal."""
    print("\n--- Warmup Curve Analysis (M90) ---")
    print(f"  Requests analyzed: {len(result.latencies_ms)}")
    print(
        f"  Fit: f(x) = {result.fit_a:.2f} * exp(-{result.fit_b:.4f} * x)"
        f" + {result.fit_c:.2f}"
    )
    print(f"  R²: {result.fit_r_squared:.4f}")
    print(f"  Cold-start penalty: {result.cold_start_penalty_ms:.2f} ms")
    print(f"  Steady-state latency: {result.steady_state_ms:.2f} ms")
    if result.convergence_index is not None:
        print(f"  Convergence at request #{result.convergence_index}")
    else:
        print("  Convergence: not detected within sample")
    print()
    print(
        render_ascii_curve(
            result.latencies_ms,
            result.fit_a,
            result.fit_b,
            result.fit_c,
            result.convergence_index,
        )
    )
    print()

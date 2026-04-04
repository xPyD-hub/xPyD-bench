"""Request latency anomaly detection using IQR method."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AnomalyInfo:
    """Information about a single anomalous request."""

    index: int
    latency_ms: float
    deviation_factor: float  # how many IQRs above Q3


@dataclass
class AnomalyResult:
    """Anomaly detection result for a benchmark run."""

    anomalies: list[AnomalyInfo]
    q1: float
    q3: float
    iqr: float
    threshold: float  # Q3 + multiplier * IQR
    multiplier: float

    @property
    def count(self) -> int:
        return len(self.anomalies)

    def to_dict(self) -> dict:
        """Serialize to JSON-friendly dict."""
        return {
            "count": self.count,
            "q1_ms": round(self.q1, 2),
            "q3_ms": round(self.q3, 2),
            "iqr_ms": round(self.iqr, 2),
            "threshold_ms": round(self.threshold, 2),
            "multiplier": self.multiplier,
            "flagged_requests": [
                {
                    "index": a.index,
                    "latency_ms": round(a.latency_ms, 2),
                    "deviation_factor": round(a.deviation_factor, 2),
                }
                for a in self.anomalies
            ],
        }


def detect_anomalies(
    latencies_ms: list[float],
    multiplier: float = 1.5,
) -> AnomalyResult | None:
    """Detect latency anomalies using IQR method.

    Args:
        latencies_ms: Per-request latencies in milliseconds.
        multiplier: IQR multiplier for threshold (0 disables detection).

    Returns:
        AnomalyResult if detection ran, None if disabled or insufficient data.
    """
    if multiplier <= 0 or len(latencies_ms) < 4:
        return None

    arr = np.array(latencies_ms)
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1
    threshold = q3 + multiplier * iqr

    anomalies: list[AnomalyInfo] = []
    for i, lat in enumerate(latencies_ms):
        if lat > threshold:
            deviation = (lat - q3) / iqr if iqr > 0 else float("inf")
            anomalies.append(
                AnomalyInfo(index=i, latency_ms=lat, deviation_factor=deviation)
            )

    # Sort by latency descending (worst first)
    anomalies.sort(key=lambda a: a.latency_ms, reverse=True)

    return AnomalyResult(
        anomalies=anomalies,
        q1=q1,
        q3=q3,
        iqr=iqr,
        threshold=threshold,
        multiplier=multiplier,
    )

"""JUnit XML export for CI integration (M49).

Generates JUnit XML from :class:`BenchmarkResult` so that CI systems
(Jenkins, GitHub Actions, GitLab CI) can natively display benchmark
outcomes.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from xpyd_bench.bench.models import BenchmarkResult


def _request_testcases(result: BenchmarkResult) -> list[ET.Element]:
    """Create a ``<testcase>`` for every request in *result*."""
    cases: list[ET.Element] = []
    for idx, req in enumerate(result.requests):
        tc = ET.SubElement(ET.Element("_"), "testcase")  # detached; re-parented later
        tc = ET.Element("testcase")
        name = f"request-{idx:04d}"
        if req.request_id:
            name = f"request-{req.request_id}"
        tc.set("name", name)
        tc.set("classname", "xpyd-bench.requests")
        tc.set("time", f"{req.latency_ms / 1000:.6f}")

        if not req.success:
            fail = ET.SubElement(tc, "failure")
            fail.set("message", req.error or "request failed")
            fail.text = req.error or ""

        if req.validation_errors:
            fail = ET.SubElement(tc, "failure")
            msg = "; ".join(req.validation_errors)
            fail.set("message", msg)
            fail.text = msg

        cases.append(tc)
    return cases


def _sla_testcases(result: BenchmarkResult) -> list[ET.Element]:
    """Create ``<testcase>`` elements for SLA checks stored in *result*."""
    sla = getattr(result, "sla_results", None)
    if not sla:
        return []

    cases: list[ET.Element] = []
    targets = sla if isinstance(sla, list) else sla.get("targets", [])
    for item in targets:
        tc = ET.Element("testcase")
        metric = item.get("metric", "unknown")
        tc.set("name", f"sla-{metric}")
        tc.set("classname", "xpyd-bench.sla")
        tc.set("time", "0")

        if not item.get("pass", True):
            fail = ET.SubElement(tc, "failure")
            actual = item.get("actual", "?")
            threshold = item.get("threshold", "?")
            msg = f"{metric}: actual={actual}, threshold={threshold}"
            fail.set("message", msg)
            fail.text = msg

        cases.append(tc)
    return cases


def build_junit_xml(result: BenchmarkResult) -> ET.Element:
    """Return a JUnit XML ``<testsuites>`` :class:`ET.Element`."""
    root = ET.Element("testsuites")

    # --- requests testsuite ---
    req_cases = _request_testcases(result)
    ts_req = ET.SubElement(root, "testsuite")
    ts_req.set("name", "xpyd-bench.requests")
    ts_req.set("tests", str(len(req_cases)))
    failures = sum(1 for tc in req_cases if tc.find("failure") is not None)
    ts_req.set("failures", str(failures))
    ts_req.set("errors", "0")
    ts_req.set("time", f"{result.total_duration_s:.6f}")
    for tc in req_cases:
        ts_req.append(tc)

    # --- SLA testsuite (only when SLA results exist) ---
    sla_cases = _sla_testcases(result)
    if sla_cases:
        ts_sla = ET.SubElement(root, "testsuite")
        ts_sla.set("name", "xpyd-bench.sla")
        ts_sla.set("tests", str(len(sla_cases)))
        sla_failures = sum(1 for tc in sla_cases if tc.find("failure") is not None)
        ts_sla.set("failures", str(sla_failures))
        ts_sla.set("errors", "0")
        ts_sla.set("time", "0")
        for tc in sla_cases:
            ts_sla.append(tc)

    return root


def write_junit_xml(result: BenchmarkResult, path: str | Path) -> None:
    """Build JUnit XML from *result* and write it to *path*."""
    root = build_junit_xml(result)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    with open(path, "wb") as fh:
        tree.write(fh, encoding="utf-8", xml_declaration=True)

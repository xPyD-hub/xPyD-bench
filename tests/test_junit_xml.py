"""Tests for M49: JUnit XML Export."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from xpyd_bench.bench.models import BenchmarkResult, RequestResult
from xpyd_bench.junit_xml import build_junit_xml, write_junit_xml


def _make_result(
    requests: list[RequestResult] | None = None,
    sla_results: list[dict] | None = None,
) -> BenchmarkResult:
    r = BenchmarkResult(
        base_url="http://localhost:8000",
        endpoint="/v1/completions",
        model="test-model",
        num_prompts=3,
        total_duration_s=1.5,
        completed=3,
        requests=requests or [],
    )
    if sla_results is not None:
        r.sla_results = sla_results  # type: ignore[attr-defined]
    return r


class TestBuildJunitXml:
    """Unit tests for build_junit_xml."""

    def test_basic_structure(self) -> None:
        reqs = [
            RequestResult(latency_ms=100.0, success=True),
            RequestResult(latency_ms=200.0, success=True),
        ]
        root = build_junit_xml(_make_result(requests=reqs))
        assert root.tag == "testsuites"
        suites = root.findall("testsuite")
        assert len(suites) == 1
        ts = suites[0]
        assert ts.get("name") == "xpyd-bench.requests"
        assert ts.get("tests") == "2"
        assert ts.get("failures") == "0"

    def test_failed_requests_produce_failures(self) -> None:
        reqs = [
            RequestResult(latency_ms=100.0, success=True),
            RequestResult(latency_ms=50.0, success=False, error="timeout"),
        ]
        root = build_junit_xml(_make_result(requests=reqs))
        ts = root.find("testsuite")
        assert ts is not None
        assert ts.get("failures") == "1"
        cases = ts.findall("testcase")
        fail_case = cases[1]
        fail_elem = fail_case.find("failure")
        assert fail_elem is not None
        assert "timeout" in (fail_elem.get("message") or "")

    def test_validation_errors_produce_failures(self) -> None:
        reqs = [
            RequestResult(
                latency_ms=100.0,
                success=True,
                validation_errors=["empty response"],
            ),
        ]
        root = build_junit_xml(_make_result(requests=reqs))
        ts = root.find("testsuite")
        assert ts is not None
        assert ts.get("failures") == "1"

    def test_request_id_in_testcase_name(self) -> None:
        reqs = [
            RequestResult(latency_ms=100.0, success=True, request_id="abc-123"),
        ]
        root = build_junit_xml(_make_result(requests=reqs))
        tc = root.find(".//testcase")
        assert tc is not None
        assert "abc-123" in (tc.get("name") or "")

    def test_sla_results_create_separate_testsuite(self) -> None:
        reqs = [RequestResult(latency_ms=100.0, success=True)]
        sla = [
            {"metric": "p99_ttft_ms", "pass": True, "actual": 50, "threshold": 100},
            {"metric": "throughput", "pass": False, "actual": 5, "threshold": 10},
        ]
        root = build_junit_xml(_make_result(requests=reqs, sla_results=sla))
        suites = root.findall("testsuite")
        assert len(suites) == 2
        sla_suite = suites[1]
        assert sla_suite.get("name") == "xpyd-bench.sla"
        assert sla_suite.get("tests") == "2"
        assert sla_suite.get("failures") == "1"

    def test_no_sla_no_extra_suite(self) -> None:
        reqs = [RequestResult(latency_ms=100.0, success=True)]
        root = build_junit_xml(_make_result(requests=reqs))
        suites = root.findall("testsuite")
        assert len(suites) == 1

    def test_latency_as_time_attribute(self) -> None:
        reqs = [RequestResult(latency_ms=1234.5, success=True)]
        root = build_junit_xml(_make_result(requests=reqs))
        tc = root.find(".//testcase")
        assert tc is not None
        assert tc.get("time") == "1.234500"


class TestWriteJunitXml:
    """Integration tests for file writing."""

    def test_write_creates_valid_xml(self, tmp_path: Path) -> None:
        reqs = [
            RequestResult(latency_ms=100.0, success=True),
            RequestResult(latency_ms=200.0, success=False, error="conn refused"),
        ]
        out = tmp_path / "results.xml"
        write_junit_xml(_make_result(requests=reqs), out)
        assert out.exists()
        content = out.read_text("utf-8")
        assert '<?xml version' in content
        # Parse to verify validity
        tree = ET.parse(out)
        root = tree.getroot()
        assert root.tag == "testsuites"

    def test_empty_requests(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.xml"
        write_junit_xml(_make_result(requests=[]), out)
        tree = ET.parse(out)
        ts = tree.find(".//testsuite")
        assert ts is not None
        assert ts.get("tests") == "0"
        assert ts.get("failures") == "0"

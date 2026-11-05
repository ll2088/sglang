"""Tests that error-terminated streams are excluded from success aggregates."""

from types import SimpleNamespace

from sglang.multimodal_gen.benchmarks.bench_serving import calculate_metrics
from sglang.multimodal_gen.benchmarks.datasets import (
    RequestFuncInput,
    RequestFuncOutput,
)


def _request() -> RequestFuncInput:
    return RequestFuncInput(
        prompt="p",
        api_url="http://localhost/v1/images/generations",
        model="m",
        num_outputs_per_prompt=1,
    )


def _args() -> SimpleNamespace:
    return SimpleNamespace(num_outputs_per_prompt=1, slo_scale=3.0)


def test_error_terminated_stream_not_counted_as_success():
    ok = RequestFuncOutput()
    ok.success = True
    ok.latency = 1.0
    ok.output_count = 1

    failed = RequestFuncOutput()
    failed.success = False
    failed.error = "internal server error during generation"
    failed.latency = 0.5
    failed.output_count = 1

    metrics = calculate_metrics(
        [ok, failed],
        total_duration=2.0,
        requests_list=[_request(), _request()],
        args=_args(),
        slo_enabled=False,
    )

    assert metrics["completed_requests"] == 1
    assert metrics["failed_requests"] == 1
    assert metrics["completed_outputs"] == 1
    assert metrics["throughput_qps"] == 0.5
    assert metrics["output_throughput_ops"] == 0.5


def test_clean_streams_still_counted_as_success():
    outputs = []
    for _ in range(2):
        out = RequestFuncOutput()
        out.success = True
        out.latency = 1.0
        out.output_count = 1
        outputs.append(out)

    metrics = calculate_metrics(
        outputs,
        total_duration=2.0,
        requests_list=[_request(), _request()],
        args=_args(),
        slo_enabled=False,
    )

    assert metrics["completed_requests"] == 2
    assert metrics["failed_requests"] == 0
    assert metrics["completed_outputs"] == 2

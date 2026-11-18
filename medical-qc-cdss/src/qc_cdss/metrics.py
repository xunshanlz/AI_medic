"""Prometheus 指标。

指标体系围绕三类问题设计：
1. 业务：质控量、问题分布、采纳率（采纳率在仓储层统计）；
2. 性能：端到端时延、TTFT、各阶段耗时（分位数）；
3. 质量：证据校验三态分布、降级率、结构化输出重试次数。

指标命名遵循 OpenTelemetry GenAI 语义约定中 ``gen_ai.*`` 的思路，
同时保留 Prometheus 惯例的 ``_total`` / ``_seconds`` 后缀。
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "METRICS_ENABLED",
    "QC_REQUESTS",
    "QC_LATENCY",
    "LLM_CALLS",
    "LLM_TTFT",
    "EVIDENCE_STATES",
    "DEGRADED_TOTAL",
    "CACHE_EVENTS",
    "STRUCTURED_RETRIES",
    "render_metrics",
]

try:  # pragma: no cover
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Histogram,
        generate_latest,
    )

    METRICS_ENABLED = True
    QC_REQUESTS = Counter("qc_requests_total", "质控请求数", ["status", "source"])
    QC_LATENCY = Histogram(
        "qc_request_latency_seconds", "质控端到端时延", ["status"],
        buckets=(0.5, 1, 2, 3, 5, 8, 13, 21, 34, 60),
    )
    LLM_CALLS = Counter("llm_calls_total", "模型调用次数", ["provider", "stage", "status"])
    LLM_TTFT = Histogram(
        "llm_ttft_seconds", "模型首 token 时延", ["provider"],
        buckets=(0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3, 6, 10),
    )
    EVIDENCE_STATES = Counter("evidence_verification_total", "证据校验结果", ["state"])
    DEGRADED_TOTAL = Counter("qc_degraded_total", "降级次数", ["reason"])
    CACHE_EVENTS = Counter("cache_events_total", "缓存命中", ["result"])
    STRUCTURED_RETRIES = Counter(
        "structured_output_retries_total", "结构化输出重试", ["schema", "reason"]
    )
except Exception:  # pragma: no cover - 未安装 prometheus_client 时降级为 no-op
    METRICS_ENABLED = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"

    class _Noop:
        def labels(self, *args: Any, **kwargs: Any) -> "_Noop":
            return self

        def inc(self, *args: Any, **kwargs: Any) -> None:
            return None

        def observe(self, *args: Any, **kwargs: Any) -> None:
            return None

    QC_REQUESTS = _Noop()
    QC_LATENCY = _Noop()
    LLM_CALLS = _Noop()
    LLM_TTFT = _Noop()
    EVIDENCE_STATES = _Noop()
    DEGRADED_TOTAL = _Noop()
    CACHE_EVENTS = _Noop()
    STRUCTURED_RETRIES = _Noop()


def render_metrics() -> tuple[bytes, str]:
    if not METRICS_ENABLED:  # pragma: no cover
        return b"# prometheus_client not installed\n", CONTENT_TYPE_LATEST
    return generate_latest(), CONTENT_TYPE_LATEST  # pragma: no cover

"""LangGraph 编排。

为什么用图而不是一条 async 函数
--------------------------------
1. **可观测的阶段边界**：每个节点是一个天然的 trace span，耗时、失败、降级
   都能单独归因；
2. **可扩展**：后续接入"人机协同"（医生修改后重新质控）、多轮追问、检查项
   灰度，只需加节点/改边，不必重构主流程；
3. **持久化执行**：编译时传入 checkpointer 即可支持断点续跑（本项目默认关闭，
   因为质控是短任务；长病程记录场景可开启 PostgresSaver）。

注意：图内不做并行 fan-out 的复杂 join，而是把"检查项 × 分片"的并行放在
``llm_qc`` 节点内部用有界并发完成——这样超时与降级语义更清晰，也便于
按请求统计并发占用。
"""

from __future__ import annotations

import operator
import time
import uuid
from typing import Annotated, Any, TypedDict

from ..domain.sections import Document
from ..logging import get_logger, new_trace_id
from ..schemas import Finding, IcdSuggestion, QcRequest, QcResult, StageTiming
from .nodes import Emitter, NodeDeps, QcNodes
from .scheduler import ConcurrencyController, Deadline, is_long_record

__all__ = ["QCState", "QcPipeline", "build_graph"]

logger = get_logger(__name__)


class QCState(TypedDict, total=False):
    request: QcRequest
    document: Document
    doc_hash: str
    deadline: Deadline
    emit: Emitter | None
    findings: Annotated[list[Finding], operator.add]
    icd: list[IcdSuggestion]
    degraded_reasons: Annotated[list[str], operator.add]
    timings: Annotated[list[StageTiming], operator.add]
    final: dict[str, Any]


def build_graph(nodes: QcNodes) -> Any:
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(QCState)
    builder.add_node("prepare", nodes.prepare)
    builder.add_node("rules", nodes.rules)
    builder.add_node("llm_qc", nodes.llm_qc)
    builder.add_node("icd", nodes.icd)
    builder.add_node("finalize", nodes.finalize)

    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "rules")
    builder.add_edge("rules", "llm_qc")
    builder.add_edge("llm_qc", "icd")
    builder.add_edge("icd", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()


class QcPipeline:
    def __init__(self, deps: NodeDeps) -> None:
        self.deps = deps
        self.nodes = QcNodes(deps)
        self.graph = build_graph(self.nodes)
        self.controller = ConcurrencyController(
            max_concurrency=deps.settings.max_concurrency,
            short_slots=deps.settings.short_request_slots,
        )

    async def run(
        self,
        request: QcRequest,
        *,
        job_id: str | None = None,
        trace_id: str | None = None,
        on_event: Emitter | None = None,
    ) -> QcResult:
        started = time.perf_counter()
        job_id = job_id or uuid.uuid4().hex
        trace_id = trace_id or new_trace_id()

        estimated = len(request.raw_text or "") or sum(
            len(str(value)) for value in (request.record or {}).values()
        )
        long_request = is_long_record(estimated)
        deadline = Deadline(self.deps.settings.request_deadline_s)

        initial: dict[str, Any] = {
            "request": request,
            "deadline": deadline,
            "emit": on_event,
        }

        try:
            async with self.controller.slot(long_request=long_request):
                state = await self.graph.ainvoke(initial, config={"recursion_limit": 32})
        except Exception as exc:  # noqa: BLE001 - 顶层兜底，保证接口有结构可解析
            logger.exception("pipeline_failed", job_id=job_id)
            return QcResult(
                job_id=job_id,
                trace_id=trace_id,
                status="failed",
                score=0.0,
                findings=[],
                icd=[],
                summary=f"质控执行失败：{type(exc).__name__}，请联系管理员并附 trace_id。",
                degraded=True,
                degraded_reasons=[f"pipeline:{type(exc).__name__}"],
                model=getattr(self.deps.generator.llm, "model", ""),
                prompt_version=self.deps.settings.prompt_version,
                timings=[],
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )

        final = state.get("final") or {}
        degraded_reasons = list(final.get("degraded_reasons") or [])
        return QcResult(
            job_id=job_id,
            trace_id=trace_id,
            status="degraded" if degraded_reasons else "succeeded",
            score=float(final.get("score", 0.0)),
            findings=list(final.get("findings") or []),
            icd=list(final.get("icd") or []),
            summary=str(final.get("summary", "")),
            degraded=bool(degraded_reasons),
            degraded_reasons=degraded_reasons[:10],
            model=getattr(self.deps.generator.llm, "model", ""),
            prompt_version=self.deps.settings.prompt_version,
            timings=list(state.get("timings") or []),
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            ttft_ms=None,
        )

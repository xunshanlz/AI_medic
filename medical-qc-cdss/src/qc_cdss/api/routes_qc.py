"""质控接口：同步、SSE 流式、任务查询、人工复核、采纳率统计。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from ..db.models import IcdSuggestionRecord, QcFinding, QcJob
from ..logging import get_logger, new_trace_id
from ..metrics import QC_LATENCY, QC_REQUESTS
from ..schemas import (
    EvidenceSpan,
    Finding,
    FindingCategory,
    IcdSuggestion,
    JobDetail,
    QcRequest,
    QcResult,
    ReviewRequest,
    Severity,
    StageTiming,
    StatsResponse,
    VerificationState,
)
from .deps import AppContainer, get_container, require_auth
from .sse import SSEBroker

__all__ = ["router"]

logger = get_logger(__name__)
router = APIRouter(prefix="/qc", tags=["质控"])


@router.post("/check", response_model=QcResult, summary="同步质控（返回完整结果）")
async def check(
    payload: QcRequest,
    container: AppContainer = Depends(get_container),
    auth: dict[str, Any] = Depends(require_auth),
) -> QcResult:
    trace_id = new_trace_id()
    result = await container.pipeline.run(payload, trace_id=trace_id)

    try:
        await container.repo.save_result(result, payload, doc_hash=_doc_hash(payload))
        await container.audit.record(
            trace_id=trace_id,
            actor=str(auth.get("sub", "system")),
            action="qc.check",
            payload={
                "job_id": result.job_id,
                "status": result.status,
                "score": result.score,
                "findings": len(result.findings),
                "degraded": result.degraded,
            },
        )
    except Exception:  # noqa: BLE001 - 落库失败不应让医生拿不到结果
        logger.exception("persist_qc_result_failed", job_id=result.job_id)

    QC_REQUESTS.labels(status=result.status, source="http").inc()
    QC_LATENCY.labels(status=result.status).observe(result.latency_ms / 1000)
    return result


@router.post("/check/stream", summary="SSE 流式质控（边算边推）")
async def check_stream(
    payload: QcRequest,
    container: AppContainer = Depends(get_container),
    auth: dict[str, Any] = Depends(require_auth),
) -> StreamingResponse:
    trace_id = new_trace_id()
    job_id = uuid.uuid4().hex
    broker = SSEBroker()

    async def runner() -> None:
        started = time.perf_counter()
        try:
            await broker.emit("job.started", {"job_id": job_id, "trace_id": trace_id})
            result = await container.pipeline.run(
                payload, job_id=job_id, trace_id=trace_id, on_event=broker.emit
            )
            result.ttft_ms = None
            try:
                await container.repo.save_result(result, payload, doc_hash=_doc_hash(payload))
                await container.audit.record(
                    trace_id=trace_id,
                    actor=str(auth.get("sub", "system")),
                    action="qc.check.stream",
                    payload={"job_id": result.job_id, "score": result.score},
                )
            except Exception:  # noqa: BLE001
                logger.exception("persist_stream_result_failed", job_id=job_id)

            QC_REQUESTS.labels(status=result.status, source="sse").inc()
            QC_LATENCY.labels(status=result.status).observe(
                (time.perf_counter() - started)
            )
            await broker.emit("result", result.model_dump(mode="json"))
        except Exception as exc:  # noqa: BLE001
            logger.exception("stream_qc_failed", job_id=job_id)
            await broker.emit(
                "error", {"message": f"{type(exc).__name__}", "trace_id": trace_id}
            )
        finally:
            await broker.close()

    task = asyncio.create_task(runner())

    async def events() -> AsyncIterator[str]:
        try:
            async for chunk in broker.stream():
                yield chunk
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/jobs/{job_id}", response_model=JobDetail, summary="查询质控结果")
async def get_job(
    job_id: str,
    container: AppContainer = Depends(get_container),
    _: dict[str, Any] = Depends(require_auth),
) -> JobDetail:
    job = await container.repo.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    rows = await container.repo.get_findings(job_id)
    findings = [_to_finding(row) for row in rows]
    result = _to_result(job, findings)
    return JobDetail(
        result=result,
        findings=findings,
        stats={
            "score": job.score,
            "findings": len(findings),
            "degraded": job.degraded,
        },
    )


@router.post("/findings/{finding_id}/review", summary="人工复核（采纳/驳回）")
async def review_finding(
    finding_id: int,
    payload: ReviewRequest,
    container: AppContainer = Depends(get_container),
    auth: dict[str, Any] = Depends(require_auth),
) -> dict[str, Any]:
    ok = await container.repo.update_review(
        finding_id,
        status=payload.status,
        reviewer=payload.reviewer,
        comment=payload.comment,
    )
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="问题不存在")

    await container.audit.record(
        trace_id=new_trace_id(),
        actor=payload.reviewer or str(auth.get("sub", "system")),
        action="finding.review",
        payload={"finding_id": finding_id, "status": payload.status, "comment": payload.comment},
    )
    return {"ok": True, "finding_id": finding_id, "status": payload.status}


@router.get("/stats", response_model=StatsResponse, summary="质控与采纳率统计")
async def stats(
    hospital_id: str | None = Query(default=None),
    container: AppContainer = Depends(get_container),
    _: dict[str, Any] = Depends(require_auth),
) -> StatsResponse:
    data = await container.repo.stats(hospital_id)
    return StatsResponse(**data)


# ---------------------------------------------------------------------- 转换
def _doc_hash(payload: QcRequest) -> str:
    """病历内容指纹：用于重复提交检测与缓存分析（不带 PII）。"""
    material = payload.raw_text or json.dumps(
        payload.record or {}, ensure_ascii=False, sort_keys=True
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _to_finding(row: QcFinding) -> Finding:
    evidence = None
    if row.quote:
        evidence = EvidenceSpan(
            start=row.start_offset,
            end=max(row.end_offset, row.start_offset),
            quote=row.quote,
            state=VerificationState(row.evidence_state),
            score=row.evidence_score,
        )
    return Finding(
        finding_id=str(row.id),
        code=row.code,
        title=row.title,
        category=FindingCategory(row.category),
        severity=Severity(row.severity),
        section=row.section,
        detail=row.detail,
        suggestion=row.suggestion,
        evidence=evidence,
        confidence=row.confidence,
        source="rule" if row.source == "rule" else "llm",
        needs_review=row.needs_review,
        review_status=row.review_status,
    )


def _to_icd(row: IcdSuggestionRecord) -> IcdSuggestion:
    evidence = None
    if row.quote:
        evidence = EvidenceSpan(start=0, end=0, quote=row.quote)
    return IcdSuggestion(
        code=row.code,
        name=row.name,
        confidence=row.confidence,
        rationale=row.rationale,
        evidence=evidence,
        verified=row.verified,
    )


def _to_result(job: QcJob, findings: list[Finding]) -> QcResult:
    return QcResult(
        job_id=job.id,
        trace_id=job.trace_id,
        status=job.status,
        score=job.score,
        findings=findings,
        icd=[_to_icd(row) for row in (job.icd_suggestions or [])],
        summary=job.summary,
        degraded=job.degraded,
        model=job.model,
        prompt_version=job.prompt_version,
        timings=[StageTiming(stage="persisted", ms=job.latency_ms)],
        latency_ms=job.latency_ms,
        ttft_ms=job.ttft_ms,
        created_at=job.created_at,
    )

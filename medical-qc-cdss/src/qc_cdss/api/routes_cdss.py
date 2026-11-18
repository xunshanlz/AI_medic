"""CDSS 接口：指南证据问答、ICD 编码推荐、急症前置分诊。

CDSS 与质控的职责边界：
- 质控：判断"病历写得对不对"，证据来自病历本身，不需要外部知识库；
- CDSS：判断"接下来该做什么"，必须检索指南并给出可追溯依据。
两者共用 LLM 网关、证据校验与审计，但检索链路只在 CDSS 中启用。
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from ..domain.evidence import VERIFIED
from ..domain.sections import build_document
from ..llm.prompts import CdssVerdict, cdss_messages
from ..logging import get_logger, new_trace_id
from ..privacy import mask_document, mask_pii
from ..schemas import (
    CdssAdvice,
    CdssEvidence,
    CdssRequest,
    CdssResponse,
    IcdRequest,
    IcdSuggestion,
    TriageRequest,
    TriageResponse,
    VerificationState,
)
from .deps import AppContainer, get_container, require_auth

__all__ = ["router"]

logger = get_logger(__name__)
router = APIRouter(prefix="/cdss", tags=["CDSS"])

_EMERGENCY = ("胸痛", "呼吸困难", "意识障碍", "昏迷", "抽搐", "大出血", "咯血", "偏瘫", "言语不清")
_URGENT = ("高热", "持续呕吐", "剧烈腹痛", "心悸", "晕厥", "黑便")


@router.post("/advice", response_model=CdssResponse, summary="指南证据驱动的临床建议")
async def advice(
    payload: CdssRequest,
    container: AppContainer = Depends(get_container),
    auth: dict[str, Any] = Depends(require_auth),
) -> CdssResponse:
    started = time.perf_counter()
    request_id = uuid.uuid4().hex
    trace_id = new_trace_id()

    if container.retriever is None or not container.retriever.chunks:
        return CdssResponse(
            request_id=request_id,
            advice=[],
            degraded=True,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    query = " ".join(filter(None, [payload.question, *payload.diagnoses]))
    contexts = await container.retriever.retrieve(query, top_k=payload.top_k)
    if not contexts:
        return CdssResponse(
            request_id=request_id,
            advice=[],
            degraded=True,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    question, _ = mask_pii(payload.question)
    messages = cdss_messages(
        question=question,
        contexts=[{"title": ctx.chunk.title, "text": ctx.chunk.text} for ctx in contexts],
        diagnoses=payload.diagnoses,
    )
    outcome = await container.generator.generate(
        CdssVerdict, messages, timeout=container.settings.branch_timeout_s, stage="cdss"
    )
    if outcome.value is None:
        return CdssResponse(
            request_id=request_id,
            advice=[],
            degraded=True,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    advices: list[CdssAdvice] = []
    for raw in outcome.value.advice:
        quote = (raw.evidence_quote or "").strip()
        evidence: list[CdssEvidence] = []
        verified = False
        if quote:
            best: tuple[Any, Any] | None = None
            for ctx in contexts:
                result = container.verifier.verify(quote, ctx.chunk.text)
                if best is None or result.score > best[1].score:
                    best = (ctx, result)
                if result.state == VERIFIED:
                    break
            if best is not None:
                ctx, result = best
                verified = result.state == VERIFIED
                evidence.append(
                    CdssEvidence(
                        doc_id=ctx.chunk.doc_id,
                        title=ctx.chunk.title,
                        chunk_id=ctx.chunk.id,
                        quote=result.matched or quote,
                        score=result.score,
                        state=VerificationState(result.state),
                        start=result.start,
                        end=result.end,
                    )
                )

        confidence = raw.confidence if verified else raw.confidence * 0.6
        advices.append(
            CdssAdvice(
                advice_type=raw.advice_type,
                content=raw.content[:2000],
                rationale=raw.rationale[:1000],
                evidence=evidence,
                confidence=round(max(0.0, min(1.0, confidence)), 3),
            )
        )

    degraded = outcome.degraded or outcome.value.insufficient_evidence
    try:
        await container.repo.save_cdss_advice(
            request_id=request_id,
            trace_id=trace_id,
            hospital_id=payload.hospital_id,
            advices=advices,
            degraded=degraded,
        )
        await container.audit.record(
            trace_id=trace_id,
            actor=str(auth.get("sub", "system")),
            action="cdss.advice",
            payload={
                "request_id": request_id,
                "advice": len(advices),
                "degraded": degraded,
            },
        )
    except Exception:  # noqa: BLE001
        logger.exception("persist_cdss_failed", request_id=request_id)

    return CdssResponse(
        request_id=request_id,
        advice=advices,
        degraded=degraded,
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )


@router.post("/icd/suggest", response_model=list[IcdSuggestion], summary="ICD-10 编码推荐")
async def icd_suggest(
    payload: IcdRequest,
    container: AppContainer = Depends(get_container),
    _: dict[str, Any] = Depends(require_auth),
) -> list[IcdSuggestion]:
    if container.icd is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="ICD 目录未加载"
        )

    document = build_document(payload.record, payload.raw_text)
    if not document.text.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="病历内容为空")

    if container.settings.pii_masking:
        document = mask_document(document)

    suggestions, _degraded = await container.icd.suggest(
        document, payload.diagnoses, top_k=payload.top_k
    )
    return suggestions


@router.post("/triage", response_model=TriageResponse, summary="急症前置分诊（纯规则）")
async def triage(
    payload: TriageRequest,
    _: dict[str, Any] = Depends(require_auth),
) -> TriageResponse:
    symptoms = payload.symptoms
    emergency = [word for word in _EMERGENCY if word in symptoms]
    if emergency:
        return TriageResponse(
            level="emergency",
            matched=emergency,
            message="存在急危重症风险信号，请立即前往急诊或拨打 120，勿在线上等待。",
        )

    urgent = [word for word in _URGENT if word in symptoms]
    if urgent:
        return TriageResponse(
            level="urgent",
            matched=urgent,
            message="症状需尽快就医评估，建议 24 小时内到相应专科门诊就诊。",
        )

    return TriageResponse(
        level="normal",
        matched=[],
        message="暂未识别到紧急风险信号，建议结合症状变化择期就诊；如加重请及时就医。",
    )

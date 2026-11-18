"""仓储层：质控结果落库、人工复核、采纳率统计、审计链写入。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ..schemas import CdssAdvice, QcRequest, QcResult
from .models import AuditLog, CdssAdviceRecord, IcdSuggestionRecord, QcFinding, QcJob

__all__ = ["QcRepository"]


class QcRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    # ------------------------------------------------------------------ 写入
    async def save_result(
        self,
        result: QcResult,
        request: QcRequest,
        *,
        doc_hash: str = "",
    ) -> None:
        async with self._session_factory() as session:
            job = QcJob(
                id=result.job_id,
                trace_id=result.trace_id,
                hospital_id=request.hospital_id,
                patient_id=request.patient_id,
                encounter_id=request.encounter_id,
                encounter_type=request.encounter_type.value,
                status=result.status,
                score=result.score,
                degraded=result.degraded,
                model=result.model,
                prompt_version=result.prompt_version,
                doc_hash=doc_hash,
                summary=result.summary,
                latency_ms=result.latency_ms,
                ttft_ms=result.ttft_ms,
                created_at=result.created_at,
                finished_at=datetime.now(timezone.utc),
            )
            session.add(job)

            for finding in result.findings:
                evidence = finding.evidence
                session.add(
                    QcFinding(
                        job_id=result.job_id,
                        code=finding.code,
                        title=finding.title,
                        category=finding.category.value,
                        severity=finding.severity.value,
                        section=finding.section,
                        detail=finding.detail,
                        suggestion=finding.suggestion,
                        quote=evidence.quote if evidence else "",
                        start_offset=evidence.start if evidence else 0,
                        end_offset=evidence.end if evidence else 0,
                        evidence_state=evidence.state.value if evidence else "invalid",
                        evidence_score=evidence.score if evidence else 0.0,
                        confidence=finding.confidence,
                        source=finding.source,
                        needs_review=finding.needs_review,
                        review_status=finding.review_status,
                    )
                )

            for suggestion in result.icd:
                session.add(
                    IcdSuggestionRecord(
                        job_id=result.job_id,
                        code=suggestion.code,
                        name=suggestion.name,
                        confidence=suggestion.confidence,
                        rationale=suggestion.rationale,
                        quote=suggestion.evidence.quote if suggestion.evidence else "",
                        verified=suggestion.verified,
                    )
                )

            await session.commit()

    async def save_cdss_advice(
        self,
        *,
        request_id: str,
        trace_id: str,
        hospital_id: str,
        advices: Sequence[CdssAdvice],
        degraded: bool,
    ) -> None:
        if not advices:
            return
        async with self._session_factory() as session:
            for advice in advices:
                session.add(
                    CdssAdviceRecord(
                        request_id=request_id,
                        trace_id=trace_id,
                        hospital_id=hospital_id,
                        advice_type=advice.advice_type,
                        content=advice.content,
                        rationale=advice.rationale,
                        evidence=[item.model_dump(mode="json") for item in advice.evidence],
                        confidence=advice.confidence,
                        degraded=degraded,
                    )
                )
            await session.commit()

    # ------------------------------------------------------------------ 查询
    async def get_job(self, job_id: str) -> QcJob | None:
        async with self._session_factory() as session:
            return await session.get(QcJob, job_id)

    async def get_findings(self, job_id: str) -> list[QcFinding]:
        async with self._session_factory() as session:
            rows = await session.scalars(
                select(QcFinding).where(QcFinding.job_id == job_id).order_by(QcFinding.id)
            )
            return list(rows)

    async def update_review(
        self,
        finding_id: int,
        *,
        status: str,
        reviewer: str,
        comment: str = "",
    ) -> bool:
        async with self._session_factory() as session:
            finding = await session.get(QcFinding, finding_id)
            if finding is None:
                return False
            finding.review_status = status
            finding.reviewer = reviewer
            finding.review_comment = comment
            finding.reviewed_at = datetime.now(timezone.utc)
            await session.commit()
            return True

    async def stats(self, hospital_id: str | None = None) -> dict[str, Any]:
        async with self._session_factory() as session:
            job_filter = [QcJob.hospital_id == hospital_id] if hospital_id else []
            total_jobs = await session.scalar(
                select(func.count()).select_from(QcJob).where(*job_filter)
            )
            total_findings = await session.scalar(
                select(func.count())
                .select_from(QcFinding)
                .join(QcJob, QcFinding.job_id == QcJob.id)
                .where(*job_filter)
            )

            severity_rows = await session.execute(
                select(QcFinding.severity, func.count())
                .join(QcJob, QcFinding.job_id == QcJob.id)
                .where(*job_filter)
                .group_by(QcFinding.severity)
            )
            review_rows = await session.execute(
                select(QcFinding.review_status, func.count())
                .join(QcJob, QcFinding.job_id == QcJob.id)
                .where(*job_filter)
                .group_by(QcFinding.review_status)
            )

        severity = {str(row[0]): int(row[1]) for row in severity_rows}
        review = {str(row[0]): int(row[1]) for row in review_rows}
        decided = review.get("accepted", 0) + review.get("rejected", 0)
        adoption = round(review.get("accepted", 0) / decided, 4) if decided else 0.0
        return {
            "total_jobs": int(total_jobs or 0),
            "total_findings": int(total_findings or 0),
            "severity": severity,
            "review": review,
            "adoption_rate": adoption,
        }

    # ------------------------------------------------------------------ 审计链
    async def last_audit_hash(self) -> str:
        async with self._session_factory() as session:
            value = await session.scalar(
                select(AuditLog.entry_hash).order_by(AuditLog.id.desc()).limit(1)
            )
            return str(value) if value else "GENESIS"

    async def insert_audit(
        self,
        *,
        trace_id: str,
        actor: str,
        action: str,
        payload_hash: str,
        prev_hash: str,
        entry_hash: str,
    ) -> None:
        async with self._session_factory() as session:
            session.add(
                AuditLog(
                    trace_id=trace_id,
                    actor=actor,
                    action=action,
                    payload_hash=payload_hash,
                    prev_hash=prev_hash,
                    entry_hash=entry_hash,
                )
            )
            await session.commit()

    async def verify_audit_chain(self, limit: int = 1000) -> tuple[bool, int]:
        """校验最近 N 条审计记录哈希链是否连续。"""
        async with self._session_factory() as session:
            rows = list(
                await session.scalars(select(AuditLog).order_by(AuditLog.id).limit(limit))
            )
        prev = "GENESIS"
        for row in rows:
            if row.prev_hash != prev:
                return False, row.id
            prev = row.entry_hash
        return True, rows[-1].id if rows else 0

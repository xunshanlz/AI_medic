"""SQLAlchemy 2.0 数据模型。

表设计围绕"一次质控 = 一个 Job + N 条 Finding + N 条 ICD 建议"展开，
并独立保存审计链。字段命名注意规避 MySQL 保留字（start/end 落库为
``start_offset`` / ``end_offset``，属性名保持语义清晰）。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

__all__ = ["Base", "QcJob", "QcFinding", "IcdSuggestionRecord", "CdssAdviceRecord", "AuditLog"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class QcJob(Base):
    __tablename__ = "qc_job"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    trace_id: Mapped[str] = mapped_column(String(32), index=True)
    hospital_id: Mapped[str] = mapped_column(String(64), index=True, default="")
    patient_id: Mapped[str] = mapped_column(String(64), default="")
    encounter_id: Mapped[str] = mapped_column(String(64), default="")
    encounter_type: Mapped[str] = mapped_column(String(20), default="inpatient")
    status: Mapped[str] = mapped_column(String(20), default="succeeded")
    score: Mapped[float] = mapped_column(Float, default=0.0)
    degraded: Mapped[bool] = mapped_column(Boolean, default=False)
    model: Mapped[str] = mapped_column(String(120), default="")
    prompt_version: Mapped[str] = mapped_column(String(32), default="")
    doc_hash: Mapped[str] = mapped_column(String(32), default="", index=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    ttft_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    findings: Mapped[list["QcFinding"]] = relationship(
        back_populates="job", cascade="all, delete-orphan", lazy="selectin"
    )
    icd_suggestions: Mapped[list["IcdSuggestionRecord"]] = relationship(
        back_populates="job", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (Index("ix_qc_job_hospital_created", "hospital_id", "created_at"),)


class QcFinding(Base):
    __tablename__ = "qc_finding"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("qc_job.id", ondelete="CASCADE"), index=True)
    code: Mapped[str] = mapped_column(String(32), index=True)
    title: Mapped[str] = mapped_column(String(120), default="")
    category: Mapped[str] = mapped_column(String(32), default="logic")
    severity: Mapped[str] = mapped_column(String(16), index=True, default="medium")
    section: Mapped[str] = mapped_column(String(32), default="")
    detail: Mapped[str] = mapped_column(Text, default="")
    suggestion: Mapped[str] = mapped_column(Text, default="")
    quote: Mapped[str] = mapped_column(Text, default="")
    start_offset: Mapped[int] = mapped_column(Integer, default=0)
    end_offset: Mapped[int] = mapped_column(Integer, default=0)
    evidence_state: Mapped[str] = mapped_column(String(16), default="verified")
    evidence_score: Mapped[float] = mapped_column(Float, default=100.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    source: Mapped[str] = mapped_column(String(16), default="llm")
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)
    review_status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    reviewer: Mapped[str] = mapped_column(String(64), default="")
    review_comment: Mapped[str] = mapped_column(String(500), default="")
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    job: Mapped[QcJob] = relationship(back_populates="findings")


class IcdSuggestionRecord(Base):
    __tablename__ = "icd_suggestion"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("qc_job.id", ondelete="CASCADE"), index=True)
    code: Mapped[str] = mapped_column(String(16), index=True)
    name: Mapped[str] = mapped_column(String(200), default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    rationale: Mapped[str] = mapped_column(Text, default="")
    quote: Mapped[str] = mapped_column(Text, default="")
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    job: Mapped[QcJob] = relationship(back_populates="icd_suggestions")


class CdssAdviceRecord(Base):
    __tablename__ = "cdss_advice"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(36), index=True)
    trace_id: Mapped[str] = mapped_column(String(32), default="")
    hospital_id: Mapped[str] = mapped_column(String(64), default="")
    advice_type: Mapped[str] = mapped_column(String(20), default="exam")
    content: Mapped[str] = mapped_column(Text, default="")
    rationale: Mapped[str] = mapped_column(Text, default="")
    evidence: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    degraded: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String(32), index=True, default="")
    actor: Mapped[str] = mapped_column(String(64), default="system")
    action: Mapped[str] = mapped_column(String(64), index=True)
    payload_hash: Mapped[str] = mapped_column(String(64), default="")
    prev_hash: Mapped[str] = mapped_column(String(64), default="")
    entry_hash: Mapped[str] = mapped_column(String(64), index=True, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

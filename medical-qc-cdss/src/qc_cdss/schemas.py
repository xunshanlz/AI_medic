"""对外契约（Pydantic v2）。

所有出入参都在此定义，路由层不做数据加工，避免"接口形状"散落在各处。
契约设计原则：
- 可选字段有默认值，便于前端渐进式接入；
- 证据区间用 ``start/end``（左闭右开）表达，前端可直接 ``text.slice``；
- 结果里带 ``model`` / ``prompt_version`` / ``degraded``，便于线上排障与灰度归因。
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "Severity",
    "FindingCategory",
    "VerificationState",
    "EncounterType",
    "EvidenceSpan",
    "Finding",
    "IcdSuggestion",
    "QcOptions",
    "QcRequest",
    "StageTiming",
    "QcResult",
    "JobDetail",
    "ReviewRequest",
    "CdssRequest",
    "CdssEvidence",
    "CdssAdvice",
    "CdssResponse",
    "IcdRequest",
    "TriageRequest",
    "TriageResponse",
    "StatsResponse",
]


class Severity(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class FindingCategory(str, Enum):
    completeness = "completeness"
    consistency = "consistency"
    logic = "logic"
    timeliness = "timeliness"
    safety = "safety"
    dosage = "dosage"
    coding = "coding"


class VerificationState(str, Enum):
    verified = "verified"
    pending = "pending"
    invalid = "invalid"


class EncounterType(str, Enum):
    inpatient = "inpatient"
    outpatient = "outpatient"
    emergency = "emergency"


class EvidenceSpan(BaseModel):
    """原文证据片段。``start``/``end`` 为左闭右开区间。"""

    model_config = ConfigDict(extra="ignore")

    start: int = Field(ge=0)
    end: int = Field(ge=0)
    quote: str = ""
    state: VerificationState = VerificationState.verified
    score: float = Field(default=100.0, ge=0, le=100)
    strategy: str = "exact"
    section: str | None = None

    @model_validator(mode="after")
    def _check_range(self) -> "EvidenceSpan":
        if self.end < self.start:
            raise ValueError("evidence.end 不能小于 start")
        return self


class Finding(BaseModel):
    model_config = ConfigDict(extra="ignore")

    finding_id: str | None = None
    code: str
    title: str
    category: FindingCategory
    severity: Severity
    section: str = ""
    detail: str = ""
    suggestion: str = ""
    evidence: EvidenceSpan | None = None
    confidence: float = Field(default=0.5, ge=0, le=1)
    source: Literal["rule", "llm"] = "llm"
    needs_review: bool = False
    review_status: Literal["pending", "accepted", "rejected"] = "pending"


class IcdSuggestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    code: str
    name: str
    confidence: float = Field(default=0.0, ge=0, le=1)
    rationale: str = ""
    evidence: EvidenceSpan | None = None
    verified: bool = False


class QcOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")

    checklist: list[str] | None = None
    enable_icd: bool = True
    enable_cdss: bool = True
    max_findings: int = Field(default=50, ge=1, le=200)
    strict_evidence: bool = True  # True: 未通过校验的 LLM 结论降级为待复核


class QcRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", json_schema_extra={
        "example": {
            "hospital_id": "demo-hospital",
            "patient_id": "P2026001",
            "encounter_id": "E2026001",
            "encounter_type": "inpatient",
            "sex": "male",
            "age": 56,
            "record": {
                "chief_complaint": "发热3天",
                "present_illness": "3天前受凉后出现发热，最高38.5℃，伴咳嗽咳痰。",
                "diagnosis": "急性上呼吸道感染",
            },
        }
    })

    hospital_id: str = Field(default="demo-hospital", max_length=64)
    patient_id: str = Field(min_length=1, max_length=64, description="脱敏后的患者标识")
    encounter_id: str = Field(min_length=1, max_length=64)
    encounter_type: EncounterType = EncounterType.inpatient
    sex: Literal["male", "female", "unknown"] = "unknown"
    age: int | None = Field(default=None, ge=0, le=150)
    record: dict[str, Any] | None = None
    raw_text: str | None = None
    admit_time: datetime | None = None
    record_time: datetime | None = None
    options: QcOptions = Field(default_factory=QcOptions)

    @model_validator(mode="after")
    def _require_content(self) -> "QcRequest":
        if not self.record and not (self.raw_text and self.raw_text.strip()):
            raise ValueError("record 与 raw_text 至少提供一个")
        return self

    @field_validator("raw_text")
    @classmethod
    def _limit_raw_text(cls, value: str | None) -> str | None:
        if value and len(value) > 200_000:
            raise ValueError("raw_text 超过 20 万字符，请先拆分病历")
        return value


class StageTiming(BaseModel):
    stage: str
    ms: float = Field(ge=0)


class QcResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    job_id: str
    trace_id: str
    status: Literal["succeeded", "degraded", "failed"] = "succeeded"
    score: float = Field(ge=0, le=100)
    findings: list[Finding] = Field(default_factory=list)
    icd: list[IcdSuggestion] = Field(default_factory=list)
    summary: str = ""
    degraded: bool = False
    degraded_reasons: list[str] = Field(default_factory=list)
    model: str = ""
    prompt_version: str = ""
    timings: list[StageTiming] = Field(default_factory=list)
    latency_ms: float = 0.0
    ttft_ms: float | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class JobDetail(BaseModel):
    result: QcResult
    findings: list[Finding] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)


class ReviewRequest(BaseModel):
    status: Literal["accepted", "rejected"]
    reviewer: str = Field(default="unknown", max_length=64)
    comment: str = Field(default="", max_length=500)


class CdssRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hospital_id: str = "demo-hospital"
    patient_id: str = "anonymous"
    encounter_id: str = "anonymous"
    question: str = Field(min_length=2, max_length=2000)
    diagnoses: list[str] = Field(default_factory=list)
    record_excerpt: str | None = Field(default=None, max_length=20_000)
    sex: Literal["male", "female", "unknown"] = "unknown"
    age: int | None = Field(default=None, ge=0, le=150)
    top_k: int = Field(default=5, ge=1, le=20)


class CdssEvidence(BaseModel):
    doc_id: str
    title: str = ""
    chunk_id: str = ""
    quote: str = ""
    score: float = 0.0
    state: VerificationState = VerificationState.pending
    start: int | None = None
    end: int | None = None


class CdssAdvice(BaseModel):
    advice_type: Literal["exam", "medication", "follow_up", "warning", "education"]
    content: str
    rationale: str = ""
    evidence: list[CdssEvidence] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0, le=1)


class CdssResponse(BaseModel):
    request_id: str
    advice: list[CdssAdvice] = Field(default_factory=list)
    disclaimer: str = "本建议由 AI 基于知识库生成，仅供临床参考，最终决策由医师作出。"
    degraded: bool = False
    latency_ms: float = 0.0


class IcdRequest(BaseModel):
    """独立 ICD 编码推荐请求（不触发全量质控）。"""

    model_config = ConfigDict(extra="ignore")

    hospital_id: str = "demo-hospital"
    diagnoses: list[str] = Field(default_factory=list)
    record: dict[str, Any] | None = None
    raw_text: str | None = None
    top_k: int = Field(default=3, ge=1, le=10)

    @model_validator(mode="after")
    def _require_content(self) -> "IcdRequest":
        if not self.record and not (self.raw_text and self.raw_text.strip()):
            raise ValueError("record 与 raw_text 至少提供一个")
        return self


class TriageRequest(BaseModel):
    hospital_id: str = "demo-hospital"
    symptoms: str = Field(min_length=1, max_length=2000)
    age: int | None = None
    sex: Literal["male", "female", "unknown"] = "unknown"


class TriageResponse(BaseModel):
    level: Literal["emergency", "urgent", "normal"]
    matched: list[str] = Field(default_factory=list)
    message: str


class StatsResponse(BaseModel):
    total_jobs: int = 0
    total_findings: int = 0
    severity: dict[str, int] = Field(default_factory=dict)
    review: dict[str, int] = Field(default_factory=dict)
    adoption_rate: float = 0.0

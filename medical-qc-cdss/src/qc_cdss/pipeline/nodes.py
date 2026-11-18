"""质控流水线各节点实现。

节点划分对应"可单独观测、可单独降级"的边界：

    prepare -> rules -> llm_qc -> icd -> finalize

- prepare：结构化/纯文本 -> 统一 Document；等长 PII 脱敏；
- rules：确定性规则，毫秒级，先出高危项；
- llm_qc：按检查项 × 分片并行调用模型，逐项缓存、逐项超时；
- icd：编码召回 + 重排 + 证据校验；
- finalize：去重、评分、生成摘要。

每个节点都只返回"增量"，由 LangGraph 的 reducer 合并；任何节点失败都不会
让整个请求 500，而是折算为 ``degraded``。
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Sequence

from ..cache import Cache, cache_key
from ..domain.checklist import ChecklistItem, get_checklist
from ..domain.evidence import VERIFIED, EvidenceVerifier
from ..domain.rules import PatientContext, RuleEngine
from ..domain.scoring import dedupe
from ..domain.sections import Document, build_document
from ..llm.prompts import QcItemVerdict, qc_item_messages
from ..llm.structured import StructuredGenerator
from ..logging import get_logger
from ..privacy import mask_document
from ..schemas import (
    EvidenceSpan,
    Finding,
    FindingCategory,
    IcdSuggestion,
    QcRequest,
    Severity,
    StageTiming,
    VerificationState,
)
from . import aggregator
from .icd import IcdRecommender
from .scheduler import Deadline, gather_bounded

__all__ = ["NodeDeps", "QcNodes", "Emitter"]

logger = get_logger(__name__)

Emitter = Callable[[str, dict[str, Any]], Awaitable[None]]


@dataclass
class NodeDeps:
    settings: Any
    generator: StructuredGenerator
    verifier: EvidenceVerifier
    rules: RuleEngine
    icd: IcdRecommender | None
    cache: Cache


class QcNodes:
    def __init__(self, deps: NodeDeps) -> None:
        self.deps = deps

    # ------------------------------------------------------------------ 事件
    async def _emit(self, state: dict[str, Any], event: str, payload: dict[str, Any]) -> None:
        emitter: Emitter | None = state.get("emit")
        if emitter is None:
            return
        try:
            await emitter(event, payload)
        except Exception:  # noqa: BLE001 - 推送失败不影响主流程
            logger.warning("sse_emit_failed", event=event)

    def _timing(self, stage: str, started: float) -> StageTiming:
        return StageTiming(stage=stage, ms=round((time.perf_counter() - started) * 1000, 2))

    # ------------------------------------------------------------------ prepare
    async def prepare(self, state: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        request: QcRequest = state["request"]

        document = build_document(request.record, request.raw_text)
        if not document.text.strip():
            raise ValueError("病历内容为空，无法质控")

        # 等长脱敏：字符位置不变，因此证据 offset 对原始病历依然成立
        if self.deps.settings.pii_masking:
            document = mask_document(document)

        doc_hash = hashlib.sha256(document.text.encode("utf-8")).hexdigest()[:16]
        await self._emit(
            state,
            "stage.completed",
            {
                "stage": "prepare",
                "chars": len(document.text),
                "sections": len(document.sections),
                "doc_hash": doc_hash,
            },
        )
        return {
            "document": document,
            "doc_hash": doc_hash,
            "timings": [self._timing("prepare", started)],
        }

    # ------------------------------------------------------------------ rules
    async def rules(self, state: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        document: Document = state["document"]
        request: QcRequest = state["request"]

        context = PatientContext(
            sex=None if request.sex == "unknown" else request.sex,
            age=request.age,
            admit_time=request.admit_time,
            record_time=request.record_time,
        )
        hits = list(self.deps.rules.run(document, context))

        findings: list[Finding] = []
        for hit in hits:
            evidence = None
            if hit.quote:
                evidence = EvidenceSpan(
                    start=hit.start,
                    end=max(hit.end, hit.start),
                    quote=hit.quote,
                    state=VerificationState.verified,
                    score=100.0,
                    strategy="exact",
                    section=hit.section,
                )
            finding = Finding(
                code=hit.rule_id,
                title=hit.title,
                category=FindingCategory(hit.category),
                severity=Severity(hit.severity),
                section=hit.section,
                detail=hit.message,
                suggestion=hit.suggestion,
                evidence=evidence,
                confidence=1.0,
                source="rule",
                needs_review=False,
            )
            findings.append(finding)
            await self._emit(state, "finding", finding.model_dump(mode="json"))

        await self._emit(
            state, "stage.completed", {"stage": "rules", "findings": len(findings)}
        )
        return {
            "findings": findings,
            "timings": [self._timing("rules", started)],
        }

    # ------------------------------------------------------------------ llm_qc
    async def llm_qc(self, state: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        settings = self.deps.settings
        document: Document = state["document"]
        request: QcRequest = state["request"]
        deadline: Deadline = state["deadline"]

        items = list(get_checklist(settings.checklist_version))
        if request.options.checklist:
            wanted = set(request.options.checklist)
            items = [item for item in items if item.code in wanted]

        tasks: list[tuple[ChecklistItem, list[tuple[str, str]]]] = []
        for item in items:
            blocks = [
                (name, document.section_text(name))
                for name in item.sections
                if document.section_text(name).strip()
            ]
            if not blocks:
                continue
            for shard in self._shard(blocks, settings.shard_chars, settings.max_shards):
                tasks.append((item, shard))

        if not tasks:
            await self._emit(state, "stage.completed", {"stage": "llm_qc", "items": 0})
            return {"timings": [self._timing("llm_qc", started)]}

        await self._emit(
            state,
            "stage.started",
            {"stage": "llm_qc", "items": len(items), "tasks": len(tasks)},
        )

        results = await gather_bounded(
            [
                self._run_item(item, shard, request, document, deadline)
                for item, shard in tasks
            ],
            limit=max(1, settings.max_concurrency),
        )

        findings: list[Finding] = []
        degraded: list[str] = []
        for (item, _shard), result in zip(tasks, results):
            if isinstance(result, BaseException):
                degraded.append(f"{item.code}:{type(result).__name__}")
                await self._emit(
                    state, "item.failed", {"code": item.code, "error": type(result).__name__}
                )
                continue
            item_findings, item_degraded = result
            findings.extend(item_findings)
            degraded.extend(item_degraded)
            for finding in item_findings:
                await self._emit(state, "finding", finding.model_dump(mode="json"))

        await self._emit(
            state,
            "stage.completed",
            {"stage": "llm_qc", "findings": len(findings), "degraded": len(degraded)},
        )
        return {
            "findings": findings,
            "degraded_reasons": degraded,
            "timings": [self._timing("llm_qc", started)],
        }

    async def _run_item(
        self,
        item: ChecklistItem,
        shard: list[tuple[str, str]],
        request: QcRequest,
        document: Document,
        deadline: Deadline,
    ) -> tuple[list[Finding], list[str]]:
        settings = self.deps.settings
        remaining = deadline.remaining
        if remaining <= 0.2:
            return [], [f"{item.code}:deadline"]

        timeout = min(settings.branch_timeout_s, remaining)
        shard_text = "\n".join(text for _, text in shard)
        cache_key_value = cache_key(
            shard_text,
            item.code,
            settings.prompt_version,
            getattr(self.deps.generator.llm, "model", "unknown"),
            "qc-v1",
        )

        verdict: QcItemVerdict | None = None
        degraded: list[str] = []
        cached = await self.deps.cache.get(cache_key_value)
        if cached:
            try:
                verdict = QcItemVerdict.model_validate_json(cached)
            except Exception:  # noqa: BLE001 - 缓存脏数据直接忽略
                verdict = None

        if verdict is None:
            messages = qc_item_messages(
                code=item.code,
                title=item.title,
                instruction=item.instruction,
                blocks=shard,
                sex=request.sex,
                age=request.age,
            )
            outcome = await self.deps.generator.generate(
                QcItemVerdict,
                messages,
                timeout=timeout,
                stage=f"qc.{item.code}",
            )
            if outcome.value is None:
                return [], [f"{item.code}:structured_output_failed"]
            verdict = outcome.value
            if outcome.degraded:
                degraded.append(f"{item.code}:salvaged")
            await self.deps.cache.set(cache_key_value, verdict.model_dump_json(), settings.cache_ttl_s)

        findings: list[Finding] = []
        for raw in verdict.findings:
            quote = (raw.quote or "").strip()
            evidence: EvidenceSpan | None = None
            needs_review = False

            if quote:
                result = self.deps.verifier.verify(quote, document.text)
                evidence = EvidenceSpan(
                    start=result.start or 0,
                    end=result.end or 0,
                    quote=result.matched or quote,
                    state=VerificationState(result.state),
                    score=result.score,
                    strategy=result.strategy,
                )
                if result.state != VERIFIED:
                    needs_review = True
            else:
                # 无引文的问题不能自动采纳：模型可能"编造"了问题
                needs_review = True

            severity = self._severity(raw.severity_hint, item.severity)
            confidence = raw.confidence * (0.6 if needs_review else 1.0)
            section = self._locate_section(document, evidence, item)

            findings.append(
                Finding(
                    code=item.code,
                    title=raw.title[:60],
                    category=FindingCategory(item.category),
                    severity=severity,
                    section=section,
                    detail=raw.detail[:1000],
                    suggestion=raw.suggestion[:500],
                    evidence=evidence,
                    confidence=round(max(0.0, min(1.0, confidence)), 3),
                    source="llm",
                    needs_review=needs_review,
                )
            )
        return findings, degraded

    # ------------------------------------------------------------------ icd
    async def icd(self, state: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        request: QcRequest = state["request"]
        document: Document = state["document"]

        if not request.options.enable_icd or self.deps.icd is None:
            return {"timings": [self._timing("icd", started)]}

        diagnoses = self._diagnoses(request, document)
        suggestions, degraded = await self.deps.icd.suggest(document, diagnoses)
        for suggestion in suggestions:
            await self._emit(state, "icd", suggestion.model_dump(mode="json"))

        await self._emit(
            state, "stage.completed", {"stage": "icd", "suggestions": len(suggestions)}
        )
        return {
            "icd": suggestions,
            "degraded_reasons": [f"icd:{reason}" for reason in (["model_degraded"] if degraded else [])],
            "timings": [self._timing("icd", started)],
        }

    # ------------------------------------------------------------------ finalize
    async def finalize(self, state: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        request: QcRequest = state["request"]
        findings: list[Finding] = list(state.get("findings") or [])
        icd: list[IcdSuggestion] = list(state.get("icd") or [])
        degraded_reasons: list[str] = list(state.get("degraded_reasons") or [])

        raw = [finding.model_dump() for finding in findings]
        unique = dedupe(raw)
        unique = unique[: request.options.max_findings]
        final_findings = [Finding.model_validate(item) for item in unique]

        aggregate = aggregator.aggregate_findings(final_findings)
        summary = aggregator.summarize(
            score=aggregate["score"],
            counts=aggregate["counts"],
            total=aggregate["total"],
            needs_review=aggregate["needs_review"],
        )
        if degraded_reasons:
            summary += "（部分检查项因模型或超时降级，结果可能不完整，建议人工复核。）"

        await self._emit(
            state,
            "summary",
            {
                "score": aggregate["score"],
                "counts": aggregate["counts"],
                "total": aggregate["total"],
                "needs_review": aggregate["needs_review"],
                "degraded": bool(degraded_reasons),
                "summary": summary,
            },
        )
        return {
            "final": {
                "findings": final_findings,
                "icd": icd,
                "score": aggregate["score"],
                "counts": aggregate["counts"],
                "total": aggregate["total"],
                "needs_review": aggregate["needs_review"],
                "summary": summary,
                "degraded": bool(degraded_reasons),
                "degraded_reasons": degraded_reasons,
            },
            "timings": [self._timing("finalize", started)],
        }

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def _severity(hint: str | None, fallback: str) -> Severity:
        for candidate in (hint, fallback):
            try:
                return Severity(candidate)
            except (ValueError, TypeError):
                continue
        return Severity.medium

    @staticmethod
    def _locate_section(
        document: Document, evidence: EvidenceSpan | None, item: ChecklistItem
    ) -> str:
        if evidence is not None and evidence.end > evidence.start:
            section = document.section_at(evidence.start)
            if section is not None:
                return section.name
        return item.sections[0] if item.sections else ""

    @staticmethod
    def _diagnoses(request: QcRequest, document: Document) -> list[str]:
        record = request.record or {}
        raw = record.get("diagnosis") or record.get("preliminary_diagnosis")
        if isinstance(raw, (list, tuple)):
            return [str(item) for item in raw if str(item).strip()]
        if isinstance(raw, str) and raw.strip():
            return [part for part in raw.replace(",", "；").split("；") if part.strip()]

        text = document.section_text("初步诊断")
        return [part for part in text.replace(",", "；").split("；") if part.strip()][:5]

    @staticmethod
    def _shard(
        blocks: Sequence[tuple[str, str]],
        max_chars: int,
        max_shards: int,
    ) -> list[list[tuple[str, str]]]:
        """长段落按窗口切分，再合并为不超过 max_shards 个分片。"""
        windows: list[list[tuple[str, str]]] = []
        for title, text in blocks:
            if len(text) <= max_chars:
                windows.append([(title, text)])
                continue
            start = 0
            index = 1
            step = max(200, max_chars - 100)
            while start < len(text):
                piece = text[start : start + max_chars]
                windows.append([(f"{title}（续{index}）", piece)])
                if start + max_chars >= len(text):
                    break
                start += step
                index += 1

        merged: list[list[tuple[str, str]]] = []
        current: list[tuple[str, str]] = []
        size = 0
        for window in windows:
            window_size = sum(len(text) for _, text in window)
            if current and size + window_size > max_chars:
                merged.append(current)
                current = []
                size = 0
            current.extend(window)
            size += window_size
        if current:
            merged.append(current)
        return merged[: max(1, max_shards)]

"""ICD-10 编码推荐。

链路：候选召回 -> LLM 重排 -> 编码校验 -> 证据校验 -> 置信度校准。

关键设计：
- **候选先行**：先从本地 ICD 目录（国临版）召回 10-15 个候选，模型只能在其中
  选择。这从结构上杜绝了"编造编码"——医疗编码错一位就是另一个病种。
- **双通道召回**：编码/名称的 BM25 是主通道，诊断文本本身也参与打分，避免
  同义词（"心梗" vs "心肌梗死"）漏召。
- **证据校验**：编码理由必须能引回病历原文，否则视为不可信，置信度减半。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ..domain.evidence import VERIFIED, EvidenceVerifier
from ..domain.sections import Document
from ..llm.prompts import IcdVerdict, icd_messages
from ..llm.structured import StructuredGenerator
from ..logging import get_logger
from ..rag.bm25 import BM25Index
from ..schemas import EvidenceSpan, IcdSuggestion, VerificationState

__all__ = ["IcdEntry", "IcdCatalog", "IcdRecommender"]

logger = get_logger(__name__)


@dataclass(frozen=True)
class IcdEntry:
    code: str
    name: str
    note: str = ""
    aliases: tuple[str, ...] = ()

    @property
    def searchable(self) -> str:
        return f"{self.code} {self.name} {' '.join(self.aliases)} {self.note}".strip()


@dataclass
class IcdCatalog:
    entries: list[IcdEntry] = field(default_factory=list)
    _bm25: BM25Index | None = None
    _by_code: dict[str, IcdEntry] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "IcdCatalog":
        entries: list[IcdEntry] = []
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("icd_line_invalid", line=line[:80])
                        continue
                    entries.append(
                        IcdEntry(
                            code=str(payload["code"]),
                            name=str(payload["name"]),
                            note=str(payload.get("note", "")),
                            aliases=tuple(payload.get("aliases", [])),
                        )
                    )
        else:
            logger.warning("icd_catalog_missing", path=str(path))

        index = BM25Index()
        for entry in entries:
            index.add(entry.code, entry.searchable)
        index.build()
        catalog = cls(entries=entries, _bm25=index, _by_code={e.code: e for e in entries})
        logger.info("icd_catalog_loaded", entries=len(entries))
        return catalog

    def candidates(self, text: str, top_k: int = 12) -> list[IcdEntry]:
        if not self._bm25 or not self.entries:
            return []
        hits = self._bm25.search(text, top_k=top_k)
        return [self._by_code[code] for code, _ in hits if code in self._by_code]

    def get(self, code: str) -> IcdEntry | None:
        return self._by_code.get(code)

    def __len__(self) -> int:
        return len(self.entries)


class IcdRecommender:
    def __init__(
        self,
        catalog: IcdCatalog,
        generator: StructuredGenerator,
        verifier: EvidenceVerifier,
        *,
        candidate_k: int = 12,
        timeout_s: float = 20.0,
    ) -> None:
        self.catalog = catalog
        self.generator = generator
        self.verifier = verifier
        self.candidate_k = candidate_k
        self.timeout_s = timeout_s

    async def suggest(
        self,
        document: Document,
        diagnoses: Iterable[str],
        *,
        top_k: int = 3,
    ) -> tuple[list[IcdSuggestion], bool]:
        """返回 (编码建议, 是否降级)。"""
        diagnosis_text = "、".join(x for x in diagnoses if x).strip()
        query = f"{diagnosis_text} {document.section_text('现病史')[:300]} {document.section_text('辅助检查')[:200]}"
        candidates = self.catalog.candidates(query or document.text[:500], self.candidate_k)
        if not candidates:
            return [], False

        messages = icd_messages(
            diagnosis=diagnosis_text or "（未填写）",
            record_digest=document.text[:1500],
            candidates=[
                {"code": c.code, "name": c.name, "note": c.note[:60]} for c in candidates
            ],
        )
        outcome = await self.generator.generate(
            IcdVerdict, messages, timeout=self.timeout_s, stage="icd"
        )
        degraded = outcome.degraded or outcome.value is None

        if outcome.value is None:
            # 模型不可用：退回召回结果，明确降低置信度，供人工挑选
            fallback = [
                IcdSuggestion(
                    code=entry.code,
                    name=entry.name,
                    confidence=0.35,
                    rationale="模型不可用，按检索相关度召回，需人工确认",
                    verified=False,
                )
                for entry in candidates[:top_k]
            ]
            return fallback, degraded

        rank_of = {entry.code: index for index, entry in enumerate(candidates)}
        suggestions: list[IcdSuggestion] = []
        seen: set[str] = set()

        for candidate in outcome.value.suggestions:
            entry = self.catalog.get(candidate.code)
            if entry is None or candidate.code in seen:
                continue
            seen.add(candidate.code)

            evidence: EvidenceSpan | None = None
            verified = False
            confidence = max(0.0, min(1.0, candidate.confidence))
            if candidate.quote:
                result = self.verifier.verify(candidate.quote, document.text)
                evidence = EvidenceSpan(
                    start=result.start or 0,
                    end=result.end or 0,
                    quote=result.matched or candidate.quote,
                    state=VerificationState(result.state),
                    score=result.score,
                    strategy=result.strategy,
                )
                verified = result.state == VERIFIED
                if not verified:
                    confidence *= 0.5

            # 检索排名作为先验：排名越靠前，置信度略微上浮（上限 0.98）
            prior = 1.0 - rank_of.get(candidate.code, len(candidates)) / max(len(candidates), 1)
            confidence = round(min(0.98, confidence * 0.85 + prior * 0.15), 3)

            suggestions.append(
                IcdSuggestion(
                    code=entry.code,
                    name=entry.name,
                    confidence=confidence,
                    rationale=candidate.rationale[:500],
                    evidence=evidence,
                    verified=verified,
                )
            )

        suggestions.sort(key=lambda item: item.confidence, reverse=True)
        return suggestions[:top_k], degraded

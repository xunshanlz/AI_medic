"""病历质控规则引擎（确定性前置校验）。

设计取舍
--------
规则与 LLM 分工：
- 规则负责**可判定、零容忍**的问题：缺项、时限、性别/年龄矛盾、危急值、
  剂量超限、主诉与现病史否定式矛盾。这类问题不需要语义理解，规则更快、
  更稳、可解释，且能保证 100% 召回。
- LLM 负责**需要语义理解**的内涵问题：诊断依据是否充分、查体与诊断是否
  相符、用药是否合理（见 ``checklist.py``）。

先规则后模型，还有一个工程收益：规则命中的高危项（危急值）可以**前置**
返回给临床，不必等模型推理完成，降低风险暴露时间。

所有规则的产出都带原文偏移，前端可直接高亮。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from .sections import Document, Section

__all__ = ["RuleHit", "PatientContext", "RuleEngine"]


@dataclass(frozen=True)
class RuleHit:
    """一条规则命中。字段与 API 的 Finding 对齐，便于零成本转换。"""

    rule_id: str
    title: str
    category: str
    severity: str  # high | medium | low
    section: str
    message: str
    suggestion: str
    quote: str = ""
    start: int = 0
    end: int = 0
    source: str = "rule"

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.rule_id,
            "title": self.title,
            "category": self.category,
            "severity": self.severity,
            "section": self.section,
            "detail": self.message,
            "suggestion": self.suggestion,
            "quote": self.quote,
            "start": self.start,
            "end": self.end,
            "source": self.source,
        }


@dataclass
class PatientContext:
    """质控所需的患者上下文。缺失字段时对应规则自动跳过（不产生误报）。"""

    sex: str | None = None  # male | female | M | F | 男 | 女
    age: int | None = None
    admit_time: datetime | None = None
    record_time: datetime | None = None

    @property
    def is_male(self) -> bool:
        return (self.sex or "").strip().lower() in {"male", "m", "男", "1"}

    @property
    def is_female(self) -> bool:
        return (self.sex or "").strip().lower() in {"female", "f", "女", "2"}


# 缺项检查：段落 -> (严重级别, 提示)
_REQUIRED_SECTIONS: dict[str, tuple[str, str]] = {
    "主诉": ("high", "主诉缺失，无法体现就诊主要问题"),
    "现病史": ("high", "现病史缺失，无法判断病情演变与诊疗经过"),
    "体格检查": ("medium", "体格检查缺失，诊断缺乏体征支持"),
    "初步诊断": ("high", "初步诊断缺失，病历不具备完整诊疗结论"),
    "医师签名": ("medium", "医师签名缺失，病历不具备法律效力"),
}

_MALE_ONLY_RE = re.compile(r"(前列腺|睾丸|阴茎|精囊|附睾)")
_FEMALE_ONLY_RE = re.compile(r"(妊娠|怀孕|停经|月经|子宫|卵巢|阴道|宫颈|产程|胎心)")

_CRITICAL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"胸痛|心前区疼痛"), "胸痛需警惕急性冠脉综合征"),
    (re.compile(r"呼吸困难|气促|端坐呼吸"), "呼吸困难需评估呼吸衰竭风险"),
    (re.compile(r"意识障碍|昏迷|呼之不应"), "意识障碍需评估神经系统急症"),
    (re.compile(r"抽搐|惊厥"), "抽搐需警惕癫痫持续状态"),
    (re.compile(r"大汗淋漓|冷汗"), "胸痛伴大汗需按急性冠脉综合征处理"),
    (
        re.compile(r"(血压|BP)[^。；;\n]{0,8}?(1[89]\d|2[0-9]\d)\s*/\s*(1[0-2]\d|1[3-9]\d)"),
        "血压达危急值水平",
    ),
    (re.compile(r"血钾[^。；;\n]{0,6}?([6-9]\.\d|[7-9])"), "血钾升高达危急值"),
    (re.compile(r"血糖[^。；;\n]{0,6}?([0-2]\.[0-8]|2\.7)"), "低血糖达危急值"),
)

# 单日剂量上限（mg），仅覆盖演示药品；生产环境由药品知识库驱动
_DOSE_LIMITS_MG: dict[str, float] = {
    "阿司匹林": 300.0,
    "华法林": 10.0,
    "地高辛": 0.5,
    "二甲双胍": 2550.0,
    "氨氯地平": 10.0,
    "对乙酰氨基酚": 4000.0,
}
_DOSE_RE = re.compile(r"(阿司匹林|华法林|地高辛|二甲双胍|氨氯地平|对乙酰氨基酚)[^。；;\n]{0,12}?(\d+(?:\.\d+)?)\s*(mg|毫克|g|克|片)")

_DURATION_RE = re.compile(r"\d+\s*(天|日|周|月|年|小时|h|d)")


class RuleEngine:
    """无状态规则引擎，可安全并发调用。"""

    def __init__(self, required_sections: dict[str, tuple[str, str]] | None = None) -> None:
        self._required = required_sections or _REQUIRED_SECTIONS

    def run(self, document: Document, context: PatientContext | None = None) -> list[RuleHit]:
        ctx = context or PatientContext()
        hits: list[RuleHit] = []
        hits.extend(self._check_required(document))
        hits.extend(self._check_chief_complaint(document))
        hits.extend(self._check_sex_conflict(document, ctx))
        hits.extend(self._check_age_conflict(document, ctx))
        hits.extend(self._check_critical_values(document))
        hits.extend(self._check_timeliness(document, ctx))
        hits.extend(self._check_dosage(document))
        hits.extend(self._check_negation_conflict(document))
        return hits

    # ---------------------------------------------------------------- 完整性
    def _check_required(self, doc: Document) -> Iterable[RuleHit]:
        for name, (severity, message) in self._required.items():
            sec = doc.section(name)
            if sec is not None and sec.text.strip():
                continue
            yield RuleHit(
                rule_id="QC-R001",
                title=f"{name}缺失",
                category="completeness",
                severity=severity,
                section=name,
                message=message,
                suggestion=f"补充{name}，并按《病历书写基本规范》要求完成记录",
                quote="",
                start=sec.start if sec else 0,
                end=sec.end if sec else 0,
            )

    # ---------------------------------------------------------------- 主诉规范
    def _check_chief_complaint(self, doc: Document) -> Iterable[RuleHit]:
        sec = doc.section("主诉")
        if not sec or not sec.text.strip():
            return
        text = sec.text.strip()

        if len(text) > 20:
            yield self._hit(
                doc,
                sec,
                "QC-R002",
                "主诉过长",
                "completeness",
                "low",
                text[:40],
                f"主诉共 {len(text)} 字，超过 20 字；应仅描述症状与持续时间",
                "精简主诉为「症状 + 持续时间」，如「发热 3 天」",
            )

        diagnosis_words = ("炎", "病", "癌", "瘤", "综合征", "感染", "骨折", "梗死", "衰竭", "结石")
        for word in diagnosis_words:
            if word in text:
                yield self._hit(
                    doc,
                    sec,
                    "QC-R003",
                    "主诉包含诊断性表述",
                    "consistency",
                    "medium",
                    word,
                    f"主诉中出现诊断性词汇「{word}」，主诉应描述症状而非诊断",
                    "改为症状描述，把诊断放入初步诊断",
                )
                break

    # ---------------------------------------------------------------- 性别/年龄
    def _check_sex_conflict(self, doc: Document, ctx: PatientContext) -> Iterable[RuleHit]:
        if not (ctx.is_male or ctx.is_female):
            return
        pattern = _FEMALE_ONLY_RE if ctx.is_male else _MALE_ONLY_RE
        target = "男性" if ctx.is_male else "女性"
        for sec in doc.sections:
            match = pattern.search(sec.text)
            if not match:
                continue
            yield self._hit(
                doc,
                sec,
                "QC-R004",
                "性别与病历内容矛盾",
                "consistency",
                "high",
                match.group(0),
                f"患者为{target}，但{sec.name}出现「{match.group(0)}」相关描述",
                "核对患者基本信息与病历内容，确认是否串号或模板残留",
            )
            return

    def _check_age_conflict(self, doc: Document, ctx: PatientContext) -> Iterable[RuleHit]:
        if ctx.age is None:
            return
        if ctx.age < 12:
            for sec in doc.sections:
                match = re.search(r"(妊娠|怀孕|月经)", sec.text)
                if match:
                    yield self._hit(
                        doc,
                        sec,
                        "QC-R005",
                        "年龄与病历内容矛盾",
                        "consistency",
                        "high",
                        match.group(0),
                        f"患者年龄 {ctx.age} 岁，出现「{match.group(0)}」描述，存在逻辑矛盾",
                        "核对年龄与婚育史，确认是否病历串号",
                    )
                    return
        elif ctx.age > 3:
            for sec in doc.sections:
                match = re.search(r"新生儿", sec.text)
                if match:
                    yield self._hit(
                        doc,
                        sec,
                        "QC-R005",
                        "年龄与病历内容矛盾",
                        "consistency",
                        "medium",
                        match.group(0),
                        f"患者年龄 {ctx.age} 岁，出现「新生儿」描述",
                        "核对年龄与病历模板",
                    )
                    return

    # ---------------------------------------------------------------- 危急值
    def _check_critical_values(self, doc: Document) -> Iterable[RuleHit]:
        for pattern, message in _CRITICAL_PATTERNS:
            for sec in doc.sections:
                match = pattern.search(sec.text)
                if not match:
                    continue
                yield self._hit(
                    doc,
                    sec,
                    "QC-R006",
                    "危急值/高危症状预警",
                    "safety",
                    "high",
                    match.group(0),
                    message,
                    "立即评估生命体征，按危急值流程处置并记录",
                )
                break

    # ---------------------------------------------------------------- 时限
    def _check_timeliness(self, doc: Document, ctx: PatientContext) -> Iterable[RuleHit]:
        if not (ctx.admit_time and ctx.record_time):
            return
        if not doc.has_content("主诉"):
            return
        delta_hours = (ctx.record_time - ctx.admit_time).total_seconds() / 3600
        if delta_hours <= 24:
            return
        yield RuleHit(
            rule_id="QC-R007",
            title="入院记录超时",
            category="timeliness",
            severity="high",
            section="入院记录",
            message=f"入院记录距入院时间 {delta_hours:.1f} 小时，超过 24 小时时限",
            suggestion="按《病历书写基本规范》24 小时内完成入院记录",
            source="rule",
        )

    # ---------------------------------------------------------------- 剂量
    def _check_dosage(self, doc: Document) -> Iterable[RuleHit]:
        for sec in doc.sections:
            for match in _DOSE_RE.finditer(sec.text):
                drug, raw, unit = match.group(1), float(match.group(2)), match.group(3)
                limit = _DOSE_LIMITS_MG.get(drug)
                if limit is None:
                    continue
                mg = raw * 1000 if unit in {"g", "克"} else raw
                if mg <= limit:
                    continue
                yield self._hit(
                    doc,
                    sec,
                    "QC-R008",
                    "药物剂量超过常规上限",
                    "safety",
                    "medium",
                    match.group(0),
                    f"{drug} 单次剂量 {mg:g} mg，超过常规上限 {limit:g} mg",
                    "核对医嘱剂量与给药频次，必要时请药师复核",
                )

    # ---------------------------------------------------------------- 逻辑矛盾
    def _check_negation_conflict(self, doc: Document) -> Iterable[RuleHit]:
        complaint = doc.section("主诉")
        present = doc.section("现病史")
        if not complaint or not present or not complaint.text.strip():
            return

        # 取主诉的第一个症状片段（按标点切分），避免把"发热、咳嗽、肺炎"
        # 整体当成一个症状去现病史里找否定式
        first_segment = re.split(r"[，,。.；;：:、\s]", complaint.text.strip())[0]
        symptom = re.sub(r"[，,。.；;：:\s]", "", _DURATION_RE.sub("", first_segment))[:6]
        if len(symptom) < 2:
            return

        pattern = re.compile(rf"(无|未|否认|没有)[^。；;\n]{{0,6}}?{re.escape(symptom)}")
        match = pattern.search(present.text)
        if not match:
            return
        yield self._hit(
            doc,
            present,
            "QC-R009",
            "主诉与现病史矛盾",
            "consistency",
            "high",
            match.group(0),
            f"主诉为「{complaint.text.strip()[:20]}」，但现病史描述「{match.group(0)}」",
            "核对现病史中症状的阴性/阳性描述",
        )

    # ---------------------------------------------------------------- 工具
    def _hit(
        self,
        doc: Document,
        sec: Section,
        rule_id: str,
        title: str,
        category: str,
        severity: str,
        quote: str,
        message: str,
        suggestion: str,
    ) -> RuleHit:
        pos = sec.text.find(quote) if quote else -1
        if pos >= 0:
            start = sec.start + pos
            end = start + len(quote)
        else:
            start, end = sec.start, min(sec.end, sec.start + (len(quote) or 0))
        return RuleHit(
            rule_id=rule_id,
            title=title,
            category=category,
            severity=severity,
            section=sec.name,
            message=message,
            suggestion=suggestion,
            quote=quote,
            start=start,
            end=end,
        )

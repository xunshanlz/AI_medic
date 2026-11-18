"""病历文书分段与原文组合。

真实 EMR 有两种输入形态：
1. 结构化字段（HIS/EMR 系统按 section 推送），
2. 纯文本（医生端复制、PDF 解析结果）。

两种形态都要落到同一个 ``Document``：既保留 section 语义（规则与提示词按
section 取数），又维护 section 在整篇文本中的绝对偏移（证据高亮需要）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "SECTION_TITLES",
    "Section",
    "Document",
    "split_sections",
    "build_document",
    "canonical_title",
]


# 常见病历段落标题（按书写规范顺序）
SECTION_TITLES: tuple[str, ...] = (
    "主诉",
    "现病史",
    "既往史",
    "个人史",
    "婚育史",
    "月经史",
    "家族史",
    "体格检查",
    "专科检查",
    "辅助检查",
    "初步诊断",
    "鉴别诊断",
    "诊疗计划",
    "病程记录",
    "手术记录",
    "医嘱",
    "医师签名",
)

# 结构化字段名 -> 中文标题。兼容 HIS 常见英文 key。
_ALIASES: dict[str, str] = {
    "chief_complaint": "主诉",
    "present_illness": "现病史",
    "history_of_present_illness": "现病史",
    "past_history": "既往史",
    "personal_history": "个人史",
    "marital_history": "婚育史",
    "menstrual_history": "月经史",
    "family_history": "家族史",
    "physical_exam": "体格检查",
    "specialist_exam": "专科检查",
    "auxiliary_exam": "辅助检查",
    "lab_results": "辅助检查",
    "diagnosis": "初步诊断",
    "preliminary_diagnosis": "初步诊断",
    "differential_diagnosis": "鉴别诊断",
    "treatment_plan": "诊疗计划",
    "progress_note": "病程记录",
    "operation_record": "手术记录",
    "orders": "医嘱",
    "physician_signature": "医师签名",
}

_HEADER_RE = re.compile(
    r"(?m)^[ \t\u3000]*(" + "|".join(map(re.escape, SECTION_TITLES)) + r")[ \t]*[:：]?[ \t]*"
)


@dataclass(frozen=True)
class Section:
    """病历中的一个段落。``start``/``end`` 是段落正文在整篇文本中的绝对区间。"""

    name: str
    title: str
    text: str
    start: int
    end: int

    def contains(self, offset: int) -> bool:
        return self.start <= offset < self.end


@dataclass(frozen=True)
class Document:
    """一篇待质控的病历。"""

    text: str
    sections: tuple[Section, ...]

    def section(self, name: str) -> Section | None:
        canonical = canonical_title(name)
        for sec in self.sections:
            if sec.name == canonical:
                return sec
        return None

    def section_at(self, offset: int) -> Section | None:
        for sec in self.sections:
            if sec.contains(offset):
                return sec
        return None

    def has_content(self, name: str) -> bool:
        sec = self.section(name)
        return bool(sec and sec.text.strip())

    def section_text(self, name: str) -> str:
        sec = self.section(name)
        return sec.text if sec else ""

    @property
    def length(self) -> int:
        return len(self.text)


def canonical_title(key: str) -> str:
    """把任意字段名/标题归一为规范中文标题。"""
    key = (key or "").strip()
    if key in _ALIASES:
        return _ALIASES[key]
    if key in SECTION_TITLES:
        return key
    # 去掉常见后缀噪声，如 "主诉（S）"
    cleaned = re.sub(r"[（(].*?[)）]", "", key).strip()
    return cleaned or key


def split_sections(raw_text: str) -> list[Section]:
    """按段落标题切分纯文本病历。

    没有识别到任何标题时，整篇作为单一 ``全文`` 段落，保证下游逻辑不空转。
    """
    if not raw_text:
        return []

    matches = list(_HEADER_RE.finditer(raw_text))
    if not matches:
        return [Section("全文", "全文", raw_text, 0, len(raw_text))]

    sections: list[Section] = []
    for idx, match in enumerate(matches):
        title = match.group(1)
        body_start = match.end()
        body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_text)
        body = raw_text[body_start:body_end]
        # 去掉段落尾部多余空行，但不改变起点，避免偏移错乱
        stripped = body.rstrip("\n\r")
        sections.append(
            Section(
                name=title,
                title=title,
                text=stripped,
                start=body_start,
                end=body_start + len(stripped),
            )
        )

    # 首个标题之前若还有内容（如页眉、患者信息），保留为"文首"
    if matches[0].start() > 0:
        head = raw_text[: matches[0].start()].rstrip("\n\r")
        if head.strip():
            sections.insert(0, Section("文首", "文首", head, 0, len(head)))
    return sections


def build_document(
    record: Mapping[str, Any] | None = None,
    raw_text: str | None = None,
) -> Document:
    """把结构化字段或纯文本统一构造成 ``Document``。

    结构化输入会按"标题：正文"的顺序拼接成完整文本，因此证据 offset 始终
    指向这段可展示、可高亮的文本。
    """
    if raw_text:
        return Document(text=raw_text, sections=tuple(split_sections(raw_text)))
    if not record:
        raise ValueError("record 与 raw_text 至少提供一个")

    parts: list[str] = []
    sections: list[Section] = []
    offset = 0
    for key, value in record.items():
        if value is None:
            continue
        text = value if isinstance(value, str) else _stringify(value)
        if not text.strip():
            continue
        title = canonical_title(str(key))
        prefix = f"{title}："
        body_start = offset + len(prefix)
        chunk = prefix + text + "\n"
        sections.append(
            Section(name=title, title=title, text=text, start=body_start, end=body_start + len(text))
        )
        parts.append(chunk)
        offset += len(chunk)

    return Document(text="".join(parts), sections=tuple(sections))


def _stringify(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "；".join(_stringify(v) for v in value if v is not None)
    if isinstance(value, Mapping):
        return "；".join(f"{k}:{v}" for k, v in value.items())
    return str(value)

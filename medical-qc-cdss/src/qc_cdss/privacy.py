"""隐私脱敏。

医疗数据出域是合规红线。即便在私有化环境，送进模型上下文的内容也应尽量
去除直接标识符（姓名、身份证、手机号、住址），只保留质控所需的临床信息。

策略：正则替换为语义化占位符（``<NAME>`` / ``<ID>``），不做可逆映射——
质控结论只需要"是哪一段病历"，不需要"是谁"。这样即使日志被导出也不泄露 PII。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["mask_pii", "mask_pii_aligned", "mask_document", "PiiStats"]

# 中文与数字同属 \w，不能用 \b 做边界；改用数字前后向断言
_ID_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
# "姓名：张三" / "患者：张三" / "联系人：李四"
# 冒号是必需的：否则"患者发热，3天"会把症状误判为姓名，导致正文被破坏、
# 证据引文无法与原文对应。
_NAME_RE = re.compile(r"(姓名|患者|联系人|家属)\s*[:：]\s*([\u4e00-\u9fff]{2,4})(?=[，。；、\s]|$)")
_ADDRESS_RE = re.compile(r"(住址|地址)\s*[:：]\s*[^\n，。；]{4,40}")


@dataclass
class PiiStats:
    ids: int = 0
    phones: int = 0
    emails: int = 0
    names: int = 0
    addresses: int = 0

    @property
    def total(self) -> int:
        return self.ids + self.phones + self.emails + self.names + self.addresses

    def as_dict(self) -> dict[str, int]:
        return {
            "ids": self.ids,
            "phones": self.phones,
            "emails": self.emails,
            "names": self.names,
            "addresses": self.addresses,
        }


def mask_pii(text: str) -> tuple[str, PiiStats]:
    """返回脱敏文本与命中统计。"""
    stats = PiiStats()
    if not text:
        return text, stats

    masked, n = _ID_RE.subn("<ID>", text)
    stats.ids = n
    masked, n = _PHONE_RE.subn("<PHONE>", masked)
    stats.phones = n
    masked, n = _EMAIL_RE.subn("<EMAIL>", masked)
    stats.emails = n

    def _name_sub(match: re.Match[str]) -> str:
        stats.names += 1
        return f"{match.group(1)}<NAME>"

    masked = _NAME_RE.sub(_name_sub, masked)

    def _addr_sub(match: re.Match[str]) -> str:
        stats.addresses += 1
        return f"{match.group(1)}<ADDRESS>"

    masked = _ADDRESS_RE.sub(_addr_sub, masked)
    return masked, stats


def mask_pii_aligned(text: str) -> str:
    """**等长**脱敏：字符数不变，因此原文偏移在脱敏文本上依然成立。

    这是让"送模型前脱敏"与"证据可溯源"共存的关键：
    模型看到的是脱敏文本，但每个字符的位置与原病历一一对应，
    溯源校验得到的 ``start/end`` 可以直接用于前端高亮原始病历。

    身份证/手机号 -> ``0``，邮箱 -> ``x``（保留 @ 与 .），姓名/地址 -> ``某``。
    不触碰换行，保证段落结构不变。
    """
    if not text:
        return text

    chars = list(text)

    def _fill(pattern: re.Pattern[str], fill: str, group: int = 0) -> None:
        for match in pattern.finditer(text):
            start, end = match.span(group)
            for i in range(start, end):
                if chars[i] not in "\r\n":
                    chars[i] = fill

    _fill(_ID_RE, "0")
    _fill(_PHONE_RE, "0")

    # 邮箱：保留 @ 与 . 的结构，其余替换为 x
    for match in _EMAIL_RE.finditer(text):
        for i in range(match.start(), match.end()):
            if chars[i] not in "@.":
                chars[i] = "x"

    _fill(_NAME_RE, "某", group=2)
    for match in _ADDRESS_RE.finditer(text):
        label_end = match.start(1) + len(match.group(1))
        for i in range(label_end, match.end()):
            if chars[i] not in "\r\n":
                chars[i] = "某"

    return "".join(chars)


def mask_document(document):
    """对整篇病历做等长脱敏，保持段落划分与偏移不变。

    直接在原文上逐段替换，而不是"脱敏后再按标题切分"——后者在段内恰好
    出现标题样式文本时会切出错误段落，导致规则与高亮错位。
    """
    from .domain.sections import Document, Section

    masked_text = mask_pii_aligned(document.text)
    if masked_text == document.text:
        return document

    sections = tuple(
        Section(
            name=sec.name,
            title=sec.title,
            text=mask_pii_aligned(sec.text),
            start=sec.start,
            end=sec.end,
        )
        for sec in document.sections
    )
    return Document(text=masked_text, sections=sections)

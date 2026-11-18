"""文本归一化与偏移映射。

为什么需要这一层
----------------
"证据可追溯"是医疗大模型产品的合规底线：模型给出的每条结论都必须能定位回
原文并高亮。但模型输出与病历原文之间普遍存在噪声——全角/半角混用、换行与
空格不一致、引号写法不同、大小写差异。直接做 ``quote in source`` 会让大量
真实存在的证据被判成"幻觉"，进而触发无意义的转人工。

本模块把原文归一化为"匹配用文本"，同时维护
``归一化下标 -> 原文下标`` 的映射；匹配命中后可无损映射回原文区间：

    n = normalize(source)
    i = n.text.find(quote_norm)
    start, end = n.span_to_original(i, i + len(quote_norm))

因此既能提高召回，又不会破坏可解释性（前端拿到的 offset 永远指向原文）。

实现约束：仅使用标准库，保证可在任意环境单测。
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["NormalizedText", "normalize", "normalize_text", "compact", "trim"]


# 零宽字符：不可见但会打断子串匹配，直接丢弃（不进入映射）
_ZERO_WIDTH = frozenset("\u200b\u200c\u200d\ufeff\u2060")

# 视作空白的字符：统一折叠为单个半角空格
_WHITESPACE = frozenset(" \t\r\n\f\v\u00a0\u3000")

# 中文/全角标点 -> 半角。注意个别字符展开为多个字符（如省略号），
# 展开后的每个字符都映射回同一个原文字符，span 映射仍然正确。
_PUNCT_MAP = {
    "\u201c": '"',  # “
    "\u201d": '"',  # ”
    "\u2018": "'",  # ‘
    "\u2019": "'",  # ’
    "\u3002": ".",  # 。
    "\uff0c": ",",  # ，
    "\u3001": ",",  # 、
    "\uff1b": ";",  # ；
    "\uff1a": ":",  # ：
    "\uff01": "!",  # ！
    "\uff1f": "?",  # ？
    "\u3010": "[",  # 【
    "\u3011": "]",  # 】
    "\u300a": "<",  # 《
    "\u300b": ">",  # 》
    "\u2014": "-",  # —
    "\uff0d": "-",  # －
    "\uff5e": "~",  # ～
    "\u2026": "...",  # …
    "\u00b7": ".",  # ·
}


# 中文数字 -> 阿拉伯数字。病历里"发热三天"与模型输出"发热3天"极常见，
# 不做等价会白白损失召回。只映射个位数字（"十"存在"十四"等组合歧义，不处理）。
_DIGIT_MAP = {
    "零": "0",
    "一": "1",
    "二": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
}


@dataclass(frozen=True)
class NormalizedText:
    """归一化文本及其到原文的偏移映射。

    Attributes:
        text: 归一化后的文本（匹配用）。
        index_map: 长度与 ``text`` 相同；``index_map[i]`` 是第 i 个归一化字符
            对应的原文下标。
        original: 原始文本，供截取与回填。
    """

    text: str
    index_map: tuple[int, ...]
    original: str

    def __post_init__(self) -> None:
        if len(self.text) != len(self.index_map):
            raise ValueError(
                f"index_map 长度 {len(self.index_map)} 与文本长度 {len(self.text)} 不一致"
            )

    def span_to_original(self, start: int, end: int) -> tuple[int, int] | None:
        """把归一化区间 ``[start, end)`` 映射回原文区间。

        区间为空或越界时返回 None（调用方据此判为"未定位"）。
        """
        if not self.text:
            return None
        start = max(0, start)
        end = min(end, len(self.text))
        if start >= end:
            return None
        return self.index_map[start], self.index_map[end - 1] + 1

    def original_slice(self, start: int, end: int) -> str:
        span = self.span_to_original(start, end)
        if span is None:
            return ""
        return self.original[span[0] : span[1]]


def normalize(text: str) -> NormalizedText:
    """归一化文本并生成偏移映射。

    规则：丢弃零宽字符 -> 全角转半角/中文标点转半角 -> 折叠连续空白为单个
    空格 -> ASCII 字母转小写。所有变换均保持字符与原文的对应关系。
    """
    if not text:
        return NormalizedText("", (), "")

    chars: list[str] = []
    index_map: list[int] = []
    pending_space: int | None = None

    for i, ch in enumerate(text):
        if ch in _ZERO_WIDTH:
            continue

        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:  # 全角 ASCII
            ch = chr(code - 0xFEE0)
        elif ch in _PUNCT_MAP:
            ch = _PUNCT_MAP[ch]
        elif ch in _DIGIT_MAP:
            ch = _DIGIT_MAP[ch]

        if ch in _WHITESPACE:
            # 连续空白只保留第一个，映射到第一个空白字符的位置
            if pending_space is None:
                pending_space = i
            continue

        if pending_space is not None:
            chars.append(" ")
            index_map.append(pending_space)
            pending_space = None

        if ch.isascii() and ch.isupper():
            ch = ch.lower()

        for expanded in ch:  # 处理 "…" -> "..." 这类一对多展开
            chars.append(expanded)
            index_map.append(i)

    return NormalizedText("".join(chars), tuple(index_map), text)


def normalize_text(text: str) -> str:
    """只需归一化文本、不关心偏移时的便捷函数。"""
    return normalize(text).text


def compact(n: NormalizedText) -> NormalizedText:
    """去掉全部空格，用于处理换行/空格不一致的场景。"""
    chars: list[str] = []
    index_map: list[int] = []
    for i, ch in enumerate(n.text):
        if ch == " ":
            continue
        chars.append(ch)
        index_map.append(n.index_map[i])
    return NormalizedText("".join(chars), tuple(index_map), n.original)


def trim(n: NormalizedText) -> NormalizedText:
    """去掉首尾空白，同时裁剪映射，保证下标仍然正确。"""
    start = 0
    end = len(n.text)
    while start < end and n.text[start] == " ":
        start += 1
    while end > start and n.text[end - 1] == " ":
        end -= 1
    if start == 0 and end == len(n.text):
        return n
    return NormalizedText(n.text[start:end], n.index_map[start:end], n.original)

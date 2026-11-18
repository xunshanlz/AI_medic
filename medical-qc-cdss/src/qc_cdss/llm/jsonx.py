"""JSON 容错解析（结构化输出治理的第二层）。

LLM 输出 JSON 的常见坏味道：markdown 代码围栏、前后夹杂解释文字、
尾逗号、Python 字面量（True/False/None）、输出被截断。

这一层不引入 json_repair 之类的重依赖，只做**可证明安全**的修复：
- 围栏剥离、括号配对提取（能处理前后散文）；
- 尾逗号、Python 字面量替换（正则，只在字符串外生效）；
- 截断时的字段级抢救（``salvage_mapping``）。

任何修复都不会改变合法 JSON 的语义，失败则抛 ``ValueError`` 交由上层降级。
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Mapping

__all__ = [
    "strip_code_fences",
    "extract_json_blocks",
    "loads_lenient",
    "salvage_mapping",
]

_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_+\-.]*\s*\n(.*?)\n?\s*```$", re.S)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_PY_LITERAL_RE = re.compile(r"(?<![\w\"'])(True|False|None)(?![\w\"'])")


def strip_code_fences(text: str) -> str:
    """去掉 ```json ... ``` 围栏（含未闭合的情况）。"""
    stripped = (text or "").strip()
    match = _FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_+\-.]*\s*\n?", "", stripped)
        stripped = re.sub(r"\n?\s*```\s*$", "", stripped)
    return stripped.strip()


def extract_json_blocks(text: str) -> list[str]:
    """从任意文本中提取括号配对的 JSON 片段（忽略字符串内的括号）。"""
    blocks: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch not in "{[":
            i += 1
            continue
        block = _balanced_slice(text, i)
        if block:
            blocks.append(block)
            i += len(block)
        else:
            i += 1
    return blocks


def _balanced_slice(text: str, start: int) -> str | None:
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _repair(candidate: str) -> str:
    repaired = _TRAILING_COMMA_RE.sub(r"\1", candidate)
    repaired = _PY_LITERAL_RE.sub(
        lambda m: {"True": "true", "False": "false", "None": "null"}[m.group(1)], repaired
    )
    return repaired


def loads_lenient(text: str) -> Any:
    """尽最大努力把模型输出解析为 Python 对象。

    Raises:
        ValueError: 所有策略均失败。
    """
    if not text or not text.strip():
        raise ValueError("空输出，无法解析 JSON")

    candidates: list[str] = []
    stripped = text.strip()
    candidates.append(stripped)
    unfenced = strip_code_fences(stripped)
    if unfenced != stripped:
        candidates.append(unfenced)
    candidates.extend(extract_json_blocks(unfenced))

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        for attempt in (candidate, _repair(candidate)):
            try:
                return json.loads(attempt)
            except (json.JSONDecodeError, TypeError):
                continue

    preview = stripped[:120].replace("\n", "\\n")
    raise ValueError(f"无法解析为 JSON：{preview}")


def salvage_mapping(text: str, keys: Iterable[str]) -> dict[str, Any]:
    """从残缺输出中抢救出顶层字段，用于第三层降级。"""
    try:
        parsed = loads_lenient(text)
    except ValueError:
        return {}
    if not isinstance(parsed, Mapping):
        return {}
    wanted = set(keys)
    return {k: v for k, v in parsed.items() if k in wanted}

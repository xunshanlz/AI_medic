"""质控结果去重、评分与统计聚合。

评分只作为"病历质量趋势"的粗粒度指标，**不用于**对医生的考核排名
（这是产品红线，见开发文档"风险与合规"）。因此算法刻意保持可解释：

    score = 100 - Σ 严重级别扣分

去重解决规则与 LLM 重复报同一问题（例如"诊断依据不足"既可能被规则命中，
也可能被模型指出），按 (code, section) + 区间重叠 + 引文前缀三重判定。
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

__all__ = ["SEVERITY_PENALTY", "SEVERITY_ORDER", "dedupe", "aggregate"]

SEVERITY_PENALTY: dict[str, float] = {"high": 10.0, "medium": 5.0, "low": 2.0}
SEVERITY_ORDER: dict[str, int] = {"high": 0, "medium": 1, "low": 2}


def _severity_value(item: Mapping[str, Any]) -> str:
    """兼容枚举与字符串两种输入（pydantic model_dump 可能返回枚举成员）。"""
    raw = item.get("severity")
    value = getattr(raw, "value", raw)
    return str(value or "low")


def _span(item: Mapping[str, Any]) -> tuple[int, int]:
    evidence = item.get("evidence") or {}
    if isinstance(evidence, Mapping):
        return int(evidence.get("start") or 0), int(evidence.get("end") or 0)
    return 0, 0


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> float:
    """两个区间的重叠比例（以较短区间为分母）。"""
    if a[1] <= a[0] or b[1] <= b[0]:
        return 0.0
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    shorter = min(a[1] - a[0], b[1] - b[0])
    return inter / shorter if shorter else 0.0


def _fingerprint(item: Mapping[str, Any]) -> str:
    quote = str(item.get("quote") or "")[:16]
    return f"{item.get('code')}|{item.get('section')}|{quote}"


def dedupe(findings: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """按严重级别优先保留，去掉重复报告。"""
    ordered = sorted(
        (dict(f) for f in findings),
        key=lambda f: (SEVERITY_ORDER.get(_severity_value(f), 9), -_span(f)[0]),
    )
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in ordered:
        fp = _fingerprint(item)
        if fp in seen:
            continue
        duplicate = False
        for existing in kept:
            if existing.get("code") != item.get("code"):
                continue
            if existing.get("section") != item.get("section"):
                continue
            if _overlap(_span(existing), _span(item)) >= 0.5:
                duplicate = True
                break
            if (
                existing.get("quote")
                and item.get("quote")
                and str(existing["quote"])[:12] == str(item["quote"])[:12]
            ):
                duplicate = True
                break
        if duplicate:
            continue
        seen.add(fp)
        kept.append(item)

    kept.sort(key=lambda f: (SEVERITY_ORDER.get(_severity_value(f), 9), _span(f)[0]))
    return kept


def aggregate(
    findings: Sequence[Mapping[str, Any]],
    base_score: float = 100.0,
) -> dict[str, Any]:
    """去重 + 评分 + 统计。"""
    unique = dedupe(findings)
    penalty = sum(SEVERITY_PENALTY.get(_severity_value(f), 0.0) for f in unique)
    score = max(0.0, round(base_score - penalty, 1))

    counts = {"high": 0, "medium": 0, "low": 0}
    for item in unique:
        severity = _severity_value(item)
        counts[severity] = counts.get(severity, 0) + 1

    needs_review = sum(
        1
        for f in unique
        if f.get("needs_review") or (f.get("evidence") or {}).get("state") == "pending"
    )
    return {
        "score": score,
        "counts": counts,
        "total": len(unique),
        "needs_review": needs_review,
        "raw_total": len(findings) if not isinstance(findings, list) else len(findings),
    }

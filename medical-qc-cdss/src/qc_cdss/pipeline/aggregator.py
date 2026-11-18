"""结果聚合：去重、评分、生成摘要。

摘要由模板生成而非模型生成：它只是结果的"读法"，用模型会引入不可控的
措辞与额外延迟，还会让同一份病历两次质控的摘要不一致（审计不友好）。
"""

from __future__ import annotations

from typing import Any, Sequence

from ..domain.scoring import aggregate
from ..schemas import Finding

__all__ = ["aggregate_findings", "summarize"]


def aggregate_findings(findings: Sequence[Finding]) -> dict[str, Any]:
    return aggregate([finding.model_dump() for finding in findings])


def summarize(score: float, counts: dict[str, int], total: int, needs_review: int) -> str:
    if total == 0:
        return "未发现明显质控问题，病历内涵质量良好。"
    parts = [f"病历质量得分 {score:.1f} 分，共发现 {total} 项问题"]
    detail = []
    if counts.get("high"):
        detail.append(f"高危 {counts['high']} 项")
    if counts.get("medium"):
        detail.append(f"中危 {counts['medium']} 项")
    if counts.get("low"):
        detail.append(f"低危 {counts['low']} 项")
    if detail:
        parts.append("（" + "、".join(detail) + "）")
    if needs_review:
        parts.append(f"，其中 {needs_review} 项待人工复核")
    parts.append("。建议优先处理高危项并补充证据不足的记录。")
    return "".join(parts)

"""多路召回融合（Reciprocal Rank Fusion）。

RRF 只使用**排名**而非原始分数，天然规避了"向量 cosine 分与 BM25 分不可比"
的问题，是工业界多路召回融合的事实标准（Milvus、Elasticsearch、Vespa 均内置）。

    score(d) = Σ_i w_i / (k + rank_i(d))

k 取 60 是 Cormack 等人在 TREC 上的经验值；本模块保留可配置。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

__all__ = ["FusedHit", "rrf_fuse"]


@dataclass(frozen=True)
class FusedHit:
    id: str
    score: float
    ranks: tuple[int, ...]

    def as_dict(self) -> dict[str, object]:
        return {"id": self.id, "score": round(self.score, 6), "ranks": list(self.ranks)}


def rrf_fuse(
    runs: Sequence[Sequence[tuple[str, float]]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
    top_k: int | None = None,
) -> list[FusedHit]:
    """融合多路召回结果。

    Args:
        runs: 每路召回的有序结果 ``[(doc_id, score), ...]``，按相关性降序。
        k: RRF 平滑常数，越大越弱化头部排名差异。
        weights: 各路的权重，默认等权。
        top_k: 截断数量。

    Returns:
        按融合分降序的 ``FusedHit``；``ranks`` 记录该文档在各路的 1-based 排名，
        用于可解释性（前端/审计可展示"词汇召回第 3、向量召回第 1"）。
    """
    if weights is not None and len(weights) != len(runs):
        raise ValueError("weights 长度必须与 runs 一致")
    if k <= 0:
        raise ValueError("k 必须为正数")

    scores: dict[str, float] = {}
    ranks: dict[str, list[int]] = {}

    for run_idx, run in enumerate(runs):
        weight = 1.0 if weights is None else float(weights[run_idx])
        if weight == 0:
            # 该路不参与，但仍需补齐 ranks 占位，保持维度一致
            pass
        seen: set[str] = set()
        for position, (doc_id, _score) in enumerate(run, start=1):
            if doc_id in seen:  # 同一路内去重，保留最高排名
                continue
            seen.add(doc_id)
            slot = ranks.setdefault(doc_id, [0] * len(runs))
            slot[run_idx] = position
            if weight:
                scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + position)

    fused = [
        FusedHit(id=doc_id, score=score, ranks=tuple(ranks[doc_id]))
        for doc_id, score in scores.items()
    ]
    fused.sort(key=lambda hit: (-hit.score, hit.ranks))
    return fused[:top_k] if top_k else fused

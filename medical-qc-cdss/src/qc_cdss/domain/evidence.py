"""确定性证据溯源校验。

背景
----
LLM 生成的质控结论常带"证据引文"，但引文可能被改写、拼接、甚至凭空编造。
医疗场景不能接受"看起来像"的引用，因此需要一层**不依赖模型**的校验器：

1. 精确匹配：归一化后子串命中 -> 认定"已核验"，返回原文 offset。
2. 近似匹配：滑动窗口 + 序列相似度，命中阈值 -> "已核验"（记录 score），
   低于阈值但仍有相似度 -> "待核验"（转人工），
3. 完全无法定位 -> "无效"，即判定为幻觉，结论不进入自动采纳。

三态设计对应产品策略：
- verified：可直接展示并计入自动采纳；
- pending：展示但强制人工复核（不阻断主流程）；
- invalid：拦截，不展示给临床，只进审计日志与模型评测集。

实现约束：默认仅用标准库（difflib）。若安装了 ``rapidfuzz`` 则自动启用其
C++ 实现（partial_ratio_alignment），性能与精度更好，但**不是**运行必需。
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from .textnorm import compact, normalize, trim

try:  # pragma: no cover - 取决于部署环境是否安装加速库
    from rapidfuzz import fuzz as _rf_fuzz
except Exception:  # pragma: no cover
    _rf_fuzz = None

__all__ = [
    "VERIFIED",
    "PENDING",
    "INVALID",
    "Verification",
    "EvidenceVerifier",
]

VERIFIED = "verified"
PENDING = "pending"
INVALID = "invalid"


@dataclass(frozen=True)
class Verification:
    """一条证据的校验结果。"""

    state: str
    score: float
    start: int | None = None
    end: int | None = None
    matched: str | None = None
    strategy: str = "none"

    @property
    def is_valid(self) -> bool:
        return self.state == VERIFIED

    @property
    def needs_review(self) -> bool:
        return self.state == PENDING

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "score": round(self.score, 2),
            "start": self.start,
            "end": self.end,
            "matched": self.matched,
            "strategy": self.strategy,
        }


class EvidenceVerifier:
    """证据引文校验器（无状态，可并发复用）。

    Args:
        accept_threshold: 近似匹配达到该分数（0-100）判定为已核验。
        review_threshold: 低于 accept 但达到该分数判定为待核验。
        min_fuzzy_len: 短于该长度的引文不做模糊匹配，避免"两字巧合"式误判。
    """

    def __init__(
        self,
        accept_threshold: float = 88.0,
        review_threshold: float = 70.0,
        min_fuzzy_len: int = 4,
    ) -> None:
        if not 0 <= review_threshold <= accept_threshold <= 100:
            raise ValueError("阈值需满足 0 <= review <= accept <= 100")
        self.accept_threshold = accept_threshold
        self.review_threshold = review_threshold
        self.min_fuzzy_len = min_fuzzy_len

    # ------------------------------------------------------------------ public
    def verify(self, quote: str, source: str) -> Verification:
        """校验 ``quote`` 是否真实存在于 ``source`` 中。"""
        quote = (quote or "").strip()
        if not quote or not source:
            return Verification(INVALID, 0.0)

        nq = trim(normalize(quote))
        ns = normalize(source)
        if not nq.text:
            return Verification(INVALID, 0.0)

        # 1) 归一化后精确匹配
        hit = ns.text.find(nq.text)
        if hit >= 0:
            span = ns.span_to_original(hit, hit + len(nq.text))
            if span:
                return Verification(
                    VERIFIED,
                    100.0,
                    span[0],
                    span[1],
                    source[span[0] : span[1]],
                    "exact",
                )

        # 2) 去空白后精确匹配（处理换行/空格差异）
        cq = trim(compact(nq))
        cs = compact(ns)
        if cq.text:
            hit = cs.text.find(cq.text)
            if hit >= 0:
                span = cs.span_to_original(hit, hit + len(cq.text))
                if span:
                    return Verification(
                        VERIFIED,
                        100.0,
                        span[0],
                        span[1],
                        source[span[0] : span[1]],
                        "exact_compact",
                    )

        # 3) 近似匹配
        if len(cq.text) < self.min_fuzzy_len:
            return Verification(INVALID, 0.0)

        best = _BestMatch(0.0, None, "fuzzy")

        score, start, end = self._fuzzy(cq.text, cs.text)
        if start is not None and score > best.score:
            span = cs.span_to_original(start, end)
            if span:
                best = _BestMatch(score, span, "fuzzy_compact")

        if best.span is None:
            score, start, end = self._fuzzy(cq.text, ns.text)
            if start is not None:
                span = ns.span_to_original(start, end)
                if span:
                    best = _BestMatch(score, span, "fuzzy")

        if best.span is None:
            return Verification(INVALID, 0.0)

        start, end = best.span
        matched = source[start:end]
        if best.score >= self.accept_threshold:
            state = VERIFIED
        elif best.score >= self.review_threshold:
            state = PENDING
        else:
            # 低于复核阈值视为幻觉：不返回区间，避免前端展示"看似有据"的高亮
            return Verification(INVALID, best.score)
        return Verification(state, best.score, start, end, matched, best.strategy)

    # ----------------------------------------------------------------- private
    def _fuzzy(self, quote: str, text: str) -> tuple[float, int | None, int | None]:
        """在 text 中寻找与 quote 最相似的片段，返回 (score, start, end)。"""
        if not text or not quote:
            return 0.0, None, None

        if _rf_fuzz is not None:  # pragma: no cover
            try:
                aligned = _rf_fuzz.partial_ratio_alignment(quote, text)
                if aligned is not None:
                    score = float(aligned.score)
                    s, e = int(aligned.dest_start), int(aligned.dest_end)
                    if e > s:
                        return score, s, e
            except Exception:
                pass

        qlen = len(quote)
        candidate_starts: set[int] = set()

        # 以首/中/尾三个锚点定位候选窗口，避免对全文做 O(n*m) 滑动
        for anchor in (quote[:4], quote[qlen // 2 : qlen // 2 + 4], quote[-4:]):
            if len(anchor) < 2:
                continue
            pos = text.find(anchor)
            guard = 0
            while pos != -1 and guard < 12:
                for delta in (0, qlen // 4, qlen // 2, (3 * qlen) // 4):
                    candidate_starts.add(max(0, pos - delta))
                pos = text.find(anchor, pos + 1)
                guard += 1

        if not candidate_starts:
            # 完全没有锚点：仅在文本与引文长度可比时才做一次全局比较
            if len(text) <= 3 * qlen:
                candidate_starts.add(0)
            else:
                return 0.0, None, None

        best_score = 0.0
        best_span: tuple[int, int] | None = None
        for start in candidate_starts:
            window = text[start : start + qlen]
            if not window:
                continue
            matcher = SequenceMatcher(None, quote, window, autojunk=False)
            ratio = matcher.ratio()
            if ratio <= best_score:
                continue
            blocks = [b for b in matcher.get_matching_blocks() if b.size > 0]
            if not blocks:
                continue
            block = max(blocks, key=lambda b: b.size)
            best_score = ratio
            best_span = (start + block.b, start + block.b + block.size)

        if best_span is None:
            return 0.0, None, None
        return best_score * 100.0, best_span[0], best_span[1]


@dataclass(frozen=True)
class _BestMatch:
    score: float
    span: tuple[int, int] | None
    strategy: str

"""轻量 BM25 检索（中文字符 n-gram，无第三方依赖）。

为什么自带 BM25
---------------
医疗问答里大量查询是**精确术语**（"社区获得性肺炎""肌钙蛋白"），纯向量检索
容易漏；而 Milvus 2.5+ 虽内置 BM25 函数，本地开发/单测不应强依赖 Milvus。
因此：

- 生产环境可用 Milvus 内置稀疏 BM25（见 ``rag/store.py``）；
- 本地与单测用本实现做词汇召回，接口一致，切换无感。

中文用"单字 + 相邻二元组"切分，无需分词词典，对未登录术语（新药名、
缩写）更鲁棒。
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable

__all__ = ["tokenize", "BM25Index"]

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[a-z0-9]+(?:[.\-/][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """中英混合切分。

    中文以二元组为主：单字切分会把"的""无"这类高频字变成噪声词，导致
    无关查询也能命中；二元组在保持召回的同时显著降低误配。
    单字词（如"癌"）单独成词，长词整体保留以强化精确术语匹配。
    """
    if not text:
        return []
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text.lower()):
        chunk = match.group(0)
        if chunk[0].isascii():
            tokens.append(chunk)
            continue
        if len(chunk) == 1:
            tokens.append(chunk)
            continue
        tokens.extend(chunk[i : i + 2] for i in range(len(chunk) - 1))
        if len(chunk) <= 8:  # 完整术语保留，提升精确匹配权重
            tokens.append(chunk)
    return tokens


class BM25Index:
    """内存 BM25（适合万级文档；十万级以上建议用 Milvus/ES）。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_ids: list[str] = []
        self._lens: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._df: Counter[str] = Counter()
        self._idf: dict[str, float] = {}
        self._avgdl = 0.0
        self._built = False

    def add(self, doc_id: str, text: str, *, title: str = "") -> None:
        tokens = tokenize(text)
        if title:
            # 标题重复两次，等价于轻度加权，避免标题词被正文稀释
            tokens = tokenize(title) * 2 + tokens
        tf = Counter(tokens)
        idx = len(self._doc_ids)
        self._doc_ids.append(doc_id)
        self._lens.append(len(tokens))
        for term, freq in tf.items():
            self._postings[term].append((idx, freq))
            self._df[term] += 1
        self._built = False

    def build(self) -> "BM25Index":
        n = len(self._doc_ids)
        self._avgdl = (sum(self._lens) / n) if n else 0.0
        self._idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in self._df.items()
        }
        self._built = True
        return self

    def __len__(self) -> int:
        return len(self._doc_ids)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if not self._built:
            self.build()
        if not self._doc_ids:
            return []

        terms = Counter(tokenize(query))
        if not terms:
            return []

        scores: dict[int, float] = defaultdict(float)
        avgdl = self._avgdl or 1.0
        for term, qf in terms.items():
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf.get(term, 0.0)
            for doc_idx, tf in postings:
                dl = self._lens[doc_idx] or 1
                denom = tf + self.k1 * (1 - self.b + self.b * dl / avgdl)
                scores[doc_idx] += idf * (tf * (self.k1 + 1) / denom) * min(qf, 3)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [(self._doc_ids[i], score) for i, score in ranked]

    def bulk_add(self, docs: Iterable[tuple[str, str, str]]) -> "BM25Index":
        """docs: (doc_id, text, title)。"""
        for doc_id, text, title in docs:
            self.add(doc_id, text, title=title)
        return self.build()

"""指南知识库的切分、索引与混合检索。

检索链路（与开发文档一致）：

    query ──┬─ 稠密检索（向量库 top 3k）──┐
            └─ 稀疏检索（BM25  top 3k）──┴─ RRF 融合 ─ 截断 top k

采用 RRF 而非分数加权，是因为 cosine 分与 BM25 分量纲不同，工业界更稳的做法
是只用排名。融合后再可选做 Cross-Encoder 精排（见 CDSS 节点），本项目默认
用 RRF 结果 + LLM 证据约束，避免引入额外模型。

切分策略：按 markdown 二级标题成段，段内按空行聚合到 ~600 字，保留 80 字
重叠——医学指南的"推荐意见"通常整段出现，过细切分会丢上下文。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..logging import get_logger
from .bm25 import BM25Index
from .embedding import Embedder
from .fusion import rrf_fuse
from .store import Chunk, VectorStore

__all__ = ["RetrievedChunk", "Retriever", "chunk_markdown", "load_documents"]

logger = get_logger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    dense_rank: int = 0
    lexical_rank: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk.id,
            "doc_id": self.chunk.doc_id,
            "title": self.chunk.title,
            "text": self.chunk.text,
            "score": round(self.score, 6),
            "dense_rank": self.dense_rank,
            "lexical_rank": self.lexical_rank,
        }


def load_documents(directory: Path) -> list[tuple[str, str, str]]:
    """读取目录下的 md/txt，返回 (doc_id, title, text)。"""
    documents: list[tuple[str, str, str]] = []
    if not directory.exists():
        logger.warning("knowledge_dir_missing", path=str(directory))
        return documents
    for path in sorted(directory.glob("*.md")) + sorted(directory.glob("*.txt")):
        raw = path.read_text(encoding="utf-8")
        title = path.stem
        for line in raw.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        documents.append((path.stem, title, raw))
    return documents


def chunk_markdown(
    doc_id: str,
    title: str,
    text: str,
    *,
    max_chars: int = 600,
    overlap: int = 80,
) -> list[Chunk]:
    """按标题与段落切分，返回带稳定 id 的 chunk。"""
    sections: list[tuple[str, str]] = []
    current_heading = "正文"
    buffer: list[str] = []

    for line in text.splitlines():
        if line.startswith("## "):
            if buffer:
                sections.append((current_heading, "\n".join(buffer).strip()))
                buffer = []
            current_heading = line[3:].strip()
        elif line.startswith("# "):
            continue
        else:
            buffer.append(line)
    if buffer:
        sections.append((current_heading, "\n".join(buffer).strip()))

    chunks: list[Chunk] = []
    for heading, body in sections:
        if not body:
            continue
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        window = ""
        for paragraph in paragraphs:
            if len(window) + len(paragraph) + 1 <= max_chars:
                window = f"{window}\n{paragraph}".strip()
                continue
            if window:
                chunks.append(_make_chunk(doc_id, title, heading, window, len(chunks)))
            tail = window[-overlap:] if overlap and window else ""
            window = f"{tail}\n{paragraph}".strip() if tail else paragraph
            # 超长段落单独切窗，避免单个 chunk 超过模型上下文预算
            while len(window) > max_chars * 2:
                chunks.append(_make_chunk(doc_id, title, heading, window[:max_chars], len(chunks)))
                window = window[max_chars - overlap :]
        if window:
            chunks.append(_make_chunk(doc_id, title, heading, window, len(chunks)))
    return chunks


def _make_chunk(doc_id: str, title: str, heading: str, body: str, index: int) -> Chunk:
    # 标题进入正文，提升词汇召回与向量语义
    text = f"{title}·{heading}\n{body}"
    return Chunk(
        id=f"{doc_id}#{index:04d}",
        doc_id=doc_id,
        title=title,
        text=text,
        metadata={"section": heading},
    )


class Retriever:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder: Embedder,
        store: VectorStore,
        bm25: BM25Index,
        *,
        rrf_k: int = 60,
    ) -> None:
        self.chunks = list(chunks)
        self.embedder = embedder
        self.store = store
        self.bm25 = bm25
        self.rrf_k = rrf_k
        self._by_id = {chunk.id: chunk for chunk in self.chunks}

    @classmethod
    async def build(
        cls,
        directory: Path,
        embedder: Embedder,
        store: VectorStore,
        *,
        rrf_k: int = 60,
        max_chars: int = 600,
    ) -> "Retriever":
        chunks: list[Chunk] = []
        for doc_id, title, text in load_documents(directory):
            chunks.extend(chunk_markdown(doc_id, title, text, max_chars=max_chars))

        if chunks:
            vectors = await embedder.embed([chunk.text for chunk in chunks])
            await store.upsert(chunks, vectors)

        bm25 = BM25Index()
        for chunk in chunks:
            bm25.add(chunk.id, chunk.text, title=chunk.title)

        logger.info("retriever_built", chunks=len(chunks), rrf_k=rrf_k)
        return cls(chunks, embedder, store, bm25, rrf_k=rrf_k)

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 8,
        doc_ids: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        if not query.strip() or not self.chunks:
            return []

        vector = (await self.embedder.embed([query]))[0]
        dense = await self.store.search(vector, top_k * 3, doc_ids=doc_ids)
        dense_run = [(hit.chunk.id, hit.score) for hit in dense]
        lexical_run = self.bm25.search(query, top_k * 3)
        if doc_ids:
            allowed = set(doc_ids)
            lexical_run = [
                (chunk_id, score)
                for chunk_id, score in lexical_run
                if self._by_id.get(chunk_id) and self._by_id[chunk_id].doc_id in allowed
            ]

        fused = rrf_fuse([dense_run, lexical_run], k=self.rrf_k, top_k=top_k)
        results: list[RetrievedChunk] = []
        for hit in fused:
            chunk = self._by_id.get(hit.id)
            if chunk is None:
                continue
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=hit.score,
                    dense_rank=hit.ranks[0] if hit.ranks else 0,
                    lexical_rank=hit.ranks[1] if len(hit.ranks) > 1 else 0,
                )
            )
        return results

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)

    async def close(self) -> None:
        await self.store.close()
        await self.embedder.close()

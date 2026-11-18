"""向量存储适配层。

三种实现共享 ``VectorStore`` 协议：
- ``MemoryVectorStore``：单测/演示，numpy 余弦相似度；
- ``MilvusVectorStore``：生产主力，支持十亿级向量与标量过滤，2.5+ 还内置
  BM25 稀疏检索（本项目仍用应用层 BM25，便于与稠密检索做 RRF）；
- ``ChromaVectorStore``：轻量落盘方案，适合边缘部署或单机试点。

Milvus/Chroma 为同步客户端，统一用 ``asyncio.to_thread`` 包装，避免阻塞事件
循环——这是 FastAPI 服务在 GPU 推理期间仍能处理其它请求的关键。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

__all__ = [
    "Chunk",
    "ScoredChunk",
    "VectorStore",
    "MemoryVectorStore",
    "MilvusVectorStore",
    "ChromaVectorStore",
    "build_vector_store",
]

try:  # 向量运算可用 numpy 时性能更好，未安装也不影响功能
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class VectorStore(Protocol):
    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...

    async def search(
        self, vector: Sequence[float], top_k: int, *, doc_ids: Sequence[str] | None = None
    ) -> list[ScoredChunk]: ...

    async def count(self) -> int: ...

    async def close(self) -> None: ...


class MemoryVectorStore:
    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, list[float]] = {}

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks 与 vectors 数量不一致")
        for chunk, vector in zip(chunks, vectors):
            self._chunks[chunk.id] = chunk
            self._vectors[chunk.id] = list(vector)

    async def search(
        self, vector: Sequence[float], top_k: int, *, doc_ids: Sequence[str] | None = None
    ) -> list[ScoredChunk]:
        allowed = set(doc_ids) if doc_ids else None
        ids = [cid for cid in self._chunks if allowed is None or self._chunks[cid].doc_id in allowed]
        if not ids:
            return []

        if _np is not None:
            matrix = _np.array([self._vectors[cid] for cid in ids], dtype="float32")
            query = _np.array(vector, dtype="float32")
            norms = _np.linalg.norm(matrix, axis=1) * (_np.linalg.norm(query) or 1.0)
            scores = (matrix @ query) / _np.maximum(norms, 1e-9)
            order = _np.argsort(-scores)[:top_k]
            return [ScoredChunk(self._chunks[ids[i]], float(scores[i])) for i in order]

        query = list(vector)
        qnorm = sum(v * v for v in query) ** 0.5 or 1.0
        scored: list[tuple[float, str]] = []
        for cid in ids:
            vec = self._vectors[cid]
            dot = sum(a * b for a, b in zip(query, vec))
            norm = (sum(v * v for v in vec) ** 0.5) or 1.0
            scored.append((dot / (qnorm * norm), cid))
        scored.sort(reverse=True)
        return [ScoredChunk(self._chunks[cid], score) for score, cid in scored[:top_k]]

    async def count(self) -> int:
        return len(self._chunks)

    async def close(self) -> None:
        self._chunks.clear()
        self._vectors.clear()


class MilvusVectorStore:
    """Milvus 2.5+（MilvusClient 简化 API）。"""

    def __init__(self, uri: str, collection: str, dim: int) -> None:
        self.uri = uri
        self.collection = collection
        self.dim = dim
        self._client: Any | None = None
        self._lock = asyncio.Lock()

    async def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        async with self._lock:
            if self._client is None:  # pragma: no cover - 依赖外部服务
                from pymilvus import MilvusClient

                client = await asyncio.to_thread(MilvusClient, uri=self.uri)

                def _create() -> None:
                    if not client.has_collection(self.collection):
                        client.create_collection(
                            collection_name=self.collection,
                            dimension=self.dim,
                            primary_field_name="id",
                            id_type="string",
                            vector_field_name="vector",
                            metric_type="COSINE",
                            auto_id=False,
                            enable_dynamic_field=True,
                        )

                await asyncio.to_thread(_create)
                self._client = client
        return self._client

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        client = await self._ensure_client()

        def _write() -> None:
            rows = [
                {
                    "id": chunk.id,
                    "vector": list(vector),
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "text": chunk.text,
                    "section": chunk.metadata.get("section", ""),
                }
                for chunk, vector in zip(chunks, vectors)
            ]
            if rows:
                client.upsert(collection_name=self.collection, data=rows)

        await asyncio.to_thread(_write)

    async def search(
        self, vector: Sequence[float], top_k: int, *, doc_ids: Sequence[str] | None = None
    ) -> list[ScoredChunk]:
        client = await self._ensure_client()
        expr = None
        if doc_ids:
            # Milvus 表达式要求字符串用双引号
            joined = ", ".join('"' + str(d).replace('"', '\\"') + '"' for d in doc_ids)
            expr = f"doc_id in [{joined}]"

        def _search() -> list[ScoredChunk]:
            hits = client.search(
                collection_name=self.collection,
                data=[list(vector)],
                limit=top_k,
                filter=expr,
                output_fields=["doc_id", "title", "text", "section"],
            )
            results: list[ScoredChunk] = []
            for hit in (hits[0] if hits else []):
                entity = hit.get("entity", {}) or {}
                chunk = Chunk(
                    id=str(hit.get("id")),
                    doc_id=str(entity.get("doc_id", "")),
                    title=str(entity.get("title", "")),
                    text=str(entity.get("text", "")),
                    metadata={"section": entity.get("section", "")},
                )
                results.append(ScoredChunk(chunk, float(hit.get("distance", 0.0))))
            return results

        return await asyncio.to_thread(_search)

    async def count(self) -> int:  # pragma: no cover
        client = await self._ensure_client()
        stats = await asyncio.to_thread(client.get_collection_stats, self.collection)
        return int(stats.get("row_count", 0))

    async def close(self) -> None:  # pragma: no cover
        if self._client is not None:
            await asyncio.to_thread(self._client.close)
            self._client = None


class ChromaVectorStore:
    def __init__(self, path: str, collection: str) -> None:
        self.path = path
        self.collection_name = collection
        self._collection: Any | None = None
        self._lock = asyncio.Lock()

    async def _ensure_collection(self) -> Any:
        if self._collection is not None:
            return self._collection
        async with self._lock:
            if self._collection is None:  # pragma: no cover - 依赖外部依赖
                import chromadb

                client = await asyncio.to_thread(chromadb.PersistentClient, path=self.path)
                self._collection = await asyncio.to_thread(
                    client.get_or_create_collection,
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
        return self._collection

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        collection = await self._ensure_collection()

        def _write() -> None:
            collection.upsert(
                ids=[c.id for c in chunks],
                embeddings=[list(v) for v in vectors],
                documents=[c.text for c in chunks],
                metadatas=[
                    {"doc_id": c.doc_id, "title": c.title, "section": c.metadata.get("section", "")}
                    for c in chunks
                ],
            )

        await asyncio.to_thread(_write)

    async def search(
        self, vector: Sequence[float], top_k: int, *, doc_ids: Sequence[str] | None = None
    ) -> list[ScoredChunk]:
        collection = await self._ensure_collection()
        where = {"doc_id": {"$in": list(doc_ids)}} if doc_ids else None

        def _search() -> list[ScoredChunk]:
            result = collection.query(
                query_embeddings=[list(vector)],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            chunks: list[ScoredChunk] = []
            ids = (result.get("ids") or [[]])[0]
            docs = (result.get("documents") or [[]])[0]
            metas = (result.get("metadatas") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]
            for cid, text, meta, distance in zip(ids, docs, metas, distances):
                chunk = Chunk(
                    id=str(cid),
                    doc_id=str((meta or {}).get("doc_id", "")),
                    title=str((meta or {}).get("title", "")),
                    text=str(text or ""),
                    metadata={"section": (meta or {}).get("section", "")},
                )
                chunks.append(ScoredChunk(chunk, 1.0 - float(distance)))  # cosine 距离转相似度
            return chunks

        return await asyncio.to_thread(_search)

    async def count(self) -> int:  # pragma: no cover
        collection = await self._ensure_collection()
        return int(await asyncio.to_thread(collection.count))

    async def close(self) -> None:
        self._collection = None


def build_vector_store(settings: Any, dim: int) -> VectorStore:
    if settings.vector_store == "milvus":
        return MilvusVectorStore(settings.milvus_uri, settings.milvus_collection, dim)
    if settings.vector_store == "chroma":
        return ChromaVectorStore(str(settings.chroma_path), settings.milvus_collection)
    return MemoryVectorStore()

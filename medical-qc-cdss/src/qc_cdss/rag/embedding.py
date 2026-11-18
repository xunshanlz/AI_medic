"""文本向量化。

两种实现：
- ``OpenAIEmbedder``：调用 OpenAI 兼容的 ``/v1/embeddings``（vLLM 部署
  BGE-M3 / Qwen3-Embedding，或云厂商 embedding API）。生产用。
- ``HashEmbedder``：字符 n-gram 特征哈希，零依赖、确定性，用于离线演示与
  单测。它不代表检索效果，只保证链路可跑通。

embedding 维度必须与向量库 collection 一致，因此 ``dim`` 由实现显式暴露。
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, Sequence

from .bm25 import tokenize

__all__ = ["Embedder", "HashEmbedder", "OpenAIEmbedder", "build_embedder"]


class Embedder(Protocol):
    dim: int

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def close(self) -> None: ...


class HashEmbedder:
    """确定性特征哈希（signed hashing trick）。"""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            return vector
        return [v / norm for v in vector]

    async def close(self) -> None:
        return None


class OpenAIEmbedder:
    """OpenAI 兼容 embedding 端点。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = "bge-m3",
        dim: int = 1024,
        timeout: float = 30.0,
        batch_size: int = 32,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "EMPTY"
        self.model = model
        self.dim = dim
        self.timeout = timeout
        self.batch_size = batch_size
        self._client = None

    @property
    def client(self):
        if self._client is None:  # pragma: no cover
            import httpx

            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            response = await self.client.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": batch},
            )
            response.raise_for_status()
            payload = response.json()
            ordered = sorted(payload["data"], key=lambda item: item["index"])
            vectors.extend([item["embedding"] for item in ordered])
        return vectors

    async def close(self) -> None:
        if self._client is not None:  # pragma: no cover
            await self._client.aclose()


def build_embedder(settings) -> Embedder:
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(
            base_url=settings.embedding_base_url or settings.llm_base_url,
            api_key=(
                settings.embedding_api_key.get_secret_value()
                if settings.embedding_api_key
                else settings.llm_api_key.get_secret_value()
            ),
            model=settings.embedding_model,
            dim=settings.embedding_dim,
        )
    return HashEmbedder(dim=settings.embedding_dim)

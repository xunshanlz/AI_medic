"""结果缓存。

缓存策略（详见开发文档"性能与容量"）：
- 以 ``section 文本 + 检查项 + prompt 版本 + 模型`` 的哈希为 key，缓存 LLM 的
  单项判读结果。同一份病历被重复提交（医生反复保存）时可直接命中，避免重复
  推理——这是私有化小算力下最有效的降本手段。
- Redis 不可用时自动退回进程内 LRU，保证功能不中断（"优雅降级"）。

注意：缓存键**必须**包含 prompt/模型版本，否则提示词迭代后仍命中旧结果，
会污染评测口径。
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Protocol

__all__ = ["Cache", "MemoryCache", "RedisCache", "build_cache", "cache_key"]


def cache_key(*parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return f"qc:{digest}"


class Cache(Protocol):
    async def get(self, key: str) -> str | None: ...

    async def set(self, key: str, value: str, ttl: int | None = None) -> None: ...

    async def close(self) -> None: ...


class MemoryCache:
    """进程内 TTL 缓存，单机开发/降级用。"""

    def __init__(self, ttl: int = 3600, max_entries: int = 4096) -> None:
        self._ttl = ttl
        self._max_entries = max_entries
        self._data: dict[str, tuple[float, str]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> str | None:
        async with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at < time.monotonic():
                self._data.pop(key, None)
                return None
            return value

    async def set(self, key: str, value: str, ttl: int | None = None) -> None:
        async with self._lock:
            if len(self._data) >= self._max_entries:
                # 优先清理已过期项，其次淘汰最早写入项
                now = time.monotonic()
                expired = [k for k, (exp, _) in self._data.items() if exp < now]
                for k in expired[: self._max_entries // 4]:
                    self._data.pop(k, None)
                if len(self._data) >= self._max_entries:
                    oldest = min(self._data.items(), key=lambda kv: kv[1][0])[0]
                    self._data.pop(oldest, None)
            self._data[key] = (time.monotonic() + (ttl or self._ttl), value)

    async def close(self) -> None:
        self._data.clear()


class RedisCache:
    """Redis 缓存（生产环境）。连接失败由 ``build_cache`` 负责降级。"""

    def __init__(self, client, ttl: int = 3600) -> None:
        self._client = client
        self._ttl = ttl

    @classmethod
    async def connect(cls, url: str, ttl: int = 3600) -> "RedisCache":
        import redis.asyncio as redis  # 延迟导入：无 Redis 环境也能 import 本模块

        client = redis.from_url(url, encoding="utf-8", decode_responses=True)
        await client.ping()
        return cls(client, ttl)

    async def get(self, key: str) -> str | None:
        return await self._client.get(key)

    async def set(self, key: str, value: str, ttl: int | None = None) -> None:
        await self._client.set(key, value, ex=ttl or self._ttl)

    async def close(self) -> None:
        await self._client.aclose()


async def build_cache(redis_url: str | None, ttl: int) -> Cache:
    if redis_url:
        try:
            return await RedisCache.connect(redis_url, ttl)
        except Exception:  # pragma: no cover - 取决于部署环境
            from .logging import get_logger

            get_logger(__name__).warning("redis_unavailable_fallback_to_memory")
    return MemoryCache(ttl=ttl)

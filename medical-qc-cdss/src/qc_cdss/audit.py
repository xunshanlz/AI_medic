"""审计日志哈希链。

合规要求（等保 2.0 / 电子病历应用管理规范）：质控结论的生成、修改、采纳都要
留痕，且日志不可被静默篡改。

实现上采用**哈希链**：每条日志的 ``entry_hash = SHA256(prev_hash + payload_hash
+ timestamp)``。任何一条被改动或删除，后续哈希全部对不上，可用一条校验脚本
定位断点。生产环境还应把日志同步到独立存储（WORM/对象存储版本化），本模块
只解决"链"的问题。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any, Protocol

__all__ = ["canonical_json", "sha256_hex", "compute_entry_hash", "AuditTrail"]


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_entry_hash(prev_hash: str, payload_hash: str, created_at: str) -> str:
    return sha256_hex(f"{prev_hash}|{payload_hash}|{created_at}")


class _AuditRepository(Protocol):
    async def last_audit_hash(self) -> str: ...

    async def insert_audit(
        self,
        *,
        trace_id: str,
        actor: str,
        action: str,
        payload_hash: str,
        prev_hash: str,
        entry_hash: str,
    ) -> None: ...


class AuditTrail:
    def __init__(self, repository: _AuditRepository) -> None:
        self.repository = repository
        # 单进程内串行化：避免并发请求同时读到同一个 prev_hash 导致链分叉。
        # 多副本部署需改用数据库行锁或独立审计服务（见 docs/02）。
        self._lock = asyncio.Lock()

    async def record(
        self,
        *,
        trace_id: str,
        actor: str,
        action: str,
        payload: dict[str, Any],
    ) -> str:
        from datetime import datetime, timezone

        async with self._lock:
            created_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
            payload_hash = sha256_hex(canonical_json(payload))
            prev_hash = await self.repository.last_audit_hash()
            entry_hash = compute_entry_hash(prev_hash, payload_hash, created_at)
            await self.repository.insert_audit(
                trace_id=trace_id,
                actor=actor,
                action=action,
                payload_hash=payload_hash,
                prev_hash=prev_hash,
                entry_hash=entry_hash,
            )
        return entry_hash

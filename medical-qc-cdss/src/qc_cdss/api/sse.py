"""SSE（Server-Sent Events）工具。

选 SSE 而不是 WebSocket：质控是"客户端提交一次、服务端持续推送"的单向流，
SSE 天然匹配，且能穿透大多数医院内网代理；断线重连由浏览器 EventSource 负责。

事件协议（前端契约）：
- ``job.started``：任务开始，含 job_id/trace_id
- ``stage.started`` / ``stage.completed``：阶段进度（prepare/rules/llm_qc/icd/finalize）
- ``finding``：单条质控问题（到达即渲染，不必等全部完成）
- ``icd``：单条编码建议
- ``summary``：评分与摘要
- ``result``：完整结果（与同步接口一致）
- ``error``：可恢复错误
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

__all__ = ["sse_event", "SSEBroker"]


def sse_event(event: str, data: dict[str, Any], *, event_id: str | None = None) -> str:
    lines: list[str] = []
    if event_id:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    payload = json.dumps(data, ensure_ascii=False, default=str)
    # 单行 JSON，避免多行 data 被部分代理错误切分
    lines.append(f"data: {payload}")
    return "\n".join(lines) + "\n\n"


class SSEBroker:
    """把 pipeline 的事件回调转成 SSE 流。"""

    def __init__(self, heartbeat_s: float = 15.0) -> None:
        self._queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()
        self.heartbeat_s = heartbeat_s

    async def emit(self, event: str, data: dict[str, Any]) -> None:
        await self._queue.put((event, data))

    async def close(self) -> None:
        await self._queue.put(None)

    async def stream(self) -> AsyncIterator[str]:
        while True:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=self.heartbeat_s)
            except TimeoutError:
                # 心跳注释帧，防止 Nginx/负载均衡器因空闲断开
                yield ": ping\n\n"
                continue
            if item is None:
                break
            event, data = item
            yield sse_event(event, data)

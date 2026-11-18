"""请求级并发控制与超时预算。

私有化场景的典型困境：GPU 只有 1-2 张卡，但门诊高峰有成百份病历同时提交。
本模块解决三件事：

1. **长短请求隔离**：短病历（<800 字）走保留槽位，避免被长病历分片推理饿死。
   这是"短请求优先"的工程实现——不是真的优先级队列，而是容量预留，简单可靠。
2. **分支超时**：单个检查项/分片有独立超时，超时即降级（返回规则结果），
   不让一个慢请求拖垮整份报告。
3. **全局截止时间**：从请求进入开始计时，超时后不再发起新的模型调用，
   保证 P99 可控。
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Awaitable, AsyncIterator, Sequence

__all__ = [
    "DeadlineExceeded",
    "Deadline",
    "ConcurrencyController",
    "gather_bounded",
    "is_long_record",
]


class DeadlineExceeded(TimeoutError):
    """全局截止时间已到。"""


@dataclass
class Deadline:
    budget_s: float

    def __post_init__(self) -> None:
        self._started = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._started

    @property
    def remaining(self) -> float:
        return max(0.0, self.budget_s - self.elapsed)

    @property
    def expired(self) -> bool:
        return self.remaining <= 0.0

    def check(self, stage: str = "") -> None:
        if self.expired:
            raise DeadlineExceeded(f"请求超时（{stage or 'unknown'}）")


class ConcurrencyController:
    def __init__(self, max_concurrency: int = 16, short_slots: int = 6) -> None:
        max_concurrency = max(1, max_concurrency)
        short_slots = min(max(1, short_slots), max_concurrency)
        # 短请求预留 short_slots 个槽位，长请求只能用剩余容量
        self.short_slots = short_slots
        self._short = asyncio.Semaphore(short_slots)
        self._long = asyncio.Semaphore(max(1, max_concurrency - short_slots))

    @asynccontextmanager
    async def slot(self, *, long_request: bool) -> AsyncIterator[None]:
        semaphore = self._long if long_request else self._short
        async with semaphore:
            yield

    @property
    def available(self) -> tuple[int, int]:
        return self._short._value, self._long._value  # noqa: SLF001 - 仅用于健康检查


def is_long_record(text: str, threshold: int = 800) -> bool:
    return len(text) >= threshold


async def gather_bounded(
    coros: Sequence[Awaitable[Any]],
    limit: int,
    *,
    timeout: float | None = None,
) -> list[Any]:
    """有界并发执行，返回结果或异常对象（不抛出，由调用方逐项降级）。"""
    if not coros:
        return []
    semaphore = asyncio.Semaphore(max(1, limit))

    async def _run(coro: Awaitable[Any]) -> Any:
        async with semaphore:
            try:
                if timeout is not None:
                    return await asyncio.wait_for(coro, timeout=timeout)
                return await coro
            except Exception as exc:  # noqa: BLE001 - 逐项隔离失败
                return exc

    return await asyncio.gather(*(_run(coro) for coro in coros))

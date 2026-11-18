"""LLM 抽象：响应对象、错误类型与统一协议。

三种实现（vLLM / 云 API / Mock）遵循同一协议，上层流水线不感知差异。
这是"私有化与混合云配置切换"能成立的前提。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol, Sequence


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str = "unknown"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    ttft_ms: float | None = None
    attempts: int = 1
    degraded: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMError(RuntimeError):
    """模型调用失败（网络、超时、5xx、限流）。"""

    def __init__(self, message: str, *, retryable: bool = True, status: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status = status


Message = dict[str, str]


class LLMClient(Protocol):
    name: str

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        schema: Any | None = None,
        timeout: float | None = None,
        stage: str = "chat",
    ) -> LLMResponse: ...

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        stage: str = "chat",
    ) -> AsyncIterator[str]: ...

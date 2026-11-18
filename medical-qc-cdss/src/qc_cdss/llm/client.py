"""OpenAI 兼容推理客户端。

为什么统一到 OpenAI 协议
------------------------
vLLM、SGLang、DeepSeek、通义、智谱都提供 ``/v1/chat/completions`` 兼容端点。
统一协议后：
- 私有化（vLLM）与云 API 只差一个 ``base_url``，无需两套适配代码；
- 评测、压测、灰度可以复用同一套脚本；
- 未来替换推理引擎（vLLM -> SGLang/TensorRT-LLM）对上层零改动。

工程细节：
1. **自己实现重试**而不是用 SDK 内置重试——需要在指标里区分"第几次尝试"，
   并区分可重试（429/5xx/超时）与不可重试（400/401）。
2. **结构化输出优先用受约束解码**（vLLM guided decoding / json_schema），
   从源头消除格式错误；不支持时退回 ``json_object``。
3. **超时分级**：单次调用超时 + 上层分支超时，二者独立控制。
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, AsyncIterator, Sequence

from ..metrics import LLM_CALLS, LLM_TTFT
from .base import LLMError, LLMResponse, Message

__all__ = ["OpenAICompatLLM", "FallbackLLM", "build_llm"]

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _usage_value(usage: Any, name: str) -> int:
    if usage is None:
        return 0
    return int(getattr(usage, name, 0) or 0)


class OpenAICompatLLM:
    """基于 openai SDK 的异步客户端（兼容 vLLM / 云厂商）。"""

    name = "openai-compatible"

    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        max_retries: int = 2,
        json_mode: str = "guided",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        client: Any | None = None,
    ) -> None:
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key or "EMPTY"
        self.model = model
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.json_mode = json_mode
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:  # pragma: no cover - 需要真实/模拟 SDK
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=0,
            )
        return self._client

    def _request_kwargs(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None,
        max_tokens: int | None,
        schema: Any | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if schema is None:
            return kwargs

        schema_json = schema.model_json_schema()
        if self.json_mode == "guided":
            # vLLM >= 0.6.4 支持 response_format=json_schema，内部走 xgrammar 受约束解码
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema_json, "strict": True},
            }
        elif self.json_mode == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        # prompt_only: 仅依赖提示词约束
        return kwargs

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        schema: Any | None = None,
        timeout: float | None = None,
        stage: str = "chat",
    ) -> LLMResponse:
        kwargs = self._request_kwargs(
            messages, temperature=temperature, max_tokens=max_tokens, schema=schema
        )
        started = time.perf_counter()
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 2):
            try:
                response = await self.client.chat.completions.create(
                    **kwargs, timeout=timeout or self.timeout
                )
                text = (response.choices[0].message.content or "") if response.choices else ""
                latency = (time.perf_counter() - started) * 1000
                usage = getattr(response, "usage", None)
                LLM_CALLS.labels(provider=self.provider, stage=stage, status="ok").inc()
                return LLMResponse(
                    text=text,
                    model=getattr(response, "model", self.model),
                    provider=self.provider,
                    prompt_tokens=_usage_value(usage, "prompt_tokens"),
                    completion_tokens=_usage_value(usage, "completion_tokens"),
                    latency_ms=latency,
                    attempts=attempt,
                )
            except Exception as exc:  # noqa: BLE001 - 需要按错误类型判定是否重试
                last_error = exc
                LLM_CALLS.labels(provider=self.provider, stage=stage, status="error").inc()
                if not self._retryable(exc) or attempt > self.max_retries:
                    break
                backoff = min(0.4 * (2 ** (attempt - 1)), 3.0) + random.uniform(0, 0.2)
                await asyncio.sleep(backoff)

        raise LLMError(
            f"{self.provider} 模型调用失败: {last_error}",
            retryable=self._retryable(last_error),
            status=getattr(last_error, "status_code", None),
        )

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        stage: str = "chat_stream",
    ) -> AsyncIterator[str]:
        kwargs = self._request_kwargs(
            messages, temperature=temperature, max_tokens=max_tokens, schema=None
        )
        started = time.perf_counter()
        first_token_at: float | None = None
        try:
            stream = await self.client.chat.completions.create(
                **kwargs, stream=True, timeout=timeout or self.timeout
            )
            async for chunk in stream:
                if not getattr(chunk, "choices", None):
                    continue
                delta = chunk.choices[0].delta.content
                if not delta:
                    continue
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                    LLM_TTFT.labels(provider=self.provider).observe(first_token_at - started)
                yield delta
            LLM_CALLS.labels(provider=self.provider, stage=stage, status="ok").inc()
        except Exception as exc:  # noqa: BLE001
            LLM_CALLS.labels(provider=self.provider, stage=stage, status="error").inc()
            raise LLMError(f"{self.provider} 流式调用失败: {exc}") from exc

    @staticmethod
    def _retryable(exc: Exception | None) -> bool:
        if exc is None:
            return False
        status = getattr(exc, "status_code", None)
        if isinstance(status, int) and status in _RETRYABLE_STATUS:
            return True
        return exc.__class__.__name__ in {
            "APIConnectionError",
            "APITimeoutError",
            "InternalServerError",
            "RateLimitError",
            "ConnectError",
            "ReadTimeout",
        }


class FallbackLLM:
    """主备降级：私有化 vLLM 故障时切到云 API（或 mock），保证接口不 5xx。"""

    name = "fallback"

    def __init__(self, primary: Any, secondary: Any) -> None:
        self.primary = primary
        self.secondary = secondary
        self.model = getattr(primary, "model", "")

    async def chat(self, messages, **kwargs) -> LLMResponse:
        try:
            return await self.primary.chat(messages, **kwargs)
        except LLMError:
            response = await self.secondary.chat(messages, **kwargs)
            response.degraded = True
            response.metadata["fallback_from"] = getattr(self.primary, "provider", self.primary.name)
            return response

    async def stream(self, messages, **kwargs) -> AsyncIterator[str]:
        produced = False
        try:
            async for chunk in self.primary.stream(messages, **kwargs):
                produced = True
                yield chunk
            return
        except LLMError:
            if produced:
                raise
        async for chunk in self.secondary.stream(messages, **kwargs):
            yield chunk


def build_llm(settings: Any) -> Any:
    """按配置构建客户端（mock / vLLM / 云 API + 可选降级）。"""
    from .mock import MockLLM

    if settings.llm_provider == "mock":
        return MockLLM(model=settings.llm_model)

    primary = OpenAICompatLLM(
        provider=settings.llm_provider,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value(),
        model=settings.llm_model,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
        json_mode=settings.llm_json_mode,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )

    fallback_provider = settings.llm_fallback_provider
    if fallback_provider == "none":
        return primary
    if fallback_provider == "mock":
        return FallbackLLM(primary, MockLLM(model="mock-fallback"))

    fallback_key = (
        settings.llm_fallback_api_key.get_secret_value()
        if settings.llm_fallback_api_key
        else "EMPTY"
    )
    secondary = OpenAICompatLLM(
        provider="cloud",
        base_url=settings.llm_fallback_base_url or "https://api.deepseek.com/v1",
        api_key=fallback_key,
        model=settings.llm_fallback_model or "deepseek-chat",
        timeout=settings.llm_timeout_s,
        max_retries=1,
        json_mode="json_object",
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    return FallbackLLM(primary, secondary)

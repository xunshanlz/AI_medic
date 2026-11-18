"""结构化输出三层治理。

医疗场景对"输出不可解析"的容忍度是零：一次 JSON 解析失败不能让整份病历
质控失败。因此设计三层：

1. **受约束生成 + 纠错重试**：首选 guided decoding（vLLM 内 xgrammar）从源头
   约束 JSON Schema；若仍失败，把校验错误回灌给模型重写（最多 N 次）。
2. **字段级抢救**：整体校验失败时，逐条校验数组元素，丢弃坏项、保留好项，
   再用默认值补齐可选字段——避免"一条脏数据毁掉整份报告"。
3. **降级标记**：仍失败则返回 ``value=None``，由上层标记 ``degraded`` 并转人工，
   接口依然 200。

这条链路同时是可观测的：每次重试都打点（schema + 失败原因），便于发现
提示词或模型版本回退。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Sequence, TypeVar, get_args, get_origin

from pydantic import BaseModel, ValidationError

from ..logging import get_logger
from ..metrics import STRUCTURED_RETRIES
from .base import LLMClient
from .jsonx import loads_lenient

__all__ = ["StructuredOutcome", "StructuredGenerator"]

T = TypeVar("T", bound=BaseModel)
logger = get_logger(__name__)


@dataclass
class StructuredOutcome(Generic[T]):
    value: T | None
    raw: str = ""
    attempts: int = 0
    degraded: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.value is not None


class StructuredGenerator:
    """把"LLM 文本"变成"可校验对象"的唯一入口。"""

    def __init__(self, llm: LLMClient, *, max_attempts: int = 2) -> None:
        self.llm = llm
        self.max_attempts = max(1, max_attempts)

    async def generate(
        self,
        schema: type[T],
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        stage: str = "structured",
    ) -> StructuredOutcome[T]:
        conversation: list[dict[str, str]] = list(messages)
        errors: list[str] = []
        last_raw = ""

        for attempt in range(1, self.max_attempts + 1):
            response = await self.llm.chat(
                conversation,
                temperature=temperature,
                max_tokens=max_tokens,
                schema=schema,
                timeout=timeout,
                stage=stage,
            )
            last_raw = response.text
            try:
                data = loads_lenient(response.text)
                return StructuredOutcome(
                    value=schema.model_validate(data),
                    raw=response.text,
                    attempts=attempt,
                )
            except (ValueError, ValidationError) as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                errors.append(error_text)
                STRUCTURED_RETRIES.labels(
                    schema.__name__, "validation" if isinstance(exc, ValidationError) else "json"
                ).inc()
                logger.warning(
                    "structured_output_invalid",
                    schema=schema.__name__,
                    attempt=attempt,
                    error=error_text[:300],
                )
                if attempt >= self.max_attempts:
                    break
                conversation = list(messages) + [
                    {"role": "assistant", "content": response.text[:4000]},
                    {
                        "role": "user",
                        "content": (
                            "上面的输出不符合要求的 JSON 结构，校验错误如下：\n"
                            f"{error_text[:800]}\n"
                            "请只输出修正后的 JSON，不要输出任何解释、markdown 围栏或多余文本。"
                        ),
                    },
                ]

        salvaged = self._salvage(schema, last_raw)
        if salvaged is not None:
            logger.warning("structured_output_salvaged", schema=schema.__name__)
            return StructuredOutcome(
                value=salvaged,
                raw=last_raw,
                attempts=self.max_attempts,
                degraded=True,
                errors=errors,
            )

        logger.error("structured_output_failed", schema=schema.__name__, errors=errors[-1:])
        return StructuredOutcome(
            value=None,
            raw=last_raw,
            attempts=self.max_attempts,
            degraded=True,
            errors=errors,
        )

    # ------------------------------------------------------------------ 第二层
    def _salvage(self, schema: type[T], text: str) -> T | None:
        try:
            data = loads_lenient(text)
        except ValueError:
            return None
        if not isinstance(data, dict):
            return None

        try:
            return schema.model_validate(data)
        except ValidationError:
            pass

        patched = dict(data)
        for name, field_info in schema.model_fields.items():
            if name not in patched:
                continue
            annotation = field_info.annotation
            if get_origin(annotation) is not list:
                continue
            args = get_args(annotation)
            if not args:
                continue
            item_type = args[0]
            if not (isinstance(item_type, type) and issubclass(item_type, BaseModel)):
                continue
            items = patched[name]
            if not isinstance(items, list):
                continue
            kept: list[dict[str, Any]] = []
            for item in items:
                try:
                    kept.append(item_type.model_validate(item).model_dump())
                except (ValidationError, TypeError):
                    continue
            patched[name] = kept

        for name, field_info in schema.model_fields.items():
            if field_info.is_required() and name not in patched:
                return None

        try:
            return schema.model_validate(patched)
        except ValidationError:
            return None

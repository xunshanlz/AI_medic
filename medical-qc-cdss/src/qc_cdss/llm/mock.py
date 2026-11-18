"""离线 Mock 模型。

用途：
1. CI / 无 GPU 环境跑通完整链路（``QC_LLM_PROVIDER=mock``）；
2. 演示与压测基线：排除模型波动，单独测量调度、检索、校验、持久化的开销。

它不是"假数据生成器"：输出严格遵循与真实模型相同的 JSON 契约，引文也确实
来自提示词中的病历原文，因此经过同一套结构化解析与溯源校验，能暴露真实
链路上的问题（例如契约漂移、offset 错位）。

注意：Mock 不参与任何效果指标（准确率/召回率）评估，评测必须使用真实模型。
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, AsyncIterator, Sequence

from .base import LLMResponse, Message

__all__ = ["MockLLM"]

_SECTION_RE = re.compile(r"<<<SECTION [^>]*>>>\n(.*?)\n<<<END>>>", re.S)
_CANDIDATE_RE = re.compile(r"^([A-Z]\d{2}(?:\.\d+)?)\t(.+?)(?:\t.*)?$", re.M)


def _first_sentence(text: str, limit: int = 40) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    for sep in ("。", "；", "\n"):
        idx = cleaned.find(sep)
        if 0 < idx <= limit:
            return cleaned[: idx + 1]
    return cleaned[:limit]


class MockLLM:
    name = "mock"

    def __init__(self, model: str = "mock-qwen", latency_ms: float = 8.0) -> None:
        self.model = model
        self._latency = latency_ms / 1000

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
        started = time.perf_counter()
        await asyncio.sleep(self._latency)
        prompt = "\n".join(m.get("content", "") for m in messages)
        payload = self._payload(schema, prompt)
        text = json.dumps(payload, ensure_ascii=False)
        return LLMResponse(
            text=text,
            model=self.model,
            provider="mock",
            prompt_tokens=len(prompt) // 2,
            completion_tokens=len(text) // 2,
            latency_ms=(time.perf_counter() - started) * 1000,
            ttft_ms=self._latency * 1000,
            metadata={"schema": getattr(schema, "__name__", None), "stage": stage},
        )

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        stage: str = "chat",
    ) -> AsyncIterator[str]:
        prompt = "\n".join(m.get("content", "") for m in messages)
        text = json.dumps(self._payload(None, prompt), ensure_ascii=False)
        for i in range(0, len(text), 24):
            await asyncio.sleep(0)
            yield text[i : i + 24]

    # ------------------------------------------------------------------ 内部
    def _payload(self, schema: Any, prompt: str) -> dict[str, Any]:
        name = getattr(schema, "__name__", "")
        sections = _SECTION_RE.findall(prompt)

        if name == "QcItemVerdict":
            # 仅对部分检查项报问题，模拟真实模型"稀疏命中"的行为
            code_match = re.search(r"【检查项】(\S+)", prompt)
            code = code_match.group(1) if code_match else ""
            if code.endswith(("01", "05")) and sections and len(sections[0].strip()) > 8:
                quote = _first_sentence(sections[0])
                return {
                    "passed": False,
                    "reasoning": "（mock）原文表述与检查项要求存在不一致，建议人工复核。",
                    "findings": [
                        {
                            "title": "病历描述前后需核对",
                            "detail": "（mock 演示）该段落的症状描述与诊断依据之间存在信息缺口，"
                            "建议补充阴性体征与检查结果。",
                            "suggestion": "补充相关阴性症状与辅助检查结果，保持前后一致。",
                            "quote": quote,
                            "confidence": 0.82,
                            "severity_hint": "medium",
                        }
                    ],
                }
            return {"passed": True, "reasoning": "（mock）未发现该检查项问题。", "findings": []}

        if name == "IcdVerdict":
            candidates = _CANDIDATE_RE.findall(prompt)
            if not candidates:
                return {"suggestions": []}
            top = candidates[:3]
            digest = sections[0] if sections else ""
            return {
                "suggestions": [
                    {
                        "code": code,
                        "confidence": round(0.82 - index * 0.14, 2),
                        "rationale": f"（mock）主要诊断与 {name_cn} 的编码规则匹配。",
                        "quote": _first_sentence(digest, 30),
                    }
                    for index, (code, name_cn) in enumerate(top)
                ]
            }

        if name == "CdssVerdict":
            if not sections:
                return {"advice": [], "insufficient_evidence": True}
            evidence = _first_sentence(sections[0], 50)
            return {
                "advice": [
                    {
                        "advice_type": "exam",
                        "content": "结合指南证据，建议完善相关检查并评估病情严重程度。",
                        "rationale": "（mock）依据知识库片段给出的通用建议。",
                        "evidence_quote": evidence,
                        "confidence": 0.7,
                    }
                ],
                "insufficient_evidence": False,
            }

        return {"content": "（mock）已收到请求。"}

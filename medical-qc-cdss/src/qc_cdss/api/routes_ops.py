"""运维接口：存活/就绪探针与运行时元信息。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ..config import Settings
from .deps import AppContainer, get_container

__all__ = ["router"]

router = APIRouter(tags=["运维"])


@router.get("/healthz", summary="存活探针")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz", summary="就绪探针")
async def readyz(request: Request) -> dict[str, Any]:
    container: AppContainer | None = getattr(request.app.state, "container", None)
    if container is None or not container.ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="未就绪")
    return {
        "status": "ready",
        "knowledge_chunks": len(container.retriever.chunks),
        "icd_entries": len(container.catalog),
        "startup_errors": container.startup_errors,
    }


@router.get("/v1/meta", summary="运行时元信息（模型/提示词/存储版本）")
async def meta(
    container: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    settings: Settings = container.settings
    return {
        "app": settings.app_name,
        "env": settings.env,
        "llm_provider": settings.llm_provider,
        "model": getattr(container.llm, "model", ""),
        "prompt_version": settings.prompt_version,
        "checklist_version": settings.checklist_version,
        "embedding_provider": settings.embedding_provider,
        "vector_store": settings.vector_store,
        "knowledge_chunks": len(container.retriever.chunks),
        "icd_entries": len(container.catalog),
        "degraded_capabilities": container.startup_errors,
    }

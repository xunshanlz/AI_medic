"""FastAPI 应用装配。

启动顺序（lifespan）：
1. 初始化日志与数据库表（幂等 create_all，生产用 Alembic 迁移）；
2. 构建依赖容器：LLM 客户端、知识库索引、ICD 目录、缓存、仓储；
3. 暴露 ``/readyz`` 供 K8s/负载均衡判断是否可接入流量。

关闭顺序相反，确保 GPU 连接、Redis、数据库连接干净释放。
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from ..config import Settings, get_settings
from ..logging import bind_trace, configure_logging, get_logger
from ..metrics import render_metrics
from . import routes_cdss, routes_ops, routes_qc
from .deps import build_container

__all__ = ["create_app", "app", "main"]

logger = get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging("DEBUG" if settings.debug else "INFO", json_output=settings.env != "dev")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = await build_container(settings)
        app.state.container = container
        logger.info(
            "startup_complete",
            env=settings.env,
            model=getattr(container.llm, "model", ""),
            llm_provider=settings.llm_provider,
            knowledge_chunks=len(container.retriever.chunks),
            icd_entries=len(container.catalog),
            startup_errors=container.startup_errors,
        )
        try:
            yield
        finally:
            await container.close()
            app.state.container = None
            logger.info("shutdown_complete")

    app = FastAPI(
        title="QC-CDSS 病历内涵质控与临床决策支持平台",
        description=(
            "面向三甲医院私有化场景：规则前置 + 大模型内涵质控 + 确定性证据溯源 + "
            "ICD 编码推荐 + 指南证据驱动的 CDSS。"
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    app.state.settings = settings
    app.state.container = None

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def trace_middleware(request: Request, call_next: Any) -> Response:
        trace_id = request.headers.get("X-Trace-Id") or ""
        trace_id = bind_trace(trace_id or None)
        request.state.trace_id = trace_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "http_request",
                method=request.method,
                path=request.url.path,
                elapsed_ms=round(elapsed_ms, 2),
            )
        response.headers["X-Trace-Id"] = trace_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "code": "VALIDATION_ERROR",
                "message": "请求参数校验失败",
                # pydantic v2 的 ctx 可能含异常对象，必须经 jsonable_encoder 转换
                "detail": jsonable_encoder(exc.errors()),
                "trace_id": getattr(request.state, "trace_id", ""),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_exception", path=request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "code": "INTERNAL_ERROR",
                "message": "服务内部错误，请稍后重试",
                "trace_id": getattr(request.state, "trace_id", ""),
            },
        )

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        payload, content_type = render_metrics()
        return Response(content=payload, media_type=content_type)

    app.include_router(routes_ops.router)
    app.include_router(routes_qc.router, prefix=settings.api_prefix)
    app.include_router(routes_cdss.router, prefix=settings.api_prefix)
    return app


app = create_app()


def main() -> None:
    """``qc-cdss`` 命令入口（等价于 uvicorn 启动）。"""
    import uvicorn

    uvicorn.run("qc_cdss.api.app:app", host="0.0.0.0", port=8000, log_config=None)

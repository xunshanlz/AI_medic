"""应用入口：用 create_app() 工厂组装应用。

为什么用工厂函数而不是直接写模块级 app：

1. 测试可以造「另一套配置的 app」，不必受环境变量影响；
2. 将来要多 worker 部署、或者同一进程里跑两套配置时，工厂更灵活；
3. uvicorn 仍然可以通过 `app.main:app` 找到默认实例，用法不变。
"""

import logging
import time

from fastapi import FastAPI

from app.api.health import router as health_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging

log = logging.getLogger("app")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="AI 病历质控与 CDSS 智能决策应用",
        description="推理服务层 + 结构化治理 + 确定性溯源校验（教学版逐步搭建）",
        version=settings.app_version,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )

    # 把配置和启动时刻挂到 app.state：路由里通过 request.app.state 读取，
    # 不搞全局变量，测试时互不干扰。
    app.state.settings = settings
    app.state.started_at = time.monotonic()

    app.include_router(health_router)

    @app.get("/", include_in_schema=False)
    async def index() -> dict:
        return {
            "service": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs" if settings.docs_enabled else None,
            "endpoints": ["/healthz", "/readyz"],
        }

    log.info("app created: env=%s version=%s", settings.env, settings.app_version)
    return app


# uvicorn 的入口点：uvicorn app.main:app
app = create_app()

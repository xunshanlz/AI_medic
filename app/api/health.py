"""健康检查与就绪探针。

为什么探针要分两个（对应项目文档 5.1.2 与 9.1 的 healthcheck）：

- /healthz 存活探针（liveness）：只回答「进程还活着吗」。
  它不检查数据库、不检查 vLLM —— 因为下游挂了不代表本进程该被重启。
  如果让 healthz 去连 Redis，Redis 一抖，容器就会被反复重启，越重启越糟。

- /readyz 就绪探针（readiness）：回答「现在能接业务请求吗」。
  它检查关键依赖；不 ready 时返回 503，让 Nginx 把流量摘走，而不是给用户报错。

这是两个不同的故障处置动作，所以不能合成一个接口。
"""

import logging
import time
from datetime import datetime

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/healthz", summary="存活探针（不检查下游）")
async def healthz(request: Request) -> dict:
    settings = request.app.state.settings
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
        "env": settings.env,
        "server_time": datetime.now().astimezone().isoformat(timespec="milliseconds"),
    }


@router.get("/readyz", summary="就绪探针（检查关键依赖）")
async def readyz(request: Request) -> JSONResponse:
    settings = request.app.state.settings
    started_at: float = request.app.state.started_at

    # 阶段 01 只有「配置已加载」这一项。
    # 阶段 02 加 llm；阶段 07/08 加 redis / mysql，每项都是一个异步可达性检查。
    checks: dict[str, str] = {"config": "ok"}

    ready = all(v == "ok" for v in checks.values())
    body = {
        "status": "ok" if ready else "unready",
        "service": settings.app_name,
        "version": settings.app_version,
        "checks": checks,
        "uptime_s": round(time.monotonic() - started_at, 1),
    }
    if not ready:
        log.warning("readyz not ready: %s", checks)
    return JSONResponse(body, status_code=200 if ready else 503)

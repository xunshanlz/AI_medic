"""结构化日志与 trace 上下文。

生产要求：每条日志都能按 ``trace_id`` 串起一次质控请求的全部阶段（含模型调用、
证据校验、降级），便于排障与审计。使用 contextvars 传递 trace，不需要在每层
函数签名里加参数。
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

__all__ = [
    "trace_id_var",
    "new_trace_id",
    "configure_logging",
    "get_logger",
    "bind_trace",
]

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")

try:  # pragma: no cover - 依赖是否安装由运行环境决定
    import structlog

    _HAS_STRUCTLOG = True
except Exception:  # pragma: no cover
    structlog = None  # type: ignore[assignment]
    _HAS_STRUCTLOG = False


def new_trace_id() -> str:
    return uuid.uuid4().hex


class _TraceFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        record.trace_id = trace_id_var.get()
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "trace_id": getattr(record, "trace_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO", *, json_output: bool = True) -> None:
    """初始化日志。structlog 存在时走结构化渲染，否则退回 stdlib JSON。"""
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_TraceFilter())
    if json_output:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(trace_id)s] %(name)s: %(message)s"))

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    if _HAS_STRUCTLOG:  # pragma: no cover
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                structlog.processors.JSONRenderer(ensure_ascii=False),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, level.upper(), logging.INFO)
            ),
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )


class _StdlibAdapter:
    """让标准库 logger 接受 structlog 风格的 ``key=value`` 字段。

    未安装 structlog（或显式关闭）时，业务代码里的
    ``logger.info("event", job_id=...)`` 不应抛 TypeError。
    """

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    def _emit(self, level: int, event: str, exc_info: bool = False, **fields: Any) -> None:
        self._logger.log(level, event, extra={"extra_fields": fields}, exc_info=exc_info)

    def debug(self, event: str, **fields: Any) -> None:
        self._emit(logging.DEBUG, event, **fields)

    def info(self, event: str, **fields: Any) -> None:
        self._emit(logging.INFO, event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._emit(logging.WARNING, event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self._emit(logging.ERROR, event, **fields)

    def exception(self, event: str, **fields: Any) -> None:
        self._emit(logging.ERROR, event, exc_info=True, **fields)

    def bind(self, **fields: Any) -> "_StdlibAdapter":
        return self


def get_logger(name: str = "qc_cdss"):
    if _HAS_STRUCTLOG:  # pragma: no cover
        return structlog.get_logger(name)
    return _StdlibAdapter(name)


def bind_trace(trace_id: str | None = None) -> str:
    """设置当前上下文 trace_id，返回最终使用的值。"""
    value = trace_id or new_trace_id()
    trace_id_var.set(value)
    return value

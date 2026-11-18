"""异步数据库会话。"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from ..config import Settings
from .models import Base

__all__ = ["build_engine", "build_session_factory", "init_models"]


def build_engine(settings: Settings) -> AsyncEngine:
    url = settings.database_url
    kwargs: dict[str, object] = {"echo": settings.db_echo, "pool_pre_ping": True}

    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        # sqlite+aiosqlite:///./data/qc_cdss.db -> 确保目录存在
        if ":///" in url:
            raw_path = url.split(":///", 1)[1]
            if raw_path and raw_path != ":memory:":
                Path(raw_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    return create_async_engine(url, **kwargs)


def build_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


async def init_models(engine: AsyncEngine) -> None:
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

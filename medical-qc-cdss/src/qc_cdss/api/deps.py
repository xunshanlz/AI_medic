"""依赖容器：服务启动时构建一次，请求间共享。

容器里持有的都是有状态/重资源对象（向量库、模型客户端、缓存、数据库连接池）。
请求级对象（Deadline、SSE 队列）不放这里，避免并发串扰。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..audit import AuditTrail
from ..cache import Cache, build_cache
from ..config import Settings, get_settings
from ..db.repo import QcRepository
from ..db.session import build_engine, build_session_factory, init_models
from ..domain.evidence import EvidenceVerifier
from ..domain.rules import RuleEngine
from ..llm.client import build_llm
from ..llm.structured import StructuredGenerator
from ..logging import get_logger
from ..pipeline.graph import QcPipeline
from ..pipeline.icd import IcdCatalog, IcdRecommender
from ..pipeline.nodes import NodeDeps
from ..rag.bm25 import BM25Index
from ..rag.embedding import build_embedder
from ..rag.retriever import Retriever
from ..rag.store import build_vector_store

__all__ = ["AppContainer", "build_container", "get_container", "require_auth"]

logger = get_logger(__name__)
_bearer = HTTPBearer(auto_error=False)


@dataclass
class AppContainer:
    settings: Settings
    llm: Any
    generator: StructuredGenerator
    verifier: EvidenceVerifier
    rules: RuleEngine
    retriever: Retriever
    catalog: IcdCatalog
    icd: IcdRecommender | None
    pipeline: QcPipeline
    cache: Cache
    repo: QcRepository
    audit: AuditTrail
    engine: Any
    session_factory: Any
    ready: bool = False
    startup_errors: list[str] = field(default_factory=list)

    async def close(self) -> None:
        await self.retriever.close()
        await self.cache.close()
        await self.engine.dispose()


async def build_container(settings: Settings | None = None) -> AppContainer:
    settings = settings or get_settings()
    engine = build_engine(settings)
    await init_models(engine)
    session_factory = build_session_factory(engine)

    llm = build_llm(settings)
    generator = StructuredGenerator(llm, max_attempts=max(2, settings.llm_max_retries + 1))
    verifier = EvidenceVerifier(
        accept_threshold=settings.evidence_accept_threshold,
        review_threshold=settings.evidence_review_threshold,
    )
    rules = RuleEngine()
    cache = await build_cache(settings.redis_url, settings.cache_ttl_s)
    repo = QcRepository(session_factory)

    embedder = build_embedder(settings)
    store = build_vector_store(settings, embedder.dim)
    startup_errors: list[str] = []
    try:
        retriever = await Retriever.build(
            settings.guidelines_dir, embedder, store, rrf_k=settings.rrf_k
        )
    except Exception as exc:  # noqa: BLE001 - 知识库不可用不应阻断质控
        logger.error("retriever_build_failed", error=str(exc))
        startup_errors.append(f"retriever:{type(exc).__name__}")
        retriever = Retriever([], embedder, store, BM25Index())

    catalog = IcdCatalog.load(settings.catalog_path)
    icd = IcdRecommender(catalog, generator, verifier) if len(catalog) else None
    if icd is None:
        startup_errors.append("icd_catalog_empty")

    deps = NodeDeps(
        settings=settings,
        generator=generator,
        verifier=verifier,
        rules=rules,
        icd=icd,
        cache=cache,
    )
    pipeline = QcPipeline(deps)

    return AppContainer(
        settings=settings,
        llm=llm,
        generator=generator,
        verifier=verifier,
        rules=rules,
        retriever=retriever,
        catalog=catalog,
        icd=icd,
        pipeline=pipeline,
        cache=cache,
        repo=repo,
        audit=AuditTrail(repo),
        engine=engine,
        session_factory=session_factory,
        ready=True,
        startup_errors=startup_errors,
    )


def get_container(request: Request) -> AppContainer:
    container: AppContainer | None = getattr(request.app.state, "container", None)
    if container is None or not container.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="服务尚未就绪"
        )
    return container


async def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict[str, Any]:
    """JWT 鉴权。``QC_AUTH_ENABLED=false`` 时放行（内网试点/联调）。"""
    settings = get_settings()
    if not settings.auth_enabled:
        return {"sub": "anonymous", "roles": ["admin"]}

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少 Bearer Token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:  # 延迟导入：未启用鉴权时不需要 PyJWT
        import jwt

        payload = jwt.decode(
            credentials.credentials,
            settings.jwt_secret.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token 无效: {type(exc).__name__}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    return dict(payload)


async def wait_ready(container: AppContainer, timeout: float = 30.0) -> bool:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if container.ready:
            return True
        await asyncio.sleep(0.1)
    return False

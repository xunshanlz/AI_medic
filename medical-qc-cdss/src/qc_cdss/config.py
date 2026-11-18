"""集中配置。

所有可变项走环境变量（``QC_`` 前缀）或 ``.env``，代码里不出现魔法值。
这样同一份镜像可以在"私有化 vLLM"与"云 API"两种形态间切换，而不需要改代码：

    QC_LLM_PROVIDER=vllm   QC_LLM_BASE_URL=http://vllm:8000/v1
    QC_LLM_PROVIDER=cloud  QC_LLM_BASE_URL=https://api.deepseek.com/v1
    QC_LLM_PROVIDER=mock   # 无 GPU / 离线演示

``QC_LLM_FALLBACK_PROVIDER`` 用于公有云主模型故障时自动降级，保证接口可用性。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "get_settings"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="QC_",
        extra="ignore",
    )

    # ---------------------------------------------------------------- 基础
    app_name: str = "qc-cdss"
    env: Literal["dev", "test", "prod"] = "dev"
    debug: bool = False
    api_prefix: str = "/v1"
    data_dir: Path = Path("./data")
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # ---------------------------------------------------------------- LLM
    llm_provider: Literal["vllm", "cloud", "mock"] = "mock"
    llm_base_url: str = "http://127.0.0.1:8000/v1"
    llm_api_key: SecretStr = SecretStr("EMPTY")
    llm_model: str = "Qwen2.5-32B-Instruct"
    llm_fallback_provider: Literal["none", "cloud", "mock"] = "mock"
    llm_fallback_base_url: str | None = None
    llm_fallback_api_key: SecretStr | None = None
    llm_fallback_model: str | None = None
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    # guided: vLLM guided_json（xgrammar）；json_object: OpenAI response_format；
    # prompt_only: 仅靠提示词约束（兼容老版本推理服务）
    llm_json_mode: Literal["guided", "json_object", "prompt_only"] = "guided"

    # ---------------------------------------------------------------- 向量化
    embedding_provider: Literal["openai", "hash"] = "hash"
    embedding_base_url: str | None = None
    embedding_api_key: SecretStr | None = None
    embedding_model: str = "bge-m3"
    embedding_dim: int = 256

    # ---------------------------------------------------------------- 检索
    vector_store: Literal["memory", "milvus", "chroma"] = "memory"
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "qc_knowledge"
    chroma_path: Path = Path("./data/chroma")
    retrieval_top_k: int = 8
    rrf_k: int = 60

    # ---------------------------------------------------------------- 缓存
    redis_url: str | None = None
    cache_ttl_s: int = 6 * 3600

    # ---------------------------------------------------------------- 数据库
    database_url: str = "sqlite+aiosqlite:///./data/qc_cdss.db"
    db_echo: bool = False

    # ---------------------------------------------------------------- 并发与超时
    max_concurrency: int = 16
    short_request_slots: int = 6
    request_deadline_s: float = 45.0
    branch_timeout_s: float = 25.0
    max_shards: int = 4
    shard_chars: int = 1200

    # ---------------------------------------------------------------- 证据校验
    evidence_accept_threshold: float = 88.0
    evidence_review_threshold: float = 70.0

    # ---------------------------------------------------------------- 鉴权
    auth_enabled: bool = False
    jwt_secret: SecretStr = SecretStr("dev-only-change-me")
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 120

    # ---------------------------------------------------------------- 隐私
    pii_masking: bool = True

    # ---------------------------------------------------------------- 版本
    prompt_version: str = "2026.11.1"
    checklist_version: str = "v1"

    @property
    def is_mock(self) -> bool:
        return self.llm_provider == "mock"

    @property
    def guidelines_dir(self) -> Path:
        return self.data_dir / "guidelines"

    @property
    def catalog_path(self) -> Path:
        return self.data_dir / "catalog" / "icd10.jsonl"

    @property
    def goldset_path(self) -> Path:
        return self.data_dir / "eval" / "golden.jsonl"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

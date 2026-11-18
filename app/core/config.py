"""配置层：所有可变的运行参数集中在这里，通过环境变量注入。

设计要点（对应项目文档 9.1「环境变量走 .env」）：
1. 配置不硬编码在代码里 —— 换医院、换模型、换端口都不用改代码。
2. 环境变量统一加 CDSS_ 前缀，避免和系统里的 PORT / HOST 之类撞车。
3. 用 pydantic-settings 而不是手写 os.getenv：类型转换与校验免费拿到，
   配错（比如把端口写成字符串）会在启动时就报错，而不是运行到一半才崩。
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置。字段名即环境变量名去掉前缀后的小写形式。"""

    model_config = SettingsConfigDict(
        env_file=".env",          # 本地开发从 .env 读，生产由容器环境变量注入
        env_file_encoding="utf-8",
        env_prefix="CDSS_",        # CDSS_ENV / CDSS_PORT / CDSS_LOG_LEVEL ...
        extra="ignore",            # 进程里还有其他无关环境变量，忽略即可
    )

    # —— 服务身份 ——
    app_name: str = "ai-medic-qc"
    app_version: str = "0.1.0"
    env: Literal["dev", "test", "prod"] = "dev"

    # —— 监听地址 ——
    # 默认只监听 127.0.0.1：开发机上不对局域网暴露；容器内再改成 0.0.0.0
    host: str = "127.0.0.1"
    port: int = 8000

    # —— 可观测 ——
    log_level: str = "INFO"
    docs_enabled: bool = True   # 生产环境建议关掉 /docs，减少暴露面


@lru_cache
def get_settings() -> Settings:
    """进程内单例。配置只在启动时读一次，避免每个请求都读文件。"""
    return Settings()

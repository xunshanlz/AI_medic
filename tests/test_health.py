"""阶段 01 的验收测试：服务骨架能起来、探针语义正确。

跑法：.venv\\Scripts\\python.exe -m pytest
"""

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def make_client() -> TestClient:
    """每个用例用独立 app 实例；_env_file=None 表示不读本机 .env，测试结果可复现。"""
    return TestClient(create_app(Settings(_env_file=None)))


def test_healthz_returns_ok() -> None:
    resp = make_client().get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "ai-medic-qc"


def test_readyz_reports_checks() -> None:
    resp = make_client().get("/readyz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["checks"]["config"] == "ok"


def test_openapi_contains_health_paths() -> None:
    resp = make_client().get("/openapi.json")
    assert resp.status_code == 200
    paths = resp.json()["paths"]
    assert "/healthz" in paths
    assert "/readyz" in paths


def test_unknown_path_returns_404() -> None:
    resp = make_client().get("/v1/not-exist")
    assert resp.status_code == 404


def test_env_var_overrides_default(monkeypatch) -> None:
    """配置来自环境变量：CDSS_ENV=prod 应该覆盖默认的 dev。"""
    monkeypatch.setenv("CDSS_ENV", "prod")
    settings = Settings(_env_file=None)
    assert settings.env == "prod"

"""接口集成测试（需要 fastapi / langgraph 等运行依赖）。

本地无依赖时会自动跳过；在容器内执行 ``pytest`` 会完整运行。
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

try:
    import pytest
except ImportError as exc:  # unittest 发现机制下跳过整个模块
    raise unittest.SkipTest("pytest 未安装，跳过接口集成测试") from exc

pytest.importorskip("fastapi")
pytest.importorskip("langgraph")

from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("db") / "test.db"
    os.environ.update(
        {
            "QC_ENV": "test",
            "QC_LLM_PROVIDER": "mock",
            "QC_VECTOR_STORE": "memory",
            "QC_EMBEDDING_PROVIDER": "hash",
            "QC_DATABASE_URL": f"sqlite+aiosqlite:///{db_path}",
            "QC_REDIS_URL": "",
            "QC_DATA_DIR": str(ROOT / "data"),
            "QC_AUTH_ENABLED": "false",
        }
    )

    from qc_cdss.config import get_settings

    get_settings.cache_clear()
    from qc_cdss.api.app import create_app

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def _sample() -> dict:
    return json.loads(
        (ROOT / "data" / "emr" / "emr_inpatient_defective.json").read_text(encoding="utf-8")
    )


def test_health_and_ready(client):
    assert client.get("/healthz").json()["status"] == "ok"
    payload = client.get("/readyz").json()
    assert payload["status"] == "ready"
    assert payload["knowledge_chunks"] > 0


def test_quality_check_returns_evidence_spans(client):
    response = client.post("/v1/qc/check", json=_sample())
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["score"] <= 100
    assert payload["findings"], "缺陷病历应至少命中一条规则"
    assert any(item["severity"] == "high" for item in payload["findings"])

    quoted = [item for item in payload["findings"] if item.get("evidence")]
    assert quoted, "规则命中应带证据区间"
    for item in quoted:
        evidence = item["evidence"]
        assert evidence["start"] <= evidence["end"]


def test_job_can_be_fetched_and_reviewed(client):
    created = client.post("/v1/qc/check", json=_sample()).json()
    job_id = created["job_id"]

    detail = client.get(f"/v1/qc/jobs/{job_id}")
    assert detail.status_code == 200
    finding_id = detail.json()["result"]["findings"][0]["finding_id"]
    assert finding_id

    review = client.post(
        f"/v1/qc/findings/{finding_id}/review",
        json={"status": "accepted", "reviewer": "dr.zhang", "comment": "确认存在该问题"},
    )
    assert review.status_code == 200
    assert review.json()["ok"] is True

    stats = client.get("/v1/qc/stats").json()
    assert stats["total_jobs"] >= 1
    assert 0.0 <= stats["adoption_rate"] <= 1.0


def test_unknown_job_returns_404(client):
    assert client.get("/v1/qc/jobs/not-exist").status_code == 404


def test_stream_emits_result_event(client):
    events: list[str] = []
    with client.stream("POST", "/v1/qc/check/stream", json=_sample()) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())
    assert "job.started" in events
    assert "result" in events


def test_cdss_advice_is_grounded(client):
    response = client.post(
        "/v1/cdss/advice",
        json={
            "question": "社区获得性肺炎门诊患者抗菌药物如何选择？",
            "diagnoses": ["社区获得性肺炎"],
            "top_k": 5,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["advice"], payload
    assert payload["advice"][0]["evidence"], "建议必须带指南证据"


def test_triage_detects_emergency(client):
    payload = client.post(
        "/v1/cdss/triage", json={"symptoms": "突发胸痛，大汗，呼吸困难"}
    ).json()
    assert payload["level"] == "emergency"


def test_validation_error_shape(client):
    response = client.post("/v1/qc/check", json={"patient_id": "x"})
    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"

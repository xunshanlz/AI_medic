#!/usr/bin/env python
"""端到端联调脚本：对运行中的服务做一次完整冒烟。

覆盖：健康检查 -> 同步质控 -> 结果查询 -> SSE 流式 -> CDSS 建议 -> 急症分诊。

用法::

    python scripts/smoke.py --base-url http://127.0.0.1:8000
    python scripts/smoke.py --sample data/emr/emr_inpatient_defective.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass


def _fail(message: str) -> int:
    print(f"[FAIL] {message}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="QC-CDSS 冒烟测试")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--sample", default=str(ROOT / "data" / "emr" / "emr_inpatient_defective.json")
    )
    parser.add_argument("--skip-stream", action="store_true")
    args = parser.parse_args()

    try:
        import httpx
    except ImportError:
        return _fail("未安装 httpx：pip install httpx")

    base = args.base_url.rstrip("/")
    sample = json.loads(Path(args.sample).read_text(encoding="utf-8"))

    with httpx.Client(timeout=180.0) as client:
        # 1. 探针
        health = client.get(f"{base}/healthz")
        if health.status_code != 200:
            return _fail(f"/healthz 返回 {health.status_code}")
        ready = client.get(f"{base}/readyz")
        if ready.status_code != 200:
            return _fail(f"/readyz 返回 {ready.status_code}: {ready.text[:200]}")
        print(f"[ok] 服务就绪：{ready.json()}")

        # 2. 同步质控
        response = client.post(f"{base}/v1/qc/check", json=sample)
        if response.status_code != 200:
            return _fail(f"质控失败 {response.status_code}: {response.text[:400]}")
        result = response.json()
        print(
            f"[ok] 质控完成：score={result['score']} findings={len(result['findings'])} "
            f"icd={[item['code'] for item in result['icd']]} "
            f"latency={result['latency_ms']}ms status={result['status']}"
        )
        for finding in result["findings"][:6]:
            evidence = finding.get("evidence") or {}
            print(
                f"     - [{finding['severity']}] {finding['code']} {finding['title']} "
                f"evidence={evidence.get('state')} score={evidence.get('score')}"
            )

        # 3. 结果查询
        job_id = result["job_id"]
        detail = client.get(f"{base}/v1/qc/jobs/{job_id}")
        if detail.status_code != 200:
            return _fail(f"任务查询失败 {detail.status_code}")
        print(f"[ok] 任务查询：{job_id} 已落库")

        # 4. SSE
        if not args.skip_stream:
            events: list[str] = []
            with client.stream("POST", f"{base}/v1/qc/check/stream", json=sample) as stream:
                if stream.status_code != 200:
                    return _fail(f"SSE 返回 {stream.status_code}")
                for line in stream.iter_lines():
                    if line.startswith("event:"):
                        events.append(line.split(":", 1)[1].strip())
            if "result" not in events:
                return _fail(f"SSE 未收到 result 事件，收到 {events}")
            print(f"[ok] SSE 事件序列：{events[:8]} … 共 {len(events)} 个")

        # 5. CDSS
        advice = client.post(
            f"{base}/v1/cdss/advice",
            json={
                "question": "社区获得性肺炎门诊患者，是否需要住院？抗菌药物如何选择？",
                "diagnoses": ["社区获得性肺炎"],
                "top_k": 5,
            },
        )
        if advice.status_code != 200:
            return _fail(f"CDSS 失败 {advice.status_code}: {advice.text[:300]}")
        payload = advice.json()
        print(
            f"[ok] CDSS：advice={len(payload['advice'])} degraded={payload['degraded']} "
            f"latency={payload['latency_ms']}ms"
        )

        # 6. 分诊
        triage = client.post(
            f"{base}/v1/cdss/triage", json={"symptoms": "突发胸痛伴大汗、呼吸困难"}
        )
        if triage.status_code != 200 or triage.json()["level"] != "emergency":
            return _fail(f"分诊结果异常：{triage.text[:200]}")
        print(f"[ok] 分诊：{triage.json()['level']} matched={triage.json()['matched']}")

    print("\n全链路冒烟通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

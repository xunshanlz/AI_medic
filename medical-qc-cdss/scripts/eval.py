#!/usr/bin/env python
"""离线评测：金标准集 -> 指标报告（Markdown + JSON）。

指标定义见 docs/03-评测方案与实施.md：
- 缺陷召回率（micro）：预期规则/检查项被命中的比例；
- 误报数：未在预期列表中的问题数（供人工复核，不计入自动评分）；
- ICD Top-1 / Top-3 命中率；
- 证据核验通过率（verified / 有引文的结论）；
- 端到端时延 P50 / P95；
- 降级率。

用法::

    python scripts/eval.py --out reports
    python scripts/eval.py --llm-provider vllm --model Qwen2.5-32B-Instruct-AWQ

注意：使用 ``--llm-provider mock`` 只能验证链路是否跑通，**不能**作为效果指标。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

from qc_cdss.cache import MemoryCache
from qc_cdss.config import get_settings
from qc_cdss.domain.evidence import EvidenceVerifier
from qc_cdss.domain.rules import RuleEngine
from qc_cdss.llm.client import build_llm
from qc_cdss.llm.structured import StructuredGenerator
from qc_cdss.pipeline.graph import QcPipeline
from qc_cdss.pipeline.icd import IcdCatalog, IcdRecommender
from qc_cdss.pipeline.nodes import NodeDeps
from qc_cdss.schemas import QcRequest


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                cases.append(json.loads(line))
    return cases


def make_pipeline(settings: Any) -> QcPipeline:
    llm = build_llm(settings)
    generator = StructuredGenerator(llm, max_attempts=2)
    verifier = EvidenceVerifier(
        accept_threshold=settings.evidence_accept_threshold,
        review_threshold=settings.evidence_review_threshold,
    )
    catalog = IcdCatalog.load(settings.catalog_path)
    icd = IcdRecommender(catalog, generator, verifier) if len(catalog) else None
    deps = NodeDeps(
        settings=settings,
        generator=generator,
        verifier=verifier,
        rules=RuleEngine(),
        icd=icd,
        cache=MemoryCache(ttl=600),
    )
    return QcPipeline(deps)


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(ratio * (len(ordered) - 1))))
    return ordered[index]


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    if args.llm_provider:
        settings.llm_provider = args.llm_provider
    if args.model:
        settings.llm_model = args.model
    if args.use_mock_fallback:
        settings.llm_fallback_provider = "mock"

    cases = load_cases(Path(args.golden) if args.golden else settings.goldset_path)
    if not cases:
        print("[ERROR] 金标准集为空")
        return 1

    pipeline = make_pipeline(settings)
    rows: list[dict[str, Any]] = []
    all_expected = 0
    all_hit = 0
    evidence_total = 0
    evidence_verified = 0
    icd_top1 = 0
    icd_top3 = 0
    latencies: list[float] = []
    degraded_cases = 0

    for case in cases:
        request = QcRequest(
            hospital_id="eval",
            patient_id=case.get("case_id", "eval"),
            encounter_id=case.get("case_id", "eval"),
            sex=case.get("sex", "unknown"),
            age=case.get("age"),
            record=case.get("record"),
        )
        started = time.perf_counter()
        result = await pipeline.run(request, job_id=f"eval-{case['case_id']}")
        latency = (time.perf_counter() - started) * 1000
        latencies.append(latency)

        expected = set(case.get("expected_codes") or [])
        found = {finding.code for finding in result.findings}
        hit = expected & found
        all_expected += len(expected)
        all_hit += len(hit)
        if result.degraded:
            degraded_cases += 1

        for finding in result.findings:
            if finding.evidence is not None:
                evidence_total += 1
                if finding.evidence.state.value == "verified":
                    evidence_verified += 1

        expected_icd = set(case.get("expected_icd") or [])
        codes = [item.code for item in result.icd]
        if expected_icd:
            if codes and codes[0] in expected_icd:
                icd_top1 += 1
            if set(codes[:3]) & expected_icd:
                icd_top3 += 1

        rows.append(
            {
                "case_id": case["case_id"],
                "name": case.get("name", ""),
                "expected": sorted(expected),
                "found": sorted(found),
                "hit": sorted(hit),
                "missing": sorted(expected - found),
                "extra": sorted(found - expected),
                "icd": codes,
                "expected_icd": sorted(expected_icd),
                "latency_ms": round(latency, 1),
                "degraded": result.degraded,
                "score": result.score,
            }
        )
        print(
            f"  {case['case_id']}  命中 {len(hit)}/{len(expected)}  "
            f"额外 {len(found - expected)}  ICD={codes[:2]}  {latency:.0f}ms"
        )

    recall = all_hit / all_expected if all_expected else 0.0
    evidence_rate = evidence_verified / evidence_total if evidence_total else 0.0
    icd_cases = sum(1 for case in cases if case.get("expected_icd"))
    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "llm_provider": settings.llm_provider,
        "model": settings.llm_model,
        "prompt_version": settings.prompt_version,
        "cases": len(cases),
        "defect_recall": round(recall, 4),
        "defect_expected": all_expected,
        "defect_hit": all_hit,
        "evidence_verified_rate": round(evidence_rate, 4),
        "evidence_total": evidence_total,
        "icd_top1_rate": round(icd_top1 / icd_cases, 4) if icd_cases else 0.0,
        "icd_top3_rate": round(icd_top3 / icd_cases, 4) if icd_cases else 0.0,
        "latency_p50_ms": round(percentile(latencies, 0.5), 1),
        "latency_p95_ms": round(percentile(latencies, 0.95), 1),
        "degraded_rate": round(degraded_cases / len(cases), 4),
        "rows": rows,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"eval_{stamp}.json"
    md_path = out_dir / f"eval_{stamp}.md"
    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 质控效果评测报告",
        "",
        f"- 生成时间：{metrics['generated_at']}",
        f"- 模型：{metrics['model']}（{metrics['llm_provider']}）",
        f"- 提示词版本：{metrics['prompt_version']}",
        f"- 样例数：{metrics['cases']}",
        "",
        "## 汇总指标",
        "",
        "| 指标 | 数值 | 说明 |",
        "| --- | --- | --- |",
        f"| 缺陷召回率 | {metrics['defect_recall']:.2%} | {metrics['defect_hit']}/{metrics['defect_expected']} |",
        f"| 证据核验通过率 | {metrics['evidence_verified_rate']:.2%} | {metrics['evidence_total']} 条带引文结论 |",
        f"| ICD Top-1 | {metrics['icd_top1_rate']:.2%} | 主要诊断编码 |",
        f"| ICD Top-3 | {metrics['icd_top3_rate']:.2%} | 候选含正确编码 |",
        f"| 时延 P50 / P95 | {metrics['latency_p50_ms']} / {metrics['latency_p95_ms']} ms | 端到端 |",
        f"| 降级率 | {metrics['degraded_rate']:.2%} | 含超时/解析失败 |",
        "",
        "## 分例明细",
        "",
        "| 用例 | 命中 | 漏报 | 额外 | ICD | 耗时(ms) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} {row['name']} | {len(row['hit'])}/{len(row['expected'])} | "
            f"{', '.join(row['missing']) or '-'} | {len(row['extra'])} | "
            f"{', '.join(row['icd'][:2]) or '-'} | {row['latency_ms']} |"
        )
    lines.extend(
        [
            "",
            "> 说明：额外问题不计入自动评分，需人工判断是否为合理发现（金标准集采用"
            "“预期问题清单”而非“穷举所有问题”，因此召回应与人工复核结合解读）。",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 56)
    print(f"缺陷召回率     {metrics['defect_recall']:.2%}")
    print(f"证据核验通过率 {metrics['evidence_verified_rate']:.2%}")
    print(f"ICD Top-1/3    {metrics['icd_top1_rate']:.2%} / {metrics['icd_top3_rate']:.2%}")
    print(f"时延 P50/P95   {metrics['latency_p50_ms']} / {metrics['latency_p95_ms']} ms")
    print(f"报告：{md_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="金标准集离线评测")
    parser.add_argument("--golden", default=None, help="金标准集路径（JSONL）")
    parser.add_argument("--out", default="reports", help="报告输出目录")
    parser.add_argument("--llm-provider", choices=["mock", "vllm", "cloud"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--use-mock-fallback", action="store_true")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())

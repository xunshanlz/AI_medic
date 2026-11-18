#!/usr/bin/env python
"""零依赖自检：在无网络、无 GPU、无第三方包的环境验证核心算法正确性。

覆盖的是整个系统里"最不能出错"的部分：
- 归一化与偏移映射（证据高亮坐标不能错位）
- 证据校验三态（精确/近似/幻觉）
- 规则引擎（含缺项、性别矛盾、危急值、剂量、否定式矛盾、时限）
- RRF 融合与 BM25 排序
- JSON 容错解析
- 评分与去重

用法::

    python scripts/selfcheck.py

退出码 0 表示全部通过。CI 可直接接入。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:  # Windows 控制台默认 GBK，统一输出编码
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

from qc_cdss.domain.evidence import INVALID, PENDING, VERIFIED, EvidenceVerifier
from qc_cdss.domain.rules import PatientContext, RuleEngine
from qc_cdss.domain.scoring import aggregate
from qc_cdss.domain.sections import build_document
from qc_cdss.domain.textnorm import normalize
from qc_cdss.llm.jsonx import loads_lenient
from qc_cdss.rag.bm25 import BM25Index
from qc_cdss.rag.fusion import rrf_fuse

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def check(name: str):
    def decorator(func):
        try:
            func()
        except Exception as exc:  # noqa: BLE001
            FAILED.append((name, f"{type(exc).__name__}: {exc}"))
        else:
            PASSED.append(name)
        return func

    return decorator


@check("文本归一化：全角/空白折叠且偏移可回映")
def _check_textnorm() -> None:
    source = "患者发热３天\n伴咳嗽"
    normalized = normalize(source)
    assert normalized.text == "患者发热3天 伴咳嗽", normalized.text
    hit = normalized.text.find("发热3天 伴咳嗽")
    start, end = normalized.span_to_original(hit, hit + len("发热3天 伴咳嗽"))
    assert source[start:end] == "发热３天\n伴咳嗽", source[start:end]


@check("证据校验：精确匹配返回原文区间")
def _check_evidence_exact() -> None:
    source = "患者主诉发热3天，伴咳嗽咳痰。"
    result = EvidenceVerifier().verify("发热3天", source)
    assert result.state == VERIFIED, result.state
    assert source[result.start : result.end] == "发热3天"


@check("证据校验：换行/空格噪声可命中")
def _check_evidence_newline() -> None:
    source = "主诉：发热3天\n伴咳嗽咳痰"
    result = EvidenceVerifier().verify("发热3天伴咳嗽", source)
    assert result.state == VERIFIED, result.state


@check("证据校验：编造引文判为无效且不返回区间")
def _check_evidence_fabricated() -> None:
    source = "患者主诉发热3天，伴咳嗽咳痰。"
    result = EvidenceVerifier().verify("患者有糖尿病病史10年，长期口服二甲双胍", source)
    assert result.state == INVALID, result.state
    assert result.start is None


@check("证据校验：近似表述判为待核验")
def _check_evidence_pending() -> None:
    verifier = EvidenceVerifier(accept_threshold=99.0, review_threshold=60.0)
    source = "患者主诉发热三天，伴咳嗽咳痰，无胸闷。"
    result = verifier.verify("患者主诉发热三天，伴咳痰", source)
    assert result.state == PENDING, result.state


@check("规则引擎：规范病历零误报")
def _check_rules_clean() -> None:
    record = {
        "chief_complaint": "发热3天",
        "present_illness": "3天前受凉后出现发热，最高38.5℃，伴咳嗽。",
        "past_history": "否认高血压、糖尿病。",
        "physical_exam": "T 38.2℃，咽部充血。",
        "diagnosis": "急性上呼吸道感染",
        "physician_signature": "张医生",
    }
    hits = RuleEngine().run(build_document(record), PatientContext(sex="male", age=30))
    assert hits == [], [hit.rule_id for hit in hits]


@check("规则引擎：问题病历命中全部预期规则")
def _check_rules_defective() -> None:
    payload = json.loads(
        (ROOT / "data" / "emr" / "emr_inpatient_defective.json").read_text(encoding="utf-8")
    )
    doc = build_document(payload["record"])
    context = PatientContext(
        sex=payload["sex"],
        age=payload["age"],
        admit_time=__import__("datetime").datetime.fromisoformat(payload["admit_time"]),
        record_time=__import__("datetime").datetime.fromisoformat(payload["record_time"]),
    )
    codes = {hit.rule_id for hit in RuleEngine().run(doc, context)}
    expected = {"QC-R001", "QC-R003", "QC-R004", "QC-R006", "QC-R007", "QC-R008", "QC-R009"}
    missing = expected - codes
    assert not missing, f"漏报规则: {sorted(missing)}，实际命中: {sorted(codes)}"


@check("规则引擎：所有命中区间可在原文截取")
def _check_rules_offsets() -> None:
    payload = json.loads(
        (ROOT / "data" / "emr" / "emr_inpatient_defective.json").read_text(encoding="utf-8")
    )
    doc = build_document(payload["record"])
    for hit in RuleEngine().run(doc, PatientContext(sex="female", age=68)):
        if hit.quote:
            assert doc.text[hit.start : hit.end] == hit.quote, hit.rule_id


@check("RRF：多路共同命中者排序靠前")
def _check_rrf() -> None:
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    lexical = [("b", 12.0), ("d", 9.0)]
    fused = rrf_fuse([dense, lexical])
    assert fused[0].id == "b", [hit.id for hit in fused]
    assert fused[0].ranks == (2, 1)


@check("BM25：中文术语可正确召回")
def _check_bm25() -> None:
    index = BM25Index().bulk_add(
        [
            ("g1", "社区获得性肺炎的诊断依据包括咳嗽咳痰发热及肺部湿啰音。", "社区获得性肺炎"),
            ("g2", "高血压患者血压控制目标一般为140/90mmHg以下。", "高血压"),
        ]
    )
    results = index.search("社区获得性肺炎 诊断", top_k=2)
    assert results and results[0][0] == "g1", results


@check("JSON 容错：围栏、散文、尾逗号均可解析")
def _check_jsonx() -> None:
    text = '好的：```json\n{"passed": True, "findings": [1, 2,],}\n``` 以上。'
    assert loads_lenient(text) == {"passed": True, "findings": [1, 2]}


@check("评分：重复报告去重且分数不为负")
def _check_scoring() -> None:
    findings = [
        {"code": "QC-LLM-01", "severity": "high", "section": "主诉", "quote": "发热", "evidence": {"start": 0, "end": 2}},
        {"code": "QC-LLM-01", "severity": "medium", "section": "主诉", "quote": "发热", "evidence": {"start": 0, "end": 2}},
    ]
    result = aggregate(findings)
    assert result["total"] == 1, result
    assert result["score"] == 90.0, result


def main() -> int:
    print("=" * 68)
    print("QC-CDSS 核心算法自检（零依赖）")
    print("=" * 68)
    for name in PASSED:
        print(f"  [PASS] {name}")
    for name, error in FAILED:
        print(f"  [FAIL] {name}\n         {error}")
    print("-" * 68)
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    raise SystemExit(main())

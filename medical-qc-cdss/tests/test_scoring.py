import unittest
from enum import Enum

from qc_cdss.domain.scoring import aggregate, dedupe


class Sev(str, Enum):
    high = "high"
    medium = "medium"


def finding(code="QC-LLM-01", severity="high", start=0, end=10, quote="发热3天", section="主诉", **extra):
    payload = {
        "code": code,
        "severity": severity,
        "section": section,
        "quote": quote,
        "evidence": {"start": start, "end": end, "state": "verified"},
    }
    payload.update(extra)
    return payload


class ScoringTest(unittest.TestCase):
    def test_score_is_penalized_by_severity(self):
        result = aggregate([finding(severity="high"), finding(code="X", severity="medium")])
        self.assertEqual(result["score"], 85.0)
        self.assertEqual(result["counts"], {"high": 1, "medium": 1, "low": 0})

    def test_duplicate_reports_are_deduped(self):
        items = [finding(), finding(severity="medium")]
        self.assertEqual(len(dedupe(items)), 1)

    def test_same_code_different_section_is_kept(self):
        items = [finding(section="主诉"), finding(section="现病史", start=50, end=60)]
        self.assertEqual(len(dedupe(items)), 2)

    def test_non_overlapping_spans_are_kept(self):
        items = [
            finding(quote="发热", start=0, end=5),
            finding(quote="咳嗽", start=100, end=105),
        ]
        self.assertEqual(len(dedupe(items)), 2)

    def test_score_never_negative(self):
        items = [finding(code=f"C{i}", severity="high", start=i * 10, end=i * 10 + 3) for i in range(20)]
        self.assertEqual(aggregate(items)["score"], 0.0)

    def test_needs_review_counts_pending_evidence(self):
        items = [finding(evidence={"start": 0, "end": 3, "state": "pending"})]
        self.assertEqual(aggregate(items)["needs_review"], 1)

    def test_enum_severity_is_supported(self):
        # pydantic model_dump 可能传枚举成员，评分不能因此全部按最低档计算
        result = aggregate([finding(severity=Sev.high)])
        self.assertEqual(result["score"], 90.0)
        self.assertEqual(result["counts"]["high"], 1)


if __name__ == "__main__":
    unittest.main()

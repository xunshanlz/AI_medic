import unittest
from datetime import datetime, timedelta

from qc_cdss.domain.rules import PatientContext, RuleEngine
from qc_cdss.domain.sections import build_document


def make_doc(**overrides):
    record = {
        "chief_complaint": "发热3天",
        "present_illness": "3天前受凉后出现发热，最高38.5℃，伴咳嗽。",
        "past_history": "否认高血压、糖尿病。",
        "physical_exam": "T 38.2℃，咽部充血。",
        "diagnosis": "急性上呼吸道感染",
        "physician_signature": "张医生",
    }
    record.update(overrides)
    return build_document(record)


class RuleEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = RuleEngine()

    def test_clean_record_has_no_findings(self):
        hits = self.engine.run(make_doc(), PatientContext(sex="male", age=30))
        self.assertEqual([h.rule_id for h in hits], [])

    def test_missing_sections_are_reported(self):
        doc = build_document({"chief_complaint": "发热3天"})
        hits = self.engine.run(doc)
        codes = {h.rule_id for h in hits}
        self.assertIn("QC-R001", codes)
        missing = {h.title for h in hits if h.rule_id == "QC-R001"}
        self.assertIn("现病史缺失", missing)
        self.assertIn("初步诊断缺失", missing)

    def test_chief_complaint_too_long(self):
        hits = self.engine.run(make_doc(chief_complaint="反复咳嗽咳痰伴气喘十余年加重一周伴发热" * 2))
        self.assertIn("QC-R002", {h.rule_id for h in hits})

    def test_chief_complaint_contains_diagnosis(self):
        hits = self.engine.run(make_doc(chief_complaint="肺炎3天"))
        self.assertIn("QC-R003", {h.rule_id for h in hits})

    def test_sex_conflict(self):
        doc = make_doc(present_illness="停经40天，妊娠试验阳性，恶心呕吐。")
        hits = self.engine.run(doc, PatientContext(sex="male", age=30))
        self.assertIn("QC-R004", {h.rule_id for h in hits})

    def test_critical_value_preempts(self):
        doc = make_doc(physical_exam="T 36.5℃，血压 190/110 mmHg，神志清楚。")
        hits = self.engine.run(doc)
        safety = [h for h in hits if h.rule_id == "QC-R006"]
        self.assertTrue(safety)
        self.assertEqual(safety[0].severity, "high")
        self.assertIn("血压", safety[0].quote)

    def test_dosage_over_limit(self):
        hits = self.engine.run(make_doc(orders="阿司匹林 500mg 口服 每日一次"))
        dosage = [h for h in hits if h.rule_id == "QC-R008"]
        self.assertTrue(dosage)
        self.assertIn("500", dosage[0].quote)

    def test_negation_conflict(self):
        doc = make_doc(present_illness="无发热，仅咳嗽咳痰3天。")
        hits = self.engine.run(doc)
        self.assertIn("QC-R009", {h.rule_id for h in hits})

    def test_timeliness(self):
        admit = datetime(2026, 1, 1, 8, 0)
        hits = self.engine.run(
            make_doc(),
            PatientContext(sex="male", age=30, admit_time=admit, record_time=admit + timedelta(hours=30)),
        )
        self.assertIn("QC-R007", {h.rule_id for h in hits})

    def test_hits_carry_valid_offsets(self):
        doc = make_doc(physical_exam="血压 190/110 mmHg。")
        for hit in self.engine.run(doc):
            if hit.quote:
                self.assertEqual(doc.text[hit.start : hit.end], hit.quote)


if __name__ == "__main__":
    unittest.main()

import unittest

from qc_cdss.domain.sections import build_document, canonical_title, split_sections


class SectionSplitTest(unittest.TestCase):
    def test_split_raw_text(self):
        raw = (
            "主诉：发热3天\n\n"
            "现病史：3天前受凉后出现发热，最高38.5℃。\n"
            "初步诊断：急性上呼吸道感染"
        )
        doc = build_document(raw_text=raw)
        names = [s.name for s in doc.sections]
        self.assertEqual(names, ["主诉", "现病史", "初步诊断"])
        for sec in doc.sections:
            self.assertEqual(doc.text[sec.start : sec.end], sec.text)
        self.assertEqual(doc.section_text("初步诊断"), "急性上呼吸道感染")

    def test_text_without_headers_becomes_single_section(self):
        doc = build_document(raw_text="患者发热三天，伴咳嗽。")
        self.assertEqual(len(doc.sections), 1)
        self.assertEqual(doc.sections[0].name, "全文")

    def test_structured_record_is_composed(self):
        doc = build_document(
            {
                "chief_complaint": "发热3天",
                "diagnosis": ["急性上呼吸道感染"],
                "physician_signature": "张医生",
            }
        )
        self.assertIn("主诉：发热3天", doc.text)
        self.assertIn("初步诊断：急性上呼吸道感染", doc.text)
        self.assertEqual(doc.section_text("初步诊断"), "急性上呼吸道感染")
        sec = doc.section("主诉")
        self.assertIsNotNone(sec)
        self.assertEqual(doc.text[sec.start : sec.end], "发热3天")

    def test_section_at_offset(self):
        doc = build_document({"chief_complaint": "发热3天", "diagnosis": "肺炎"})
        offset = doc.text.find("肺炎")
        sec = doc.section_at(offset)
        self.assertIsNotNone(sec)
        self.assertEqual(sec.name, "初步诊断")

    def test_canonical_title_aliases(self):
        self.assertEqual(canonical_title("chief_complaint"), "主诉")
        self.assertEqual(canonical_title("physical_exam"), "体格检查")
        self.assertEqual(canonical_title("主诉（S）"), "主诉")

    def test_missing_section_returns_none(self):
        doc = build_document({"chief_complaint": "发热3天"})
        self.assertIsNone(doc.section("手术记录"))
        self.assertFalse(doc.has_content("手术记录"))


if __name__ == "__main__":
    unittest.main()

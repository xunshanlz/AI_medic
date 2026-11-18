import unittest

from qc_cdss.domain.sections import build_document
from qc_cdss.privacy import mask_document, mask_pii, mask_pii_aligned


class PrivacyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = (
            "姓名：张三，男，身份证110101199001011234，电话13800138000，"
            "邮箱zhangsan@example.com，住址：北京市朝阳区建国路1号。"
        )

    def test_mask_pii_replaces_identifiers(self):
        masked, stats = mask_pii(self.text)
        self.assertNotIn("张三", masked)
        self.assertNotIn("110101199001011234", masked)
        self.assertNotIn("13800138000", masked)
        self.assertNotIn("zhangsan@example.com", masked)
        self.assertGreater(stats.total, 0)

    def test_aligned_masking_preserves_length(self):
        masked = mask_pii_aligned(self.text)
        self.assertEqual(len(masked), len(self.text))

    def test_aligned_masking_removes_identifiers(self):
        masked = mask_pii_aligned(self.text)
        self.assertNotIn("张三", masked)
        self.assertNotIn("110101199001011234", masked)
        self.assertNotIn("13800138000", masked)
        self.assertNotIn("zhangsan@example.com", masked)
        self.assertNotIn("北京市朝阳区", masked)

    def test_aligned_masking_keeps_line_structure(self):
        source = "主诉：发热3天\n现病史：姓名：张三，发热3天。"
        masked = mask_pii_aligned(source)
        self.assertEqual(len(masked), len(source))
        self.assertEqual(masked.count("\n"), source.count("\n"))
        # 段落后半句的偏移未漂移
        self.assertEqual(masked.find("发热3天。"), source.find("发热3天。"))

    def test_name_masking_keeps_sentence_structure(self):
        self.assertEqual(mask_pii_aligned("姓名：张三，发热3天。"), "姓名：某某，发热3天。")

    def test_symptom_is_not_mistaken_for_name(self):
        # 无冒号时不得把"发热"当姓名，否则正文被破坏、证据无法回溯
        self.assertEqual(mask_pii_aligned("患者发热，3天后复诊。"), "患者发热，3天后复诊。")

    def test_empty_text(self):
        self.assertEqual(mask_pii_aligned(""), "")
        masked, stats = mask_pii("")
        self.assertEqual(masked, "")
        self.assertEqual(stats.total, 0)

    def test_mask_document_preserves_structure_and_offsets(self):
        document = build_document(
            {
                "chief_complaint": "发热3天",
                "present_illness": "姓名：张三，发热3天，伴咳嗽。",
                "diagnosis": "社区获得性肺炎",
            }
        )
        masked = mask_document(document)
        self.assertEqual(len(masked.text), len(document.text))
        self.assertEqual([s.name for s in masked.sections], [s.name for s in document.sections])
        for original, redacted in zip(document.sections, masked.sections):
            self.assertEqual(original.start, redacted.start)
            self.assertEqual(original.end, redacted.end)
        self.assertNotIn("张三", masked.text)


if __name__ == "__main__":
    unittest.main()

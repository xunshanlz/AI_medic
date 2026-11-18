import unittest

from qc_cdss.llm.jsonx import extract_json_blocks, loads_lenient, salvage_mapping, strip_code_fences


class JsonRepairTest(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(loads_lenient('{"a": 1}'), {"a": 1})

    def test_fenced_json(self):
        text = '```json\n{"ok": true, "items": [1, 2]}\n```'
        self.assertEqual(loads_lenient(text), {"ok": True, "items": [1, 2]})

    def test_json_with_prose_around_it(self):
        text = '好的，以下是质控结果：{"findings": [{"code": "QC-LLM-01"}]} 希望对您有帮助。'
        self.assertEqual(loads_lenient(text)["findings"][0]["code"], "QC-LLM-01")

    def test_trailing_comma_and_python_literals(self):
        text = '{"a": True, "b": None, "c": [1, 2,],}'
        self.assertEqual(loads_lenient(text), {"a": True, "b": None, "c": [1, 2]})

    def test_truncated_output_raises_then_salvages_nothing(self):
        with self.assertRaises(ValueError):
            loads_lenient('{"findings": [{"code": "QC-LLM-01"')
        self.assertEqual(salvage_mapping('{"findings": [', ["findings"]), {})

    def test_salvage_mapping_keeps_known_keys(self):
        text = '{"findings": [], "extra": 1}'
        self.assertEqual(salvage_mapping(text, ["findings"]), {"findings": []})

    def test_extract_blocks_handles_braces_in_strings(self):
        blocks = extract_json_blocks('prefix {"text": "包含 } 花括号"} suffix')
        self.assertEqual(blocks, ['{"text": "包含 } 花括号"}'])

    def test_strip_code_fences_without_language(self):
        self.assertEqual(strip_code_fences("```\n{}\n```"), "{}")

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            loads_lenient("   ")


if __name__ == "__main__":
    unittest.main()

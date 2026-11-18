import unittest

from qc_cdss.domain.textnorm import compact, normalize, normalize_text, trim


class TextNormalizeTest(unittest.TestCase):
    def test_fullwidth_and_case(self):
        self.assertEqual(normalize_text("ＡＢＣ　１２３"), "abc 123")

    def test_span_maps_back_to_original(self):
        source = "患者发热３天"
        normalized = normalize(source)
        hit = normalized.text.find("发热3天")
        self.assertGreaterEqual(hit, 0)
        start, end = normalized.span_to_original(hit, hit + len("发热3天"))
        self.assertEqual(source[start:end], "发热３天")

    def test_collapsed_whitespace_maps_to_first_char(self):
        source = "发热  3天"
        normalized = normalize(source)
        self.assertEqual(normalized.text, "发热 3天")
        start, end = normalized.span_to_original(0, len(normalized.text))
        self.assertEqual(source[start:end], source)

    def test_zero_width_characters_are_dropped(self):
        self.assertEqual(normalize_text("发\u200b热"), "发热")

    def test_compact_and_trim(self):
        normalized = normalize("  发热 3 天  ")
        self.assertEqual(trim(normalized).text, "发热 3 天")
        self.assertEqual(compact(trim(normalized)).text, "发热3天")

    def test_empty_input(self):
        self.assertEqual(normalize("").text, "")
        self.assertIsNone(normalize("").span_to_original(0, 1))


if __name__ == "__main__":
    unittest.main()

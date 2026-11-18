import unittest

from qc_cdss.domain.evidence import INVALID, PENDING, VERIFIED, EvidenceVerifier


class EvidenceVerifierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.verifier = EvidenceVerifier()

    def test_exact_match_returns_original_offsets(self):
        source = "患者主诉发热3天，伴咳嗽咳痰。"
        result = self.verifier.verify("发热3天", source)
        self.assertEqual(result.state, VERIFIED)
        self.assertEqual(result.strategy, "exact")
        self.assertEqual(source[result.start : result.end], "发热3天")

    def test_fullwidth_digits_are_matched(self):
        source = "患者发热３天，最高３９．２℃。"
        result = self.verifier.verify("发热3天", source)
        self.assertEqual(result.state, VERIFIED)
        self.assertEqual(source[result.start : result.end], "发热３天")

    def test_newline_and_space_noise_is_tolerated(self):
        source = "主诉：发热3天\n伴咳嗽咳痰"
        result = self.verifier.verify("发热3天伴咳嗽", source)
        self.assertEqual(result.state, VERIFIED)
        self.assertIn("发热3天", result.matched or "")

    def test_fabricated_quote_is_invalid(self):
        source = "患者主诉发热3天，伴咳嗽咳痰，无胸闷。"
        result = self.verifier.verify("患者有糖尿病病史10年，长期口服二甲双胍", source)
        self.assertEqual(result.state, INVALID)
        self.assertIsNone(result.start)

    def test_close_paraphrase_can_be_pending(self):
        strict = EvidenceVerifier(accept_threshold=99.0, review_threshold=60.0)
        source = "患者主诉发热三天，伴咳嗽咳痰，无胸闷。"
        result = strict.verify("患者主诉发热三天，伴咳痰", source)
        self.assertEqual(result.state, PENDING)
        self.assertIsNotNone(result.start)
        self.assertEqual(source[result.start : result.end], result.matched)

    def test_chinese_numeral_duration_is_matched(self):
        source = "患者发热三天，伴咳嗽。"
        result = self.verifier.verify("发热3天", source)
        self.assertEqual(result.state, VERIFIED)
        self.assertEqual(source[result.start : result.end], "发热三天")

    def test_short_quote_skips_fuzzy_matching(self):
        result = self.verifier.verify("胸闷", "患者发热3天，伴咳嗽咳痰。")
        self.assertEqual(result.state, INVALID)

    def test_span_never_exceeds_source_bounds(self):
        source = "发热"
        result = self.verifier.verify("发热", source)
        self.assertEqual((result.start, result.end), (0, 2))

    def test_verification_serialization(self):
        result = self.verifier.verify("发热3天", "患者发热3天。")
        payload = result.as_dict()
        self.assertEqual(payload["state"], VERIFIED)
        self.assertEqual(payload["score"], 100.0)


if __name__ == "__main__":
    unittest.main()

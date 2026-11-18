import unittest

from qc_cdss.rag.bm25 import BM25Index, tokenize


class BM25Test(unittest.TestCase):
    def setUp(self) -> None:
        self.index = BM25Index().bulk_add(
            [
                ("g1", "社区获得性肺炎的诊断依据包括咳嗽咳痰发热及肺部湿啰音。", "社区获得性肺炎"),
                ("g2", "高血压患者血压控制目标一般为140/90mmHg以下。", "高血压"),
                ("g3", "2型糖尿病患者首选二甲双胍，需监测糖化血红蛋白。", "2型糖尿病"),
            ]
        )

    def test_tokenize_chinese_bigrams(self):
        tokens = tokenize("肺炎")
        self.assertIn("肺炎", tokens)

    def test_query_hits_expected_document(self):
        results = self.index.search("社区获得性肺炎 诊断", top_k=2)
        self.assertEqual(results[0][0], "g1")

    def test_unknown_terms_return_empty(self):
        self.assertEqual(self.index.search("完全无关的词汇组合xyz"), [])

    def test_scores_are_sorted_desc(self):
        results = self.index.search("糖尿病 二甲双胍", top_k=3)
        self.assertEqual(results[0][0], "g3")
        self.assertEqual([s for _, s in results], sorted([s for _, s in results], reverse=True))

    def test_build_is_idempotent(self):
        self.index.build()
        first = self.index.search("高血压")
        self.index.build()
        self.assertEqual(first, self.index.search("高血压"))


if __name__ == "__main__":
    unittest.main()

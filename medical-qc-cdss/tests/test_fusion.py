import unittest

from qc_cdss.rag.fusion import rrf_fuse


class RRFFusionTest(unittest.TestCase):
    def test_shared_document_wins(self):
        dense = [("a", 0.91), ("b", 0.88), ("c", 0.71)]
        lexical = [("b", 12.4), ("d", 9.1)]
        fused = rrf_fuse([dense, lexical])
        self.assertEqual(fused[0].id, "b")
        self.assertEqual(fused[0].ranks, (2, 1))

    def test_ranks_are_one_based_and_recorded(self):
        fused = rrf_fuse([[("x", 1.0), ("y", 0.5)], [("y", 2.0), ("z", 1.0)]])
        by_id = {hit.id: hit for hit in fused}
        self.assertEqual(by_id["x"].ranks, (1, 0))
        self.assertEqual(by_id["z"].ranks, (0, 2))
        self.assertGreater(by_id["y"].score, by_id["z"].score)

    def test_weights_shift_ranking(self):
        dense = [("a", 0.9), ("b", 0.8)]
        lexical = [("b", 5.0)]
        unweighted = rrf_fuse([dense, lexical])
        weighted = rrf_fuse([dense, lexical], weights=[100.0, 1.0])
        self.assertEqual(unweighted[0].id, "b")
        self.assertEqual(weighted[0].id, "a")

    def test_top_k_truncates(self):
        fused = rrf_fuse([[("a", 1), ("b", 1), ("c", 1)]], top_k=2)
        self.assertEqual(len(fused), 2)

    def test_invalid_weights_raise(self):
        with self.assertRaises(ValueError):
            rrf_fuse([[("a", 1)]], weights=[1.0, 2.0])

    def test_duplicate_ids_within_one_run_are_deduped(self):
        fused = rrf_fuse([[("a", 1.0), ("a", 0.9)]])
        self.assertEqual(len(fused), 1)


if __name__ == "__main__":
    unittest.main()

import unittest

from scripts.analyze_llada_query_overlap_longbench import summarize_rows


class SummaryTest(unittest.TestCase):
    def test_prompts_are_equal_weight(self):
        rows = [
            dict(task="a", layer=1, ratio=0.5, recall=0.2),
            dict(task="a", layer=1, ratio=0.5, recall=0.8),
            dict(task="b", layer=1, ratio=0.5, recall=1.0),
        ]
        dataset, overall = summarize_rows(rows, tasks=("a", "b"))
        self.assertEqual([row["mean"] for row in dataset], [0.5, 1.0])
        self.assertAlmostEqual(overall[0]["mean"], 2 / 3)


if __name__ == "__main__":
    unittest.main()

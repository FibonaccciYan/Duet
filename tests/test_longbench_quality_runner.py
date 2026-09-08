import unittest

import torch

from scripts.run_longbench_quality import GEN_LENGTHS, TASKS, truncate_middle


class LongBenchQualityRunnerTest(unittest.TestCase):
    def test_requested_tasks_and_generation_lengths_are_defined(self):
        self.assertEqual(set(TASKS), {"hotpotqa", "triviaqa", "narrativeqa", "qasper", "multifieldqa_en"})
        self.assertEqual(GEN_LENGTHS["narrativeqa"], 128)

    def test_middle_truncation_keeps_both_ends(self):
        values = torch.arange(10).unsqueeze(0)
        truncated, changed = truncate_middle(values, 6)
        self.assertTrue(changed)
        self.assertEqual(truncated.tolist(), [[0, 1, 2, 7, 8, 9]])


if __name__ == "__main__":
    unittest.main()

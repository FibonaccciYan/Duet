import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.run_longbench_quality import (
    GEN_LENGTHS,
    TASKS,
    answer_scores,
    load_completed,
    summarize,
    truncate_middle,
)


class LongBenchQualityRunnerTest(unittest.TestCase):
    def test_tasks_and_middle_truncation(self):
        self.assertEqual(len(TASKS), 5)
        self.assertEqual(GEN_LENGTHS["narrativeqa"], 128)
        truncated, changed = truncate_middle(torch.arange(10).unsqueeze(0), 6)
        self.assertTrue(changed)
        self.assertEqual(truncated.tolist(), [[0, 1, 2, 7, 8, 9]])

    def test_scoring_and_resume(self):
        f1, exact_match = answer_scores("The blue whale", ["blue whale", "orca"])
        self.assertEqual((f1, exact_match), (1.0, 1.0))
        row = {"task": "hotpotqa", "index": 0, "f1": f1, "exact_match": exact_match}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.jsonl"
            path.write_text("not json\n" + json.dumps(row) + "\n", encoding="utf-8")
            self.assertEqual(load_completed(path), {("hotpotqa", 0): row})
        self.assertEqual(summarize([row])["overall"]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()

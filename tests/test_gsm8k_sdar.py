import importlib.util
import unittest
from pathlib import Path


UTILS_PATH = (
    Path(__file__).parents[1]
    / "eval_instruct/lm_eval/tasks/gsm8k/sdar_utils.py"
)
SPEC = importlib.util.spec_from_file_location("gsm8k_sdar_utils", UTILS_PATH)
UTILS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(UTILS)


class SDARGSM8KTest(unittest.TestCase):
    def test_boxed_answer_uses_official_gsm8k_target(self):
        doc = {"answer": "work\n#### 18"}
        response = ["The answer is $\\boxed{18}$."]

        self.assertEqual(UTILS.process_results(doc, response), {"exact_match": 1})

    def test_last_boxed_answer_wins(self):
        doc = {"answer": "work\n#### 3"}
        response = ["First $\\boxed{2}$, corrected to $\\boxed{3}$."]

        self.assertEqual(UTILS.process_results(doc, response), {"exact_match": 1})


if __name__ == "__main__":
    unittest.main()

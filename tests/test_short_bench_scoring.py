"""CPU regression tests; run with math-verify and PyYAML installed."""
import importlib.util
from pathlib import Path
import re
import sys
import types
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / 'eval_instruct/lm_eval/tasks'
spec = importlib.util.spec_from_file_location('math500_scoring_test', TASKS / 'math_500/utils.py')
math500 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math500)


class ScoringTests(unittest.TestCase):
    def test_gold_identity(self):
        for gold in ['42', 'p-q', r'\sqrt{51}', r'3\sqrt{13}',
                     r'\left(3,\frac{\pi}{2}\right)', '6-5i', r'\text{Evelyn}']:
            with self.subTest(gold=gold):
                scores = math500.process_results({'answer': gold}, [r'\boxed{' + gold + '}'])
                self.assertEqual(scores['math_verify'], 1)
                self.assertEqual(scores['exact_match'], 1)

    def test_numeric_fragment_false_positive(self):
        scores = math500.process_results({'answer': r'70\sqrt{2}'}, [r'\boxed{70}'])
        self.assertEqual(scores['math_verify'], 0)
        self.assertEqual(set(scores), {'math_verify', 'exact_match'})

    def test_boxes(self):
        self.assertEqual(math500.last_boxed_answer(r'\boxed{2} then \fbox{\frac{1}{2}}'), r'\frac{1}{2}')
        self.assertEqual(math500.last_boxed_answer(r'\boxed{\{1,2\}}'), r'\{1,2\}')
        self.assertIsNone(math500.last_boxed_answer(r'\boxed{2} then \boxed{\frac{1}'))
        self.assertIsNone(math500.last_boxed_answer('no answer'))
        self.assertEqual(math500.process_results({'answer': '2'}, ['2'])['exact_match'], 0)

    def test_import_collision(self):
        previous = sys.modules.get('utils')
        try:
            sys.modules['utils'] = types.ModuleType('utils')
            spec.loader.exec_module(importlib.util.module_from_spec(spec))
        finally:
            if previous is None:
                sys.modules.pop('utils', None)
            else:
                sys.modules['utils'] = previous

    def test_gsm8k_config_extraction(self):
        config = yaml.safe_load((TASKS / 'gsm8k/gsm8k.yaml').read_text())
        rule = next(f for f in config['filter_list'] if f['name'] == 'flexible-extract')['filter'][0]
        pattern = re.compile(rule['regex_pattern'])
        for text, expected in [(r'work 42. Answer \boxed{6} $$', '6'),
                               ('Answer -1,234.50 ..', '-1,234.50'),
                               ('Answer 7.', '7'), ('$$ ..', None)]:
            matches = pattern.findall(text)
            self.assertEqual(matches[rule['group_select']] if matches else None, expected)


if __name__ == '__main__':
    unittest.main()

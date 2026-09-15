import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "scripts/attention_mass/plot_attention_mass.py"
SPEC = importlib.util.spec_from_file_location("plot_attention_mass", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AttentionMassPlotTest(unittest.TestCase):
    def test_budget_indices_are_fixed_from_first_step(self):
        mass = np.array([[[[0.8, 0.2]], [[0.1, 0.9]]]], dtype=np.float32)
        curves = MODULE.budget_coverage(mass, [1])
        np.testing.assert_allclose(curves[0, 0], [0.8, 0.1])

    def test_blocks_select_independently_and_align_by_resolved_tokens(self):
        mass = np.array(
            [[[[0.8, 0.2]], [[0.6, 0.4]], [[0.1, 0.9]], [[0.3, 0.7]]]],
            dtype=np.float32,
        )
        blocks = np.array([0, 0, 1, 1])
        resolved = np.array([0, 2, 0, 2])
        curves = MODULE.budget_coverage(mass, [1], blocks, np.full(4, 2))
        np.testing.assert_allclose(curves[0, 0], [0.8, 0.6, 0.9, 0.7])
        progress, aligned = MODULE.expand_steps_to_tokens(curves, blocks, resolved, 3)
        np.testing.assert_array_equal(progress, [0, 1, 2])
        np.testing.assert_allclose(aligned[0], [[0.8, 0.8, 0.6], [0.9, 0.9, 0.7]])

    def test_samples_have_equal_weight_despite_different_block_counts(self):
        first = np.array([[[1.0, 1.0], [0.0, 0.0]]])
        second = np.array([[[0.2, 0.2]]])
        sample_curves = MODULE.aggregate_samples([first, second])
        np.testing.assert_allclose(sample_curves[0], [[0.5, 0.5], [0.2, 0.2]])
        np.testing.assert_allclose(sample_curves.mean(axis=1), [[0.35, 0.35]])


if __name__ == "__main__":
    unittest.main()

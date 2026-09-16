from __future__ import annotations

import unittest

import torch

from src.losa.generation import _should_stop_sdar_denoising as should_stop_losa
from src.losa_v2.generation import _should_stop_sdar_denoising as should_stop_losa_v2


class SDARDenoisingEarlyStopTest(unittest.TestCase):
    def test_non_sequential_stops_when_all_masks_are_resolved(self):
        resolved = torch.zeros((1, 32), dtype=torch.bool)
        for should_stop in (should_stop_losa, should_stop_losa_v2):
            self.assertTrue(
                should_stop(
                    step=3,
                    steps=32,
                    remasking_strategy="confidence",
                    active_mask=resolved,
                )
            )

    def test_non_sequential_continues_while_masks_remain(self):
        active = torch.tensor([[False, True, False]])
        for should_stop in (should_stop_losa, should_stop_losa_v2):
            self.assertFalse(
                should_stop(
                    step=3,
                    steps=32,
                    remasking_strategy="confidence",
                    active_mask=active,
                )
            )

    def test_sequential_preserves_fixed_step_behavior(self):
        resolved = torch.zeros((1, 32), dtype=torch.bool)
        for should_stop in (should_stop_losa, should_stop_losa_v2):
            self.assertFalse(
                should_stop(
                    step=31,
                    steps=32,
                    remasking_strategy="sequential",
                    active_mask=resolved,
                )
            )
            self.assertTrue(
                should_stop(
                    step=32,
                    steps=32,
                    remasking_strategy="sequential",
                    active_mask=resolved,
                )
            )


if __name__ == "__main__":
    unittest.main()

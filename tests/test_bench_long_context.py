import argparse
import unittest

import torch

from scripts.original.performance.long_context_benchmark import exact_prompt, generation_kwargs


class NarrativePromptTest(unittest.TestCase):
    def test_middle_truncates_real_prompt(self):
        source = torch.arange(10).unsqueeze(0)
        actual = exact_prompt(None, 6, source.device, source)
        self.assertEqual(actual.tolist(), [[0, 1, 2, 7, 8, 9]])

    def test_full_mask_flag_is_forwarded_only_for_llada(self):
        args = argparse.Namespace(
            model="llada", gen_length=32, block_length=32, steps=None,
            remasking_strategy="sequential", llada_full_mask=True, ablation=False,
        )
        tokenizer = argparse.Namespace(mask_token_id=1)
        self.assertFalse(generation_kwargs(args, tokenizer, None)["maskless_attention"])

    def test_llada_ablation_is_full_mask_only_for_dense_control(self):
        args = argparse.Namespace(
            model="llada", mode="query_prefix", runtime_mode="dense",
            gen_length=32, block_length=32, steps=None,
            remasking_strategy="sequential", llada_full_mask=False, ablation=True,
        )
        tokenizer = argparse.Namespace(mask_token_id=1)
        self.assertFalse(generation_kwargs(args, tokenizer, None)["maskless_attention"])
        args.runtime_mode = "maskless_dense"
        self.assertTrue(generation_kwargs(args, tokenizer, None)["maskless_attention"])


if __name__ == "__main__":
    unittest.main()

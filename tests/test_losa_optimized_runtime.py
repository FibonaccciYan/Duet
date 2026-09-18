from __future__ import annotations

import unittest

import torch

from src.optimized.losa.operators import (
    GQAMode,
    OptimizedPageMetadata,
    compact_prefix_attention,
    compact_selected_pages,
    dense_attention,
    locality_scores,
    quest_page_scores_grouped,
)
from src.optimized.losa.triton_ops import (
    compact_prefix_attention_triton,
    compact_selected_pages_triton,
    dense_attention_triton,
    locality_scores_triton,
    quest_group_mean_scores_triton,
)
from src.runtime import load_runtime


class LoSAOptimizedRuntimeTest(unittest.TestCase):
    def test_runtime_is_registered(self):
        runtime = load_runtime("losa_optimized", family="llada")
        self.assertEqual(type(runtime).__name__, "LoSAOptimizedRuntime")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_triton_fast_paths_match_oracle(self):
        torch.manual_seed(7)
        device = torch.device("cuda")
        dtype = torch.float16
        rows, num_q, num_kv, dim = 5, 8, 2, 32
        page_size, prefix_length = 8, 123
        num_pages = (prefix_length + page_size - 1) // page_size
        q = torch.randn(rows, num_q, dim, device=device, dtype=dtype)
        previous = q * 0.9
        k = torch.randn(prefix_length, num_kv, dim, device=device, dtype=dtype)
        v = torch.randn_like(k)

        padded = torch.cat(
            (
                k,
                torch.zeros(
                    num_pages * page_size - prefix_length,
                    num_kv,
                    dim,
                    device=device,
                    dtype=dtype,
                ),
            )
        ).reshape(num_pages, page_size, num_kv, dim)
        valid = (
            torch.arange(num_pages * page_size, device=device)
            .reshape(num_pages, page_size)
            .lt(prefix_length)
        )
        k_min = padded.masked_fill(
            ~valid[:, :, None, None], float("inf")
        ).amin(dim=1)
        k_max = padded.masked_fill(
            ~valid[:, :, None, None], float("-inf")
        ).amax(dim=1)
        metadata = OptimizedPageMetadata(k_min, k_max, prefix_length, page_size)

        torch.testing.assert_close(
            locality_scores_triton(q, previous),
            locality_scores(q, previous),
            atol=2e-5,
            rtol=2e-5,
        )
        expected_scores = quest_page_scores_grouped(
            q, metadata, GQAMode.GROUP_MEAN
        )
        actual_scores = quest_group_mean_scores_triton(q, k_min, k_max)
        torch.testing.assert_close(
            actual_scores, expected_scores, atol=2e-3, rtol=2e-3
        )

        selected = actual_scores.topk(4, dim=-1).indices
        expected_pages = compact_selected_pages(
            selected,
            num_pages,
            num_kv_heads=num_kv,
            group_size=num_q // num_kv,
        )
        actual_pages = compact_selected_pages_triton(
            selected,
            num_pages,
            num_kv_heads=num_kv,
            group_size=num_q // num_kv,
        )
        torch.testing.assert_close(actual_pages, expected_pages)

        per_query_selected = torch.randint(
            0,
            num_pages,
            (rows, num_q, 4),
            device=device,
            dtype=torch.int64,
        )
        expected_per_query_pages = compact_selected_pages(
            per_query_selected,
            num_pages,
            num_kv_heads=num_kv,
            group_size=num_q // num_kv,
        )
        actual_per_query_pages = compact_selected_pages_triton(
            per_query_selected,
            num_pages,
            num_kv_heads=num_kv,
            group_size=num_q // num_kv,
        )
        torch.testing.assert_close(
            actual_per_query_pages, expected_per_query_pages
        )

        expected_out, expected_lse = compact_prefix_attention(
            q,
            k,
            v,
            actual_pages,
            page_size=page_size,
            prefix_length=prefix_length,
        )
        actual_out, actual_lse = compact_prefix_attention_triton(
            q,
            k,
            v,
            actual_pages,
            page_size=page_size,
            prefix_length=prefix_length,
        )
        torch.testing.assert_close(
            actual_out, expected_out, atol=2e-2, rtol=2e-2
        )
        torch.testing.assert_close(
            actual_lse, expected_lse, atol=2e-2, rtol=2e-2
        )

        block_k = torch.randn(17, num_kv, dim, device=device, dtype=dtype)
        block_v = torch.randn_like(block_k)
        expected_block = dense_attention(q, block_k, block_v)
        actual_block = dense_attention_triton(q, block_k, block_v)
        torch.testing.assert_close(
            actual_block[0], expected_block[0], atol=2e-2, rtol=2e-2
        )
        torch.testing.assert_close(
            actual_block[1], expected_block[1], atol=2e-2, rtol=2e-2
        )


if __name__ == "__main__":
    unittest.main()

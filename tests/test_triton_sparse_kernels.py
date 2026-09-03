import unittest

import torch

from src.sparse.triton_kernels import (
    TRITON_AVAILABLE,
    adamas_distances,
    attention_output_lse,
    losa_query_delta,
)


@unittest.skipUnless(
    torch.cuda.is_available() and TRITON_AVAILABLE,
    "requires CUDA and Triton",
)
class TritonSparseKernelsTest(unittest.TestCase):
    def test_adamas_distances_match_broadcast_reference(self):
        torch.manual_seed(0)
        query = torch.randint(
            0, 4, (1, 8, 5, 128), device="cuda", dtype=torch.int32
        )
        full_key = torch.randint(
            0, 4, (1, 2, 23, 128), device="cuda", dtype=torch.int32
        )
        key = full_key[:, :, 3:20]
        actual = adamas_distances(query, key)
        groups = query.shape[1] // key.shape[1]
        expected = (
            query.reshape(1, key.shape[1], groups, query.shape[2], 128)[
                ..., None, :
            ]
            - key[:, :, None, None]
        ).abs().sum(dim=-1)[0].reshape(-1, key.shape[2])
        torch.testing.assert_close(actual, expected)

    def test_losa_query_delta_matches_mse_reference(self):
        torch.manual_seed(1)
        query = torch.randn(1, 8, 7, 128, device="cuda", dtype=torch.float16)
        previous = torch.randn(1, 8, 16, 128, device="cuda", dtype=torch.float16)
        positions = torch.tensor([0, 3, 5, 7, 9, 12, 15], device="cuda")
        actual = losa_query_delta(query, previous, positions)
        expected = (
            query - previous.index_select(2, positions)
        ).float().pow(2).mean(dim=(1, 3))[0]
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-4)

    def test_attention_output_and_lse_match_reference(self):
        torch.manual_seed(2)
        query = torch.randn(1, 8, 5, 128, device="cuda", dtype=torch.float16)
        key = torch.randn(1, 2, 257, 128, device="cuda", dtype=torch.float16)
        value = torch.randn_like(key)
        mask = torch.ones(1, 1, 5, 257, device="cuda", dtype=torch.bool)
        mask[..., -7:] = False
        actual, actual_lse = attention_output_lse(query, key, value, mask)

        repeated_key = key.repeat_interleave(4, dim=1)
        repeated_value = value.repeat_interleave(4, dim=1)
        scores = torch.matmul(query, repeated_key.transpose(-2, -1))
        scores = scores * (query.shape[-1] ** -0.5)
        scores = scores.masked_fill(~mask, -torch.inf)
        expected_lse = torch.logsumexp(scores.float(), dim=-1)
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        expected = torch.matmul(weights, repeated_value)
        torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(actual_lse, expected_lse, rtol=2e-3, atol=2e-3)

    def test_bfloat16_attention_keeps_reference_path(self):
        query = torch.randn(1, 4, 3, 128, device="cuda", dtype=torch.bfloat16)
        key = torch.randn(1, 1, 17, 128, device="cuda", dtype=torch.bfloat16)
        mask = torch.zeros(1, 1, 3, 17, device="cuda", dtype=torch.bfloat16)
        self.assertIsNone(attention_output_lse(query, key, key, mask))

if __name__ == "__main__":
    unittest.main()

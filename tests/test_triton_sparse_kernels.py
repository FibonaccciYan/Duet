import unittest

import torch

from src.sparse.triton_kernels import (
    _route_moe,
    adamas_distances,
    attention_output_lse,
    block_causal_prefill,
    fused_swiglu,
    losa_query_delta,
    rms_norm,
    rotary_embedding,
)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TritonSparseKernelsTest(unittest.TestCase):
    def test_rms_norm_matches_reference(self):
        torch.manual_seed(5)
        hidden = torch.randn(
            1, 7, 4096, device="cuda", dtype=torch.float16
        )
        weight = torch.randn(4096, device="cuda", dtype=torch.float16)
        actual = rms_norm(hidden, weight, 1e-6)
        expected = torch.nn.functional.rms_norm(
            hidden, (hidden.shape[-1],), weight, 1e-6
        )

        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)

    def test_fused_swiglu_matches_linear_reference(self):
        torch.manual_seed(4)
        hidden = torch.randn(1, 7, 64, device="cuda", dtype=torch.float16) * 0.1
        gate = torch.randn(128, 64, device="cuda", dtype=torch.float16) * 0.1
        up = torch.randn_like(gate) * 0.1
        actual = fused_swiglu(hidden, gate, up)
        expected = torch.nn.functional.silu(
            torch.nn.functional.linear(hidden, gate)
        ) * torch.nn.functional.linear(hidden, up)

        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)

    def test_moe_route_matches_expert_groups(self):
        expert_ids = torch.tensor([2, 0, 1, 2, 1, 1], device="cuda")
        counts, offsets, order = _route_moe(expert_ids, 4)

        self.assertEqual(counts.tolist(), [1, 3, 2, 0])
        self.assertEqual(offsets.tolist(), [0, 1, 4, 6, 6])
        self.assertEqual(expert_ids.index_select(0, order).tolist(), [0, 1, 1, 1, 2, 2])

    def test_rotary_preserves_rounding_and_strides(self):
        for dtype in (torch.float16, torch.bfloat16):
            x = torch.randn(1, 7, 4, 128, device="cuda", dtype=dtype).transpose(1, 2)
            cos = torch.randn(1, 7, 64, device="cuda", dtype=dtype)
            sin = torch.randn_like(cos)
            a = x[..., :64]
            rotated = torch.cat((-a[..., 32:], a[..., :32]), -1)
            expected = torch.cat((a*cos[:, None] + rotated*sin[:, None], x[..., 64:]), -1)
            torch.testing.assert_close(rotary_embedding(x, cos, sin), expected, rtol=0, atol=0)

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
        self.assertEqual(actual.dtype, torch.int32)
        groups = query.shape[1] // key.shape[1]
        expected = (
            query.reshape(1, key.shape[1], groups, query.shape[2], 128)[
                ..., None, :
            ]
            - key[:, :, None, None]
        ).abs().sum(dim=-1)[0].reshape(-1, key.shape[2]).to(torch.int32)
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

    def test_losa_query_delta_supports_key_weights(self):
        query = torch.randn(1, 4, 7, 16, device="cuda", dtype=torch.bfloat16)
        previous = torch.randn(1, 4, 12, 16, device="cuda", dtype=torch.bfloat16)
        positions = torch.tensor([0, 2, 3, 5, 7, 9, 11], device="cuda")
        weights = torch.rand(4, 16, device="cuda")

        actual = losa_query_delta(query, previous, positions, weights=weights)
        difference = query.float() - previous.index_select(2, positions).float()
        expected = (difference.square() * weights[None, :, None, :]).mean((1, 3))[0]

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

    def test_block_causal_prefill_matches_explicit_mask(self):
        torch.manual_seed(3)
        query = torch.randn(1, 8, 64, 128, device="cuda", dtype=torch.float16)
        key = torch.randn(1, 2, 128, 128, device="cuda", dtype=torch.float16)
        value = torch.randn_like(key)
        for block_length in (4, 32):
            query_blocks = torch.arange(64, 128, device="cuda") // block_length
            key_blocks = torch.arange(128, device="cuda") // block_length
            mask = (key_blocks[None, :] <= query_blocks[:, None])[None, None]

            expected, _ = attention_output_lse(query, key, value, mask)
            actual = block_causal_prefill(query, key, value, block_length)

            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_bfloat16_attention_matches_reference(self):
        query = torch.randn(1, 4, 3, 128, device="cuda", dtype=torch.bfloat16)
        key = torch.randn(1, 1, 257, 128, device="cuda", dtype=torch.bfloat16)
        mask = torch.zeros(1, 1, 3, 257, device="cuda", dtype=torch.bfloat16)
        output, lse = attention_output_lse(query, key, key, mask)

        repeated_key = key.repeat_interleave(4, dim=1)
        scores = torch.matmul(query, repeated_key.transpose(-2, -1)) / 128**0.5
        expected_lse = torch.logsumexp(scores.float(), dim=-1)
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        expected = torch.matmul(weights, repeated_key)
        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(lse, expected_lse, rtol=2e-3, atol=2e-3)

if __name__ == "__main__":
    unittest.main()

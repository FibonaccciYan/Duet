import unittest
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sparse.moe_expert_patch import _TRITON_AVAILABLE, patch_moe_experts


class _TinyExpert(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)


class _TinyMoE(nn.Module):
    def __init__(self, num_experts=4, hidden_size=32, intermediate_size=16):
        super().__init__()
        self.experts = nn.ModuleList(
            [_TinyExpert(hidden_size, intermediate_size) for _ in range(num_experts)]
        )
        self.moe_infer = self._reference_moe_infer

    def _reference_moe_infer(self, x, topk_ids, topk_weight):
        topk_ids = topk_ids.reshape(-1, topk_ids.shape[-1])
        topk_weight = topk_weight.reshape_as(topk_ids)
        result = torch.zeros_like(x)
        flat_ids = topk_ids.flatten()
        flat_weights = topk_weight.flatten()
        token_ids = torch.arange(x.shape[0], device=x.device).repeat_interleave(topk_ids.shape[1])
        for expert_id, expert in enumerate(self.experts):
            selected = flat_ids == expert_id
            if not selected.any():
                continue
            token_x = x.index_select(0, token_ids[selected])
            gate = expert.gate_proj(token_x)
            up = expert.up_proj(token_x)
            expert_out = expert.down_proj(torch.nn.functional.silu(gate) * up)
            result.index_add_(
                0,
                token_ids[selected],
                expert_out * flat_weights[selected].to(expert_out.dtype).unsqueeze(-1),
            )
        return result


@unittest.skipUnless(
    torch.cuda.is_available() and _TRITON_AVAILABLE,
    "requires CUDA and Triton",
)
class MoEExpertPatchTest(unittest.TestCase):
    def test_packed_kernel_matches_reference(self):
        torch.manual_seed(0)
        model = _TinyMoE().cuda().to(dtype=torch.bfloat16).eval()
        x = torch.randn(7, 32, device="cuda", dtype=torch.bfloat16)
        topk_ids = torch.tensor(
            [[0, 1], [0, 3], [2, 1], [3, 0], [2, 3], [1, 2], [0, 3]],
            device="cuda",
        )
        topk_weight = torch.tensor(
            [[0.6, 0.4], [0.7, 0.3], [0.2, 0.8], [0.5, 0.5],
             [0.4, 0.6], [0.9, 0.1], [0.3, 0.7]],
            device="cuda",
            dtype=torch.float32,
        )

        with torch.inference_mode():
            expected = model.moe_infer(x, topk_ids, topk_weight)
            self.assertEqual(patch_moe_experts(model), 1)
            actual = model.moe_infer(x, topk_ids, topk_weight)

        torch.testing.assert_close(actual, expected, rtol=5e-2, atol=2e-3)


if __name__ == "__main__":
    unittest.main()

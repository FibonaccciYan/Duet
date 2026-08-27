import types
import unittest
from unittest import mock

import torch

from src.sparse.sdar_block_diffusion_patch import (
    _sample_with_confidence,
    _select_transfer,
    _sparse_cached_forward,
    patch_model,
)


class _FakeSDAR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.config = types.SimpleNamespace(model_type="sdar")
        self.generation_config = types.SimpleNamespace(eos_token_id=[14, 15])
        self.generate = lambda *args, **kwargs: None
        self.store_calls = 0

    @property
    def device(self):
        return self.anchor.device

    def forward(self, input_ids, attention_mask, store_kv=False, **kwargs):
        self.assert_attention_shape = (input_ids.shape[1], attention_mask.shape[-1])
        self.store_calls += int(store_kv)
        logits = torch.zeros(*input_ids.shape, 16, device=input_ids.device)
        logits[..., 2] = 10
        return types.SimpleNamespace(logits=logits)


class SDARBlockDiffusionPatchTest(unittest.TestCase):
    def test_sampling_confidence_uses_probability_before_top_k(self):
        logits = torch.tensor([[[2.0, 1.0]]])
        token, confidence = _sample_with_confidence(
            logits, temperature=1.0, top_k=1, top_p=1.0
        )

        self.assertEqual(token.item(), 0)
        torch.testing.assert_close(
            confidence,
            torch.softmax(logits, dim=-1)[..., 0],
        )

    def test_layer2_selector_matches_final_logit_postprocessing(self):
        class ShiftLayer(torch.nn.Module):
            def forward(self, hidden_states, **kwargs):
                return (hidden_states + 1,)

        class Base:
            layers = [ShiftLayer(), ShiftLayer(), ShiftLayer()]
            norm = staticmethod(lambda hidden_states: hidden_states * 10)
            embed_tokens = staticmethod(
                lambda input_ids: input_ids.float().unsqueeze(-1)
            )
            rotary_emb = staticmethod(
                lambda hidden_states, position_ids: (
                    torch.ones_like(hidden_states),
                    torch.zeros_like(hidden_states),
                )
            )

        model = types.SimpleNamespace(
            model=Base(),
            lm_head=torch.nn.Identity(),
            config=types.SimpleNamespace(
                sdar_sparse_dlm_ratio=0.5,
                sdar_sparse_dlm_selection_interval=1,
                sdar_sparse_dlm_dense_fallback_mask_count=0,
            ),
        )
        input_ids = torch.tensor([[1, 15]])
        with mock.patch(
            "src.sparse.sdar_block_diffusion_patch._select_positions",
            return_value=None,
        ) as select_positions:
            _sparse_cached_forward(
                model,
                input_ids,
                torch.arange(2).unsqueeze(0),
                (),
                {"sparse_cache": None, "step": 0},
                mask_id=15,
                temperature=0.7,
                top_k=7,
                top_p=0.9,
            )

        selection_hidden = select_positions.call_args.args[1]
        torch.testing.assert_close(
            selection_hidden,
            (input_ids.float().unsqueeze(-1) + 2) * 10,
        )
        self.assertEqual(select_positions.call_args.kwargs["top_k"], 7)

    def test_sparse_forward_returns_only_selected_mask_logits(self):
        class ShiftLayer(torch.nn.Module):
            def forward(self, hidden_states, **kwargs):
                return (hidden_states + 1,)

        class Base:
            layers = [ShiftLayer(), ShiftLayer(), ShiftLayer()]
            norm = staticmethod(lambda hidden_states: hidden_states)
            embed_tokens = staticmethod(
                lambda input_ids: input_ids.float().unsqueeze(-1)
            )
            rotary_emb = staticmethod(
                lambda hidden_states, position_ids: (
                    torch.ones_like(hidden_states),
                    torch.zeros_like(hidden_states),
                )
            )

        model = types.SimpleNamespace(
            model=Base(),
            lm_head=torch.nn.Identity(),
            config=types.SimpleNamespace(
                sdar_sparse_dlm_ratio=0.5,
                sdar_sparse_dlm_selection_interval=1,
                sdar_sparse_dlm_dense_fallback_mask_count=0,
            ),
        )
        input_ids = torch.tensor([[1, 15, 15]])
        with mock.patch(
            "src.sparse.sdar_block_diffusion_patch._select_positions",
            return_value=torch.tensor([0, 2]),
        ):
            logits, logit_positions = _sparse_cached_forward(
                model,
                input_ids,
                torch.arange(3).unsqueeze(0),
                (),
                {"sparse_cache": mock.Mock(), "step": 0},
                mask_id=15,
                temperature=0.0,
                top_k=7,
                top_p=1.0,
            )

        self.assertEqual(logit_positions.tolist(), [2])
        torch.testing.assert_close(logits, torch.tensor([[[18.0]]]))

    def test_transfer_ignores_masks_without_full_depth_logits(self):
        transfer = _select_transfer(
            mask=torch.tensor([[True, True, True]]),
            confidence=torch.tensor([[-torch.inf, 0.8, -torch.inf]]),
            minimum=3,
            strategy="low_confidence_static",
            threshold=1.0,
        )

        self.assertEqual(transfer.tolist(), [[False, True, False]])

    def test_patch_generates_blocks_and_stores_each_completed_block(self):
        model = patch_model(_FakeSDAR(), query_sparse=False)
        output = model.generate(
            inputs=torch.tensor([[7, 8, 9]]),
            gen_length=4,
            block_length=4,
            steps=4,
            temperature=0,
            threshold=0.85,
            mask_id=15,
            eos_id=14,
        )

        self.assertEqual(output.tolist(), [[2, 2, 2, 2]])
        self.assertEqual(model.store_calls, 2)
        self.assertEqual(model.assert_attention_shape, (4, 8))


if __name__ == "__main__":
    unittest.main()

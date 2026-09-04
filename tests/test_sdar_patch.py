import types
import unittest
import weakref
from unittest import mock

import torch
from transformers.cache_utils import DynamicCache

from src.sparse.sparse_ops import (
    _apply_rotary,
    _attention_output_lse,
    _new_losa_state,
)
from src.sparse.sdar_patch import (
    _sample_with_confidence,
    _sdar_losa_attention_forward,
    _select_transfer,
    _sparse_cached_forward,
    patch_model,
)
from src.sparse.sdar_generate import entropy_from_logits


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
    def test_prefix_sparse_accepts_per_layer_compact_lengths(self):
        seen_key_lengths = []

        class RecordLayer(torch.nn.Module):
            def forward(self, hidden_states, attention_mask, **kwargs):
                seen_key_lengths.append(attention_mask.shape[-1])
                return (hidden_states,)

        class Base:
            layers = [RecordLayer(), RecordLayer()]
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
            config=types.SimpleNamespace(sdar_losa=False),
        )
        prefix_cache = (
            (torch.zeros(1, 1, 2, 1), torch.zeros(1, 1, 2, 1)),
            (torch.zeros(1, 1, 3, 1), torch.zeros(1, 1, 3, 1)),
        )
        _sparse_cached_forward(
            model,
            torch.tensor([[1, 2]]),
            torch.arange(2).unsqueeze(0),
            prefix_cache,
            {"sparse_cache": None, "step": 0},
            mask_id=15,
            temperature=0.0,
            top_k=7,
            top_p=1.0,
            query_sparse=False,
        )

        self.assertEqual(seen_key_lengths, [4, 5])

    def test_patch_exposes_sdar_prefix_sparse_configuration(self):
        model = patch_model(
            _FakeSDAR(),
            query_sparse=False,
            prefix_sparse=True,
            prefix_token_budget=17,
            prefix_chunk_size=9,
        )

        self.assertTrue(model.config.sdar_prefix_sparse)
        self.assertEqual(model.config.sdar_prefix_token_budget, 17)
        self.assertEqual(model.config.sdar_prefix_chunk_size, 9)

    def test_losa_patch_does_not_register_parent_as_attention_child(self):
        class Attention(torch.nn.Module):
            def forward(self, *args, **kwargs):
                return None

        model = _FakeSDAR()
        model.model = torch.nn.Module()
        layer = torch.nn.Module()
        layer.self_attn = Attention()
        model.model.layers = torch.nn.ModuleList([layer])

        patch_model(model, query_sparse=False, losa=True)
        model.eval()

        self.assertEqual(list(layer.self_attn.children()), [])
        self.assertIs(layer.self_attn._sdar_losa_model_ref(), model)

    def test_losa_full_active_budget_matches_dense_attention(self):
        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_idx = 0
                self.head_dim = 2
                self.num_attention_heads = 1
                self.num_key_value_heads = 1
                self.num_key_value_groups = 1
                self.q_proj = torch.nn.Linear(2, 2, bias=False)
                self.k_proj = torch.nn.Linear(2, 2, bias=False)
                self.v_proj = torch.nn.Linear(2, 2, bias=False)
                self.o_proj = torch.nn.Linear(2, 2, bias=False)
                self.q_norm = torch.nn.Identity()
                self.k_norm = torch.nn.Identity()
                for projection in (
                    self.q_proj,
                    self.k_proj,
                    self.v_proj,
                    self.o_proj,
                ):
                    projection.weight.data.copy_(torch.eye(2))

            def dense(
                self,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_value=None,
                store_kv=False,
                **kwargs,
            ):
                batch, length, _ = hidden_states.shape
                query = self.q_proj(hidden_states).view(batch, length, 1, 2)
                key = self.k_proj(hidden_states).view(batch, length, 1, 2)
                value = self.v_proj(hidden_states).view(batch, length, 1, 2)
                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)
                cos, sin = position_embeddings
                query = _apply_rotary(query, cos, sin)
                key = _apply_rotary(key, cos, sin)
                if past_key_value is not None and store_kv:
                    key, value = past_key_value.update(key, value, 0)
                mask = attention_mask.unsqueeze(1)
                output, _ = _attention_output_lse(
                    query, key, value, mask, self.num_key_value_groups
                )
                output = output.transpose(1, 2).reshape(batch, length, 2)
                return self.o_proj(output), None

        attention = Attention()
        selection_state = {"losa_states": {}}
        context = {
            "selection_state": selection_state,
            "prefix_cache_length": 2,
            "active_topk": 2,
            "pending_losa": [],
            "pending_losa_queries": [],
            "query_positions": torch.arange(2),
        }
        class Context:
            pass

        model = Context()
        model._sdar_losa_context = context
        attention._sdar_losa_model_ref = weakref.ref(model)
        attention._sdar_losa_dense_forward = attention.dense
        prefix = (
            torch.tensor([[[[0.5, -0.5], [1.0, 0.25]]]]),
            torch.tensor([[[[0.25, 1.0], [-0.5, 0.5]]]]),
        )
        hidden = torch.tensor([[[1.0, 0.5], [-0.25, 0.75]]])
        cos = torch.ones(1, 2, 2)
        sin = torch.zeros_like(cos)
        mask = torch.ones(1, 2, 4, dtype=torch.bool)

        first_cache = DynamicCache.from_legacy_cache((prefix,))
        first = _sdar_losa_attention_forward(
            attention,
            hidden,
            (cos, sin),
            mask,
            past_key_value=first_cache,
            store_kv=True,
        )[0]
        dense_first = attention.dense(
            hidden,
            (cos, sin),
            mask,
            past_key_value=DynamicCache.from_legacy_cache((prefix,)),
            store_kv=True,
        )[0]
        torch.testing.assert_close(first, dense_first)

        _, positions, query = context["pending_losa_queries"][0]
        state = _new_losa_state(query, 2)
        state["previous_query"].index_copy_(2, positions, query)
        _, positions, prefix_output, prefix_lse = context["pending_losa"][0]
        state["prefix_output"].index_copy_(2, positions, prefix_output)
        state["prefix_lse"].index_copy_(2, positions, prefix_lse)
        state["valid"][0, positions] = True
        selection_state["losa_states"][0] = state
        context["pending_losa"].clear()
        context["pending_losa_queries"].clear()

        changed_hidden = hidden + torch.tensor([[[0.1, 0.0], [0.0, -0.1]]])
        losa_output = _sdar_losa_attention_forward(
            attention,
            changed_hidden,
            (cos, sin),
            mask,
            past_key_value=DynamicCache.from_legacy_cache((prefix,)),
            store_kv=True,
        )[0]
        dense_output = attention.dense(
            changed_hidden,
            (cos, sin),
            mask,
            past_key_value=DynamicCache.from_legacy_cache((prefix,)),
            store_kv=True,
        )[0]
        torch.testing.assert_close(losa_output, dense_output, atol=1e-6, rtol=1e-6)

    def test_sampling_confidence_matches_official_filtered_probability(self):
        logits = torch.tensor([[[2.0, 1.0]]])
        token, confidence = _sample_with_confidence(
            logits, temperature=1.0, top_k=1, top_p=1.0
        )

        self.assertEqual(token.item(), 0)
        torch.testing.assert_close(confidence, torch.ones_like(confidence))

    def test_greedy_sampling_is_available_for_deterministic_eval(self):
        logits = torch.tensor([[[2.0, 1.0]]])
        token, confidence = _sample_with_confidence(
            logits, temperature=0.0, top_k=1, top_p=0.5
        )

        self.assertEqual(token.item(), 0)
        torch.testing.assert_close(
            confidence,
            torch.softmax(logits, dim=-1)[..., 0],
        )

    def test_entropy_is_computed_over_each_position_vocab(self):
        logits = torch.tensor([[[0.0, 0.0], [10.0, -10.0]]])

        entropy = entropy_from_logits(logits, temperature=0.0)

        torch.testing.assert_close(
            entropy,
            torch.tensor([[torch.log(torch.tensor(2.0)), 0.0]]),
            atol=1e-6,
            rtol=0,
        )

        unfiltered = entropy_from_logits(
            logits[:, :1], temperature=1.0, top_k=None, top_p=None
        )
        torch.testing.assert_close(
            unfiltered,
            torch.tensor([[torch.log(torch.tensor(2.0))]]),
            atol=1e-6,
            rtol=0,
        )

    def test_entropy_bounded_transfer_uses_lowest_cumulative_entropy(self):
        transfer = _select_transfer(
            mask=torch.tensor([[True, True, True]]),
            confidence=torch.zeros(1, 3),
            minimum=1,
            strategy="entropy_bounded",
            threshold=1.0,
            entropy=torch.tensor([[0.1, 0.2, 0.8]]),
            entropy_budget=0.35,
        )

        self.assertEqual(transfer.tolist(), [[True, True, False]])

    def test_layer6_selector_matches_final_logit_postprocessing(self):
        class ShiftLayer(torch.nn.Module):
            def forward(self, hidden_states, **kwargs):
                return (hidden_states + 1,)

        class Base:
            layers = [ShiftLayer() for _ in range(7)]
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
            "src.sparse.sdar_patch._select_positions",
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
                strategy="entropy_bounded",
                threshold=0.6,
                entropy_budget=0.35,
            )

        selection_hidden = select_positions.call_args.args[1]
        torch.testing.assert_close(
            selection_hidden,
            (input_ids.float().unsqueeze(-1) + 6) * 10,
        )
        self.assertEqual(select_positions.call_args.kwargs["top_k"], 7)
        self.assertEqual(
            select_positions.call_args.kwargs["strategy"], "entropy_bounded"
        )
        self.assertEqual(select_positions.call_args.kwargs["threshold"], 0.6)
        self.assertEqual(select_positions.call_args.kwargs["entropy_budget"], 0.35)

    def test_sparse_forward_keeps_layer6_logits_for_unselected_masks(self):
        class ShiftLayer(torch.nn.Module):
            def forward(self, hidden_states, **kwargs):
                return (hidden_states + 1,)

        class Base:
            layers = [ShiftLayer() for _ in range(7)]
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
            "src.sparse.sdar_patch._select_positions",
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

        self.assertEqual(logit_positions.tolist(), [1, 2])
        torch.testing.assert_close(logits, torch.tensor([[[21.0], [22.0]]]))

        with mock.patch(
            "src.sparse.sdar_patch._select_positions",
            return_value=torch.tensor([0, 2]),
        ):
            deep_logits, deep_positions = _sparse_cached_forward(
                model,
                input_ids,
                torch.arange(3).unsqueeze(0),
                (),
                {"sparse_cache": mock.Mock(), "step": 0},
                mask_id=15,
                temperature=0.0,
                top_k=7,
                top_p=1.0,
                deep_only_transfer=True,
            )

        self.assertEqual(deep_positions.tolist(), [2])
        torch.testing.assert_close(deep_logits, torch.tensor([[[22.0]]]))

    def test_sparse_forward_reuses_late_layer_inputs(self):
        seen = []

        class RecordLayer(torch.nn.Module):
            def forward(
                self,
                hidden_states,
                position_ids,
                position_embeddings,
                attention_mask,
                **kwargs,
            ):
                seen.append(
                    (
                        hidden_states.shape[1],
                        id(position_ids),
                        id(position_embeddings[0]),
                        id(position_embeddings[1]),
                        id(attention_mask),
                    )
                )
                return (hidden_states + 1,)

        class Base:
            layers = [RecordLayer() for _ in range(9)]
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
        with mock.patch(
            "src.sparse.sdar_patch._select_positions",
            return_value=torch.tensor([0, 2]),
        ):
            _sparse_cached_forward(
                model,
                torch.tensor([[1, 15, 15]]),
                torch.arange(3).unsqueeze(0),
                (),
                {"sparse_cache": mock.Mock(), "step": 0},
                mask_id=15,
                temperature=0.0,
                top_k=7,
                top_p=1.0,
            )

        self.assertEqual([entry[0] for entry in seen], [3] * 6 + [2] * 3)
        self.assertEqual(len({entry[1:] for entry in seen[:6]}), 1)
        self.assertEqual(len({entry[1:] for entry in seen[6:]}), 1)

    def test_transfer_ignores_masks_without_logits(self):
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

    def test_query_sparse_uses_dense_first_step(self):
        model = patch_model(
            _FakeSDAR(),
            query_sparse=True,
            top_k=7,
            selection_layer=4,
            deep_only_transfer=True,
        )
        self.assertEqual(model.config.sdar_query_selection_layer, 4)
        self.assertTrue(model.config.sdar_sparse_dlm_deep_only_transfer)

        def sparse_forward(model, block_tokens, *args, **kwargs):
            positions = torch.where(block_tokens[0] == 15)[0]
            logits = torch.zeros(1, positions.numel(), 16)
            logits[..., 2] = 10
            return logits, positions

        with mock.patch(
            "src.sparse.sdar_patch._sparse_cached_forward",
            side_effect=sparse_forward,
        ) as sparse:
            output = model.generate(
                inputs=torch.tensor([[7, 8, 9]]),
                gen_length=4,
                block_length=4,
                steps=4,
                temperature=0,
                threshold=1.0,
                mask_id=15,
                eos_id=14,
            )

        self.assertEqual(output.tolist(), [[2, 2, 2, 2]])
        self.assertEqual(sparse.call_count, 3)
        self.assertTrue(all(call.args[7] == 7 for call in sparse.call_args_list))
        self.assertFalse(
            any(call.kwargs["refresh_late_kv"] for call in sparse.call_args_list)
        )
        self.assertEqual(model.store_calls, 4)

    def test_large_block_refreshes_late_kv_once(self):
        model = patch_model(_FakeSDAR(), query_sparse=True, refresh_step=2)

        def sparse_forward(model, block_tokens, *args, **kwargs):
            positions = torch.where(block_tokens[0] == 15)[0]
            logits = torch.zeros(1, positions.numel(), 16)
            logits[..., 2] = 10
            return logits, positions

        with mock.patch(
            "src.sparse.sdar_patch._sparse_cached_forward",
            side_effect=sparse_forward,
        ) as sparse:
            model.generate(
                inputs=torch.full((1, 16), 7),
                gen_length=16,
                block_length=16,
                steps=16,
                temperature=0,
                threshold=1.0,
                mask_id=15,
                eos_id=14,
            )

        refresh_calls = [
            call.kwargs["refresh_late_kv"] for call in sparse.call_args_list
        ]
        self.assertEqual(sum(refresh_calls), 1)
        self.assertTrue(refresh_calls[2])

        model = patch_model(_FakeSDAR(), query_sparse=True, refresh_step=-2)
        with mock.patch(
            "src.sparse.sdar_patch._sparse_cached_forward",
            side_effect=sparse_forward,
        ) as sparse:
            model.generate(
                inputs=torch.full((1, 16), 7),
                gen_length=16,
                block_length=16,
                steps=16,
                temperature=0,
                threshold=1.0,
                mask_id=15,
                eos_id=14,
            )

        self.assertTrue(
            all(call.kwargs["refresh_late_kv"] for call in sparse.call_args_list)
        )


if __name__ == "__main__":
    unittest.main()

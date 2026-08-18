import argparse
import os
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from local_demo_sparse_patch import patch_model as patch_llada_demo_sparse
from src.sparse.block_cache_sparse_dlm_patch import patch_model as patch_llada_block_cache_sparse


def parse_args():
    parser = argparse.ArgumentParser(description="LLaDA layer-wise query position recall test")
    parser.add_argument("--model_path", type=str, default="/data0/ysy/models/LLaDA2.1-mini")
    parser.add_argument("--prompt", type=str, default="Write a short story about history.")
    parser.add_argument("--gen_length", type=int, default=int(os.environ.get("GEN_LENGTH", "512")))
    parser.add_argument("--block_length", type=int, default=int(os.environ.get("BLOCK_LENGTH", "32")))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("STEPS", "32")))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--editing_threshold", type=float, default=0.0)
    parser.add_argument("--num_to_transfer", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--use_model_generate", action="store_true")
    parser.add_argument("--local_demo_sparse_attn", action="store_true")
    parser.add_argument("--block_cache_sparse_dlm", action="store_true")
    parser.add_argument("--demo_sparse_ratio", type=float, default=0.5)
    parser.add_argument("--sparse_dlm_top_k", type=int, default=64)
    parser.add_argument("--sparse_dlm_selection_interval", type=int, default=4)
    parser.add_argument("--demo_sparse_mode", choices=["kv", "q"], default="kv")
    parser.add_argument("--demo_sparse_dense_fallback_mask_count", type=int, default=4)
    parser.add_argument("--mask_id", type=int, default=156895)
    parser.add_argument(
        "--layer_candidate_ratio",
        type=float,
        nargs="+",
        default=None,
        help="Candidate ratios for layer-wise query_position recall.",
    )
    parser.add_argument(
        "--enable_layer_candidate_ratios",
        action="store_true",
        help="Enable layer-wise query_position recall statistics.",
    )
    parser.add_argument(
        "--debug_transfer_tokens",
        action="store_true",
        help="Print per-step generated/edited token ids and positions at the end.",
    )
    parser.add_argument(
        "--plot_attention_every_n",
        type=int,
        default=32,
        help="Save attention heatmaps every N recorded steps when plotting is enabled.",
    )
    parser.add_argument("--no_plot_attentions", action="store_true")
    return parser.parse_args()


def print_run_config(args):
    print("Run configuration:")
    for key, value in sorted(vars(args).items()):
        print(f"{key}={value}")
    print()


def plot_step_attentions(step_attentions, step_idx, save_dir):
    # step_attentions: tuple[num_layers], each tensor is [bs, num_heads, q_len, k_len]
    num_layers = len(step_attentions)
    num_heads = step_attentions[0].shape[1]
    q_len = step_attentions[0].shape[2]
    k_len = step_attentions[0].shape[3]
    plt.rcParams["image.interpolation"] = "none"
    tick_step = 8 if q_len < 64 else 16
    xticks = list(range(0, k_len, tick_step))
    yticks = list(range(0, q_len, tick_step))

    fig, axes = plt.subplots(
        num_layers,
        num_heads,
        figsize=(max(8, num_heads * 1.1), max(8, num_layers * 1.1)),
        squeeze=False,
    )
    fig.set_dpi(300)

    for layer_idx in range(num_layers):
        layer_attn = step_attentions[layer_idx][0]
        for head_idx in range(num_heads):
            ax = axes[layer_idx][head_idx]
            ax.imshow(
                layer_attn[head_idx].detach().float().cpu().numpy(),
                cmap="viridis",
                interpolation="none",
                aspect="equal",
                resample=False,
                origin="upper",
                # extent=(-0.5, k_len - 0.5, q_len - 0.5, -0.5),
            )
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.tick_params(axis="both", which="both", labelsize=5, length=2)
            if layer_idx == 0:
                ax.set_title(f"H{head_idx}", fontsize=6)
            if head_idx == 0:
                ax.set_ylabel(f"L{layer_idx}", fontsize=6)

    fig.suptitle(f"Attention Weights (Step {step_idx + 1})", fontsize=12)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"attn_step_{step_idx + 1}.png")
    fig.savefig(save_path, dpi=600)
    plt.close(fig)
    print(f"Saved attention heatmap: {save_path}")


def _sanitize_attentions(attentions):
    if attentions is None:
        return None
    valid = tuple(att.detach().cpu() for att in attentions if att is not None)
    return valid if valid else None


def select_heatmap_steps(step_attentions, step_post_steps):
    valid_indices = [i for i, attn in enumerate(step_attentions) if attn is not None]
    if not valid_indices:
        return []

    selected = []
    for pos, idx in enumerate(valid_indices):
        is_last_valid = pos == len(valid_indices) - 1
        if is_last_valid:
            selected.append(idx)
            continue

        next_idx = valid_indices[pos + 1]
        cur_post = step_post_steps[idx]
        next_post = step_post_steps[next_idx]

        # End of one block: post_steps has entered post phase (>0), and the
        # next recorded step resets to 0 (new block starts).
        if cur_post > 0 and next_post == 0:
            selected.append(idx)

    if not selected:
        selected = [valid_indices[-1]]
    return selected


def select_heatmap_steps_every_n(step_attentions, interval):
    valid_indices = [i for i, attn in enumerate(step_attentions) if attn is not None]
    if not valid_indices:
        return []

    if interval <= 0:
        return valid_indices

    selected = [idx for idx in valid_indices if (idx + 1) % interval == 0]
    if valid_indices[-1] not in selected:
        selected.append(valid_indices[-1])
    return selected


def _compute_transfer_index(
    model,
    active_logits,
    active_block_mask,
    old_block_tokens,
    prompt_mask_in_block,
    allowed_mask_positions=None,
    temperature=0.0,
    top_p=None,
    top_k=None,
    threshold=0.5,
    editing_threshold=0.0,
    num_to_transfer=1,
):
    x0, x0_p = model._sample_with_temperature_topk_topp(
        active_logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    mask_transfer_index = torch.zeros_like(x0, dtype=torch.bool)
    if allowed_mask_positions is not None:
        active_mask_candidates = active_block_mask & allowed_mask_positions
    else:
        active_mask_candidates = active_block_mask
    if active_mask_candidates.sum() > 0:
        mask_confidence = torch.where(active_mask_candidates, x0_p, -torch.inf)
        high_conf_mask = (mask_confidence[0] > threshold) & active_mask_candidates[0]
        num_high_confidence = high_conf_mask.sum().item()

        if num_high_confidence >= num_to_transfer:
            mask_transfer_index[0] = high_conf_mask
        else:
            num_available = active_mask_candidates.sum().item()
            if num_available > 0:
                _, idx = torch.topk(
                    mask_confidence[0],
                    k=min(num_to_transfer, num_available),
                )
                mask_transfer_index[0, idx] = True

    editing_transfer_index = torch.zeros_like(x0, dtype=torch.bool)
    non_mask_positions = ~active_block_mask
    non_prompt_positions = ~prompt_mask_in_block
    editable_positions = non_mask_positions & non_prompt_positions[None, :]
    editing_confidence = torch.where(editable_positions, x0_p, -torch.inf)
    high_conf_editing = (editing_confidence[0] > editing_threshold) & editable_positions[0]

    token_changed = x0[0] != old_block_tokens[0]
    editing_transfer_index[0] = high_conf_editing & token_changed
    final_transfer_index = mask_transfer_index | editing_transfer_index
    return final_transfer_index, x0, x0_p


def _compute_candidate_scores(
    model,
    active_logits,
    active_block_mask,
    old_block_tokens,
    prompt_mask_in_block,
    temperature=0.0,
    top_p=None,
    top_k=None,
    editing_threshold=0.0,
):
    x0, x0_p = model._sample_with_temperature_topk_topp(
        active_logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    mask_scores = torch.where(active_block_mask, x0_p, -torch.inf)

    non_mask_positions = ~active_block_mask
    non_prompt_positions = ~prompt_mask_in_block
    editable_positions = non_mask_positions & non_prompt_positions[None, :]
    token_changed = x0[0] != old_block_tokens[0]
    editable_positions = editable_positions & token_changed[None, :]
    edit_scores = torch.where(editable_positions, x0_p, -torch.inf)
    if editing_threshold is not None:
        edit_scores = torch.where(edit_scores > editing_threshold, edit_scores, -torch.inf)

    candidate_scores = torch.maximum(mask_scores, edit_scores)
    return candidate_scores


def _select_positions_by_ratio(candidate_scores, ratio):
    valid = torch.isfinite(candidate_scores)
    valid_count = int(valid.sum().item())
    if valid_count <= 0:
        return torch.zeros_like(candidate_scores, dtype=torch.bool)
    select_count = min(max(1, math.ceil(valid_count * float(ratio))), valid_count)
    scores = torch.where(valid, candidate_scores, -torch.inf)
    _, idx = torch.topk(scores[0], k=select_count)
    selected = torch.zeros_like(candidate_scores, dtype=torch.bool)
    selected[0, idx] = True
    return selected


def _layer_hidden_states_for_recall(outputs, num_layers):
    # outputs.hidden_states contains embedding, pre-layer states, and final norm.
    # For layers 0..N-2, hidden_states[i + 1] is the output of layer i. The
    # last entry is the normalized final representation used by lm_head.
    hidden_states = outputs.hidden_states
    layer_states = []
    for layer_idx in range(num_layers):
        if layer_idx < num_layers - 1:
            layer_states.append(hidden_states[layer_idx + 1])
        else:
            layer_states.append(hidden_states[-1])
    return layer_states


@torch.no_grad()
def generate_with_trace(
    model,
    inputs: torch.Tensor,
    tokenizer=None,
    temperature: float = 0.0,
    block_length: int = 32,
    steps: int = 32,
    gen_length: int = 512,
    top_p=None,
    top_k=None,
    eos_early_stop: bool = True,
    minimal_topk: int = 1,
    threshold: float = 0.5,
    editing_threshold: float = 0.0,
    max_post_steps: int = 16,
    eos_id: int = 156892,
    mask_id: int = 156895,
    num_to_transfer: int = 1,
    layer_candidate_ratios=None,
    debug_transfer_tokens: bool = False,
):
    # Mirror modeling_llada2_moe.py::generate and collect per-step traces.
    steps = min(steps, gen_length // minimal_topk)
    input_ids = inputs.to(model.device)

    prompt_length = input_ids.shape[1]
    if hasattr(model.config, "llada_demo_sparse_ratio"):
        model.config.llada_demo_sparse_prompt_length = int(prompt_length)

    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=model.device))
    block_diffusion_attention_mask = (
        block_mask.repeat_interleave(block_length, dim=0)
        .repeat_interleave(block_length, dim=1)
        .unsqueeze(0)
        .unsqueeze(0)
    ).to(torch.bfloat16)

    position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)
    x = torch.full((1, total_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids.clone()

    prefill_blocks = prompt_length // block_length

    step_attentions = []
    step_post_steps = []
    step_tokens = []
    transfer_index = []
    history = []
    step_transfer_events = []
    step_predicted_query_positions = []
    step_predicted_query_position_groups = []
    num_layers = len(model.model.layers)
    collect_layer_candidate_recall = layer_candidate_ratios is not None
    if collect_layer_candidate_recall and not layer_candidate_ratios:
        layer_candidate_ratios = [1.0]
    layer_candidate_ratios = (
        [float(ratio) for ratio in layer_candidate_ratios]
        if collect_layer_candidate_recall
        else []
    )
    ratio_keys = [f"{ratio:.6f}" for ratio in layer_candidate_ratios]
    ratio_layer_recall_hits = {
        ratio_key: torch.zeros(num_layers, dtype=torch.long) for ratio_key in ratio_keys
    }
    ratio_layer_recall_total = {
        ratio_key: torch.zeros(num_layers, dtype=torch.long) for ratio_key in ratio_keys
    }
    ratio_layer_step_recalls = {ratio_key: [] for ratio_key in ratio_keys}
    exact_layer_recall_hits = torch.zeros(num_layers, dtype=torch.long)
    exact_layer_recall_total = torch.zeros(num_layers, dtype=torch.long)
    exact_layer_step_recalls = []

    for num_block in range(prefill_blocks, num_blocks):
        current_window_end = (num_block + 1) * block_length
        cur_x = x[:, :current_window_end]
        cur_attn_mask = block_diffusion_attention_mask[
            :, :, :current_window_end, :current_window_end
        ]
        cur_position_ids = position_ids[:, :current_window_end]

        block_start_pos = num_block * block_length
        post_steps = 0

        while True:
            old_block_tokens = cur_x[:, -block_length:].clone()
            active_block_mask = cur_x[:, -block_length:] == mask_id
            if not torch.any(active_block_mask).item():
                post_steps += 1
            if post_steps > max_post_steps:
                break

            prompt_mask_in_block = torch.zeros(
                block_length, dtype=torch.bool, device=model.device
            )
            if block_start_pos < prompt_length:
                prompt_end_in_block = min(prompt_length - block_start_pos, block_length)
                prompt_mask_in_block[:prompt_end_in_block] = True

            outputs = model.forward(
                cur_x,
                attention_mask=cur_attn_mask,
                position_ids=cur_position_ids,
                output_attentions=True,
                output_hidden_states=collect_layer_candidate_recall,
            )
            predicted_query_positions = getattr(
                model.config,
                "llada_demo_sparse_last_query_positions",
                None,
            )
            if predicted_query_positions is not None:
                predicted_query_positions = [
                    int(pos) - block_start_pos
                    for pos in predicted_query_positions[0].tolist()
                    if block_start_pos <= int(pos) < current_window_end
                ]
            predicted_query_position_groups = None
            if predicted_query_positions is not None:
                predicted_query_position_groups = {
                    "decoded": [],
                    "mask": [],
                }
                for rel_pos in predicted_query_positions:
                    if bool(active_block_mask[0, rel_pos].item()):
                        predicted_query_position_groups["mask"].append(rel_pos)
                    else:
                        predicted_query_position_groups["decoded"].append(rel_pos)
            allowed_mask_positions = None
            if predicted_query_position_groups is not None:
                allowed_mask_positions = torch.zeros_like(active_block_mask)
                if predicted_query_position_groups["mask"]:
                    allowed_mask_positions[
                        0,
                        torch.tensor(
                            predicted_query_position_groups["mask"],
                            device=active_block_mask.device,
                            dtype=torch.long,
                        ),
                    ] = True
            logits = outputs.logits
            active_logits = logits[:, -block_length:, :]
            final_transfer_index, x0, _ = _compute_transfer_index(
                model,
                active_logits,
                active_block_mask,
                old_block_tokens,
                prompt_mask_in_block,
                allowed_mask_positions=allowed_mask_positions,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                threshold=threshold,
                editing_threshold=editing_threshold,
                num_to_transfer=num_to_transfer,
            )

            step_target_count = int(final_transfer_index.sum().item())
            if collect_layer_candidate_recall and step_target_count > 0:
                layer_states = _layer_hidden_states_for_recall(outputs, num_layers)
                for layer_idx, layer_hidden in enumerate(layer_states):
                    layer_logits = model.lm_head(layer_hidden).float()
                    layer_active_logits = layer_logits[:, -block_length:, :]
                    exact_layer_transfer_index, _, _ = _compute_transfer_index(
                        model,
                        layer_active_logits,
                        active_block_mask,
                        old_block_tokens,
                        prompt_mask_in_block,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        threshold=threshold,
                        editing_threshold=editing_threshold,
                        num_to_transfer=num_to_transfer,
                    )
                    exact_hit = int(
                        (exact_layer_transfer_index & final_transfer_index).sum().item()
                    )
                    exact_layer_recall_hits[layer_idx] += exact_hit
                    exact_layer_recall_total[layer_idx] += step_target_count

                    layer_candidate_scores = _compute_candidate_scores(
                        model,
                        layer_active_logits,
                        active_block_mask,
                        old_block_tokens,
                        prompt_mask_in_block,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        editing_threshold=editing_threshold,
                    )
                    for ratio_key, ratio in zip(ratio_keys, layer_candidate_ratios):
                        layer_candidate_index = _select_positions_by_ratio(
                            layer_candidate_scores,
                            ratio,
                        )
                        hit = int((layer_candidate_index & final_transfer_index).sum().item())
                        ratio_layer_recall_hits[ratio_key][layer_idx] += hit
                        ratio_layer_recall_total[ratio_key][layer_idx] += step_target_count

            if collect_layer_candidate_recall:
                if step_target_count > 0:
                    exact_layer_step_recalls.append(
                        (
                            exact_layer_recall_hits.float()
                            / exact_layer_recall_total.clamp_min(1).float()
                        ).tolist()
                    )
                else:
                    exact_layer_step_recalls.append([float("nan")] * num_layers)

                for ratio_key in ratio_keys:
                    if step_target_count > 0:
                        ratio_layer_step_recalls[ratio_key].append(
                            (
                                ratio_layer_recall_hits[ratio_key].float()
                                / ratio_layer_recall_total[ratio_key].clamp_min(1).float()
                            ).tolist()
                        )
                    else:
                        ratio_layer_step_recalls[ratio_key].append([float("nan")] * num_layers)

            step_events = []
            relative_positions = final_transfer_index[0].nonzero(as_tuple=True)[0]
            for rel_pos in relative_positions.tolist():
                abs_pos = current_window_end - block_length + rel_pos
                old_token_id = int(old_block_tokens[0, rel_pos].item())
                new_token_id = int(x0[0, rel_pos].item())
                kind = "generate" if bool(active_block_mask[0, rel_pos].item()) else "edit"
                step_events.append(
                    {
                        "step": len(step_transfer_events),
                        "block": int(num_block),
                        "post_steps": int(post_steps),
                        "kind": kind,
                        "position": int(abs_pos),
                        "block_position": int(rel_pos),
                        "old_token_id": old_token_id,
                        "new_token_id": new_token_id,
                    }
            )
            step_transfer_events.append(step_events)
            step_predicted_query_positions.append(predicted_query_positions)
            step_predicted_query_position_groups.append(predicted_query_position_groups)

            if final_transfer_index.any():
                cur_x[:, -block_length:][final_transfer_index] = x0[final_transfer_index]

            if debug_transfer_tokens:
                if predicted_query_position_groups is None:
                    query_debug = (
                        "decoded_query_positions=None\n"
                        "mask_query_positions=None"
                    )
                else:
                    query_debug = (
                        "decoded_query_positions="
                        f"{predicted_query_position_groups['decoded']}\n"
                        "mask_query_positions="
                        f"{predicted_query_position_groups['mask']}"
                    )
                print(
                    f"-----------------step={len(step_transfer_events) - 1} block={num_block} "
                    f"post_steps={post_steps} "
                    f"events={len(step_events)}-----------------\n"
                    f"{query_debug} "
                )
                if tokenizer is not None:
                    for event in step_events:
                        new_text = tokenizer.decode(
                            [event["new_token_id"]],
                            skip_special_tokens=False,
                        )
                        if event["kind"] == "edit":
                            old_text = tokenizer.decode(
                                [event["old_token_id"]],
                                skip_special_tokens=False,
                            )
                            print(
                                "  "
                                f"{event['kind']} pos={event['position']} "
                                f"block_pos={event['block_position']} "
                                f"old_text={old_text!r} -> new_text={new_text!r}"
                            )
                        else:
                            print(
                                "  "
                                f"{event['kind']} pos={event['position']} "
                                f"block_pos={event['block_position']} "
                                f"text={new_text!r}"
                            )
                print("----------------------------------------------------------------------")

            # Save traces on full-length canvas for easier step inspection.
            full_step_tokens = x.clone()
            full_step_tokens[:, :current_window_end] = cur_x

            full_transfer = torch.zeros_like(x, dtype=torch.bool)
            full_transfer[:, current_window_end - block_length : current_window_end][
                final_transfer_index
            ] = True

            sanitized_attn = _sanitize_attentions(outputs.attentions)
            step_attentions.append(sanitized_attn)
            step_post_steps.append(post_steps)
            step_tokens.append(full_step_tokens.detach().cpu())
            transfer_index.append(full_transfer.detach().cpu())
            history.append(full_step_tokens.detach().cpu())

            if active_block_mask.sum() == 0 and not final_transfer_index.any():
                break

        x[:, :current_window_end] = cur_x
        if eos_early_stop:
            generated_part = x[0, prompt_length:current_window_end]
            if (generated_part == mask_id).sum() == 0:
                eos_positions = (generated_part == eos_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    break

    generated_answer = x[:, : prompt_length + gen_length]
    eos_positions = (generated_answer[0][prompt_length:] == eos_id).nonzero(as_tuple=True)[0]
    first_eos_position = eos_positions[0].item() if len(eos_positions) > 0 else gen_length
    sequences = generated_answer[:, prompt_length : prompt_length + first_eos_position + 1]

    return SimpleNamespace(
        sequences=sequences.detach().cpu(),
        step_attentions=step_attentions,
        step_post_steps=step_post_steps,
        step_tokens=step_tokens,
        transfer_index=transfer_index,
        history=history,
        step_transfer_events=step_transfer_events,
        step_predicted_query_positions=step_predicted_query_positions,
        step_predicted_query_position_groups=step_predicted_query_position_groups,
        exact_layer_recall_hits=exact_layer_recall_hits,
        exact_layer_recall_total=exact_layer_recall_total,
        exact_layer_step_recalls=exact_layer_step_recalls,
        ratio_keys=ratio_keys,
        layer_candidate_ratios=layer_candidate_ratios,
        ratio_layer_recall_hits=ratio_layer_recall_hits,
        ratio_layer_recall_total=ratio_layer_recall_total,
        ratio_layer_step_recalls=ratio_layer_step_recalls,
    )


def print_layer_recall_summary(output):
    if not output.layer_candidate_ratios:
        return

    print("\nLayer-wise query_position recall:")
    exact_hits = output.exact_layer_recall_hits.tolist()
    exact_totals = output.exact_layer_recall_total.tolist()
    exact_recalls = [
        hit / total if total else float("nan")
        for hit, total in zip(exact_hits, exact_totals)
    ]
    exact_valid = [recall for recall in exact_recalls if not math.isnan(recall)]
    exact_mean = sum(exact_valid) / len(exact_valid) if exact_valid else float("nan")
    print("\nexact_decode_recall:")
    print(f"mean_exact_decode_recall={exact_mean:.6f}")
    for layer_idx, (hit, total, recall) in enumerate(
        zip(exact_hits, exact_totals, exact_recalls)
    ):
        print(
            f"layer={layer_idx:02d} exact_decode_recall={recall:.6f} "
            f"hit={hit} total={total}"
        )

    for ratio_key in output.ratio_keys:
        ratio = float(ratio_key)
        hits = output.ratio_layer_recall_hits[ratio_key].tolist()
        totals = output.ratio_layer_recall_total[ratio_key].tolist()
        recall_values = [hit / total for hit, total in zip(hits, totals) if total]
        mean_recall = sum(recall_values) / len(recall_values) if recall_values else float("nan")
        print(f"\nratio={ratio:.3f} mean_recall={mean_recall:.6f}")
        for layer_idx, (hit, total) in enumerate(zip(hits, totals)):
            recall = hit / total if total else float("nan")
            print(f"layer={layer_idx:02d} recall={recall:.6f} hit={hit} total={total}")


def print_transfer_token_debug(output, tokenizer):
    print("\nPer-step transfer token debug:")
    if not output.step_transfer_events and not output.step_predicted_query_positions:
        print("No transfer events recorded.")
        return

    max_steps = max(
        len(output.step_transfer_events),
        len(output.step_predicted_query_positions),
    )
    for step_idx in range(max_steps):
        step_events = (
            output.step_transfer_events[step_idx]
            if step_idx < len(output.step_transfer_events)
            else []
        )
        query_positions = (
            output.step_predicted_query_positions[step_idx]
            if step_idx < len(output.step_predicted_query_positions)
            else None
        )
        query_position_groups = (
            output.step_predicted_query_position_groups[step_idx]
            if step_idx < len(output.step_predicted_query_position_groups)
            else None
        )
        if step_events:
            first = step_events[0]
            block = first["block"]
            post_steps = first["post_steps"]
        else:
            block = "unknown"
            post_steps = "unknown"
        if query_position_groups is None:
            query_debug = "decoded_query_positions=None mask_query_positions=None"
        else:
            query_debug = (
                "decoded_query_positions="
                f"{query_position_groups['decoded']} "
                "mask_query_positions="
                f"{query_position_groups['mask']}"
            )
        print(
            f"step={step_idx} block={block} post_steps={post_steps} "
            f"{query_debug} events={len(step_events)}"
        )
        for event in step_events:
            new_text = tokenizer.decode(
                [event["new_token_id"]],
                skip_special_tokens=False,
            )
            if event["kind"] == "edit":
                old_text = tokenizer.decode(
                    [event["old_token_id"]],
                    skip_special_tokens=False,
                )
                print(
                    "  "
                    f"{event['kind']} pos={event['position']} "
                    f"block_pos={event['block_position']} "
                    f"old_text={old_text!r} -> new_text={new_text!r}"
                )
            else:
                print(
                    "  "
                    f"{event['kind']} pos={event['position']} "
                    f"block_pos={event['block_position']} "
                    f"text={new_text!r}"
                )


args = parse_args()
if args.use_model_generate and args.local_demo_sparse_attn:
    raise ValueError("--use_model_generate and --local_demo_sparse_attn are mutually exclusive")
if args.use_model_generate and args.block_cache_sparse_dlm:
    raise ValueError("--use_model_generate and --block_cache_sparse_dlm are mutually exclusive")
if args.block_cache_sparse_dlm and args.local_demo_sparse_attn:
    raise ValueError("--block_cache_sparse_dlm and --local_demo_sparse_attn are mutually exclusive")
print_run_config(args)

model_path = args.model_path
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto",
    attn_implementation="sdpa" if args.block_cache_sparse_dlm else "eager",
    dtype=torch.bfloat16,
)
# model = model.to(torch.bfloat16)
model.eval()
torch.cuda.empty_cache()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

if args.local_demo_sparse_attn:
    patch_llada_demo_sparse(
        model,
        ratio=args.demo_sparse_ratio,
        block_length=args.block_length,
        mask_id=args.mask_id,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        sparse_mode=args.demo_sparse_mode,
        threshold=args.threshold,
        editing_threshold=args.editing_threshold,
        num_to_transfer=args.num_to_transfer,
        dense_fallback_mask_count=args.demo_sparse_dense_fallback_mask_count,
    )
    print(
        "Applied LLaDA local demo sparse attention: "
        f"mode={args.demo_sparse_mode}, "
        f"ratio={args.demo_sparse_ratio}, block_length={args.block_length}"
    )

if args.block_cache_sparse_dlm:
    patch_llada_block_cache_sparse(
        model,
        ratio=args.demo_sparse_ratio,
        top_k=args.sparse_dlm_top_k,
        selection_interval=args.sparse_dlm_selection_interval,
        dense_fallback_mask_count=args.demo_sparse_dense_fallback_mask_count,
    )
    print(
        "Applied LLaDA block-cache sparse DLM: "
        f"ratio={args.demo_sparse_ratio}, "
        f"top_k={args.sparse_dlm_top_k}, "
        f"selection_interval={args.sparse_dlm_selection_interval}"
    )

input_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": args.prompt}],
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
)

if args.use_model_generate or args.block_cache_sparse_dlm:
    sequences = model.generate(
        inputs=input_ids,
        eos_early_stop=True,
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps=args.steps,
        threshold=args.threshold,
        editing_threshold=args.editing_threshold,
        num_to_transfer=args.num_to_transfer,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        mask_id=args.mask_id,
    )
    generated_answer = tokenizer.decode(sequences[0], skip_special_tokens=True)
else:
    output = generate_with_trace(
        model,
        inputs=input_ids,
        tokenizer=tokenizer,
        eos_early_stop=True,
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps=args.steps,
        threshold=args.threshold,
        editing_threshold=args.editing_threshold,
        num_to_transfer=args.num_to_transfer,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        mask_id=args.mask_id,
        layer_candidate_ratios=(
            args.layer_candidate_ratio if args.enable_layer_candidate_ratios else None
        ),
        debug_transfer_tokens=args.debug_transfer_tokens,
    )

    print_layer_recall_summary(output)

    if output.step_attentions and not args.no_plot_attentions:
        selected = select_heatmap_steps_every_n(
            output.step_attentions,
            args.plot_attention_every_n,
        )
        for step_idx in selected:
            plot_step_attentions(output.step_attentions[step_idx], step_idx, "attn_plots")
    else:
        print("No step_attentions found in output.")

    generated_answer = tokenizer.decode(output.sequences[0], skip_special_tokens=True)

print(generated_answer)

# prompt_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
# prompt_tokens_bracketed = " ".join(f"[{tok}]" for tok in prompt_tokens)
# print(f"prompt_tokens: {prompt_tokens_bracketed}")

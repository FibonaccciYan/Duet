import argparse
import torch
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def top_k_logits(logits, k):
    if k is None or k <= 0:
        return logits
    else:
        values, _ = torch.topk(logits, min(int(k), logits.shape[-1]))
        min_values = values[..., -1, None]
        return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)


def top_p_logits(logits, p):
    if p is None or p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask_indices = torch.scatter(torch.full_like(logits, False, dtype=torch.bool),
                                 -1, sorted_indices, sorted_mask)
    logits = logits.masked_fill(mask_indices, float('-inf'))
    return logits


def sample_with_temperature_topk_topp(logits, temperature=1.0, top_k=0, top_p=1.0):
    orig_shape = logits.shape[:-1]    # [batch, block]
    vocab_size = logits.shape[-1]

    logits = logits.reshape(-1, vocab_size)  # [batch*block, vocab]

    if temperature is None or temperature <= 0:
        token = logits.argmax(dim=-1)
        probs = F.softmax(logits, dim=-1)
        token_prob = torch.gather(probs, -1, token.unsqueeze(-1)).squeeze(-1)
        return token.view(*orig_shape), token_prob.view(*orig_shape)

    if temperature != 1.0:
        logits = logits / temperature
    if top_k is not None and top_k > 0:
        logits = top_k_logits(logits, top_k)
    if top_p is not None and top_p < 1.0:
        logits = top_p_logits(logits, top_p)
    probs = F.softmax(logits, dim=-1)  # shape: [batch*block, vocab]
    assert probs.dim() == 2
    token = torch.multinomial(probs, num_samples=1)  # [batch*block, 1]
    token_prob = torch.gather(probs, -1, token)     # [batch*block, 1]

    return token.view(*orig_shape), token_prob.view(*orig_shape)


def entropy_from_logits(logits, temperature=1.0, top_k=0, top_p=1.0):
    """Return categorical entropy for every token position."""
    logits = logits.float()
    if temperature is not None and temperature > 0:
        logits = logits / temperature
        logits = top_k_logits(logits, top_k)
        logits = top_p_logits(logits, top_p)
    log_probs = F.log_softmax(logits, dim=-1)
    terms = torch.where(
        torch.isfinite(log_probs), log_probs.exp() * log_probs, 0.0
    )
    return -terms.sum(dim=-1)


def get_num_transfer_tokens(block_length, steps):
    if block_length <= 0 or steps <= 0 or steps > block_length:
        raise ValueError("SDAR requires 1 <= denoising_steps <= block_length")
    base = block_length // steps
    remainder = block_length % steps
    num_transfer_tokens = torch.zeros(steps, dtype=torch.int64) + base
    num_transfer_tokens[:remainder] += 1
    return num_transfer_tokens


def select_transfer(
    mask,
    confidence,
    minimum,
    strategy,
    threshold,
    entropy=None,
    entropy_budget=None,
):
    """Apply SDAR's transfer rule to positions with available predictions."""
    if strategy == "entropy_bounded" and (
        entropy is None or entropy_budget is None
    ):
        raise ValueError(
            "entropy and entropy_budget are required for entropy_bounded"
        )
    transfer = torch.zeros_like(mask)
    for batch_idx in range(mask.shape[0]):
        available = torch.isfinite(
            entropy[batch_idx] if strategy == "entropy_bounded" else confidence[batch_idx]
        )
        positions = torch.where(mask[batch_idx] & available)[0]
        count = min(int(minimum), positions.numel())
        if not count:
            continue
        if strategy == "entropy_bounded":
            values, order = torch.sort(entropy[batch_idx, positions])
            budget_count = int(
                torch.searchsorted(
                    torch.cumsum(values, dim=0),
                    values.new_tensor(float(entropy_budget)),
                    right=False,
                ).item()
            )
            count = min(max(count, budget_count, 1), positions.numel())
            selected = positions[order[:count]]
        elif strategy == "sequential":
            selected = positions[:count]
        else:
            scores = confidence[batch_idx, positions]
            if strategy == "low_confidence_dynamic":
                high = positions[scores > threshold]
                selected = high if high.numel() >= count else positions[
                    torch.topk(scores, count).indices
                ]
            elif strategy == "low_confidence_static":
                selected = positions[torch.topk(scores, count).indices]
            else:
                raise ValueError(f"Unknown remasking strategy: {strategy}")
        transfer[batch_idx, selected] = True
    return transfer


@torch.no_grad()
def block_diffusion_generate(
        model,
        prompt,
        mask_id,
        gen_length=128,
        block_length=32,
        denoising_steps=32,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        remasking_strategy='sequential',
        confidence_threshold=0.85,
        eb_threshold=None,
        stopping_criteria_idx=None,
        denoise_fn=None,
    ):

    model.eval()
    if remasking_strategy == "entropy_bounded" and eb_threshold is None:
        raise ValueError("eb_threshold is required for entropy_bounded transfer")
    input_ids = prompt['input_ids']
    prompt_length = input_ids.shape[1]
    past_key_values = DynamicCache()

    num_blocks = (prompt_length + gen_length +
                  block_length - 1) // block_length
    total_length = num_blocks * block_length

    block_mask = torch.tril(torch.ones(
        num_blocks, num_blocks, device=model.device))
    block_diffusion_attention_mask = block_mask.repeat_interleave(block_length, dim=0)\
                                               .repeat_interleave(block_length, dim=1).unsqueeze(0)
    position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)

    x = torch.full((1, total_length), mask_id,
                   dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length

    # Prefill stage.  The block-causal mask lets aligned prompt chunks be
    # written to the same KV cache independently.  Keeping the query side
    # bounded avoids SDPA materializing an O(prompt_length**2) score tensor.
    if prefill_length > 0:
        prefill_chunk_length = (
            prefill_length if prefill_length <= 256 * block_length
            else 128 * block_length
        )
        for chunk_start in range(0, prefill_length, prefill_chunk_length):
            chunk_end = min(chunk_start + prefill_chunk_length, prefill_length)
            cur_x = x[:, chunk_start:chunk_end]
            cur_attn_mask = block_diffusion_attention_mask[
                :, chunk_start:chunk_end, :chunk_end
            ]
            cur_position_ids = position_ids[:, chunk_start:chunk_end]
            model(cur_x,
                  attention_mask=cur_attn_mask,
                  position_ids=cur_position_ids,
                  past_key_values=past_key_values,
                  use_cache=True,
                  store_kv=True)

    num_transfer_tokens = get_num_transfer_tokens(
        block_length, denoising_steps)

    # Decode stage
    for num_block in range(prefill_blocks, num_blocks):
        cur_x = x[:, num_block*block_length:(num_block+1)*block_length].clone()
        cur_attn_mask = block_diffusion_attention_mask[
            :, num_block*block_length:(num_block+1)*block_length, :(num_block+1)*block_length
        ]
        cur_position_ids = position_ids[:, num_block *
                                        block_length:(num_block+1)*block_length]
        for step in range(denoising_steps + 1):
            mask_index = (cur_x == mask_id)
            if mask_index.sum() == 0:
                # Store kv cache
                model(cur_x,
                      attention_mask=cur_attn_mask,
                      position_ids=cur_position_ids,
                      past_key_values=past_key_values,
                      use_cache=True,
                      store_kv=True)
                break

            if step == denoising_steps:
                raise RuntimeError(
                    f"SDAR block {num_block} still contains masks after "
                    f"{denoising_steps} steps"
                )

            # Denosing
            logit_positions = None
            if denoise_fn is None:
                logits = model(cur_x,
                               attention_mask=cur_attn_mask,
                               position_ids=cur_position_ids,
                               past_key_values=past_key_values,
                               use_cache=True,
                               store_kv=False).logits
            else:
                logits, logit_positions = denoise_fn(
                    model=model,
                    block_tokens=cur_x,
                    attention_mask=cur_attn_mask,
                    position_ids=cur_position_ids,
                    past_key_values=past_key_values,
                    block_start=num_block * block_length,
                    block_end=(num_block + 1) * block_length,
                    step=step,
                    minimum=int(num_transfer_tokens[step]),
                )

            # Sampling
            x0, x0_p = sample_with_temperature_topk_topp(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            x0_entropy = (
                entropy_from_logits(logits, temperature, top_k, top_p)
                if remasking_strategy == "entropy_bounded"
                else None
            )

            if logit_positions is not None:
                full_x0 = cur_x.clone()
                full_x0.index_copy_(1, logit_positions, x0)
                full_confidence = torch.full_like(
                    cur_x, -torch.inf, dtype=x0_p.dtype
                )
                full_confidence.index_copy_(1, logit_positions, x0_p)
                x0, x0_p = full_x0, full_confidence
                if x0_entropy is not None:
                    full_entropy = torch.full_like(
                        cur_x, torch.inf, dtype=x0_entropy.dtype
                    )
                    full_entropy.index_copy_(1, logit_positions, x0_entropy)
                    x0_entropy = full_entropy

            # Sampling strategy
            confidence = torch.where(mask_index, x0_p, -torch.inf)
            transfer_index = select_transfer(
                mask_index,
                confidence,
                num_transfer_tokens[step],
                remasking_strategy,
                confidence_threshold,
                entropy=x0_entropy,
                entropy_budget=eb_threshold,
            )

            cur_x[transfer_index] = x0[transfer_index]

        x[:, num_block*block_length:(num_block+1)*block_length] = cur_x
        if stopping_criteria_idx is not None and any(
            torch.any(x[:, prompt_length:] == stop_idx)
            for stop_idx in stopping_criteria_idx
        ):
            break

    return x


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the pretrained model directory")
    parser.add_argument("--trust_remote_code", action='store_true')
    parser.add_argument("--mask_id", type=int, default=None,
                        help="Mask token id for Diffusion")
    parser.add_argument("--prompt_length", type=int, default=4096,
                        help="Maximum prompt length in tokens")
    parser.add_argument("--gen_length", type=int, default=20480,
                        help="Maximum generation length in tokens")
    parser.add_argument("--block_length", type=int, default=32,
                        help="Length of token block to replace each denoising step")
    parser.add_argument("--denoising_steps", type=int, default=32,
                        help="Number of denoising steps (iterations)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Top-K sampling (0 to disable)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-P sampling probability threshold")
    parser.add_argument("--remasking_strategy", type=str, default="sequential",
                        choices=["low_confidence_dynamic",
                                 "low_confidence_static",
                                 "sequential",
                                 "entropy_bounded"],
                        help="Strategy for remasking tokens")
    parser.add_argument("--confidence_threshold", type=float, default=0.85,
                        help="Confidence threshold for low-confidence remasking")
    parser.add_argument("--eb_threshold", type=float, default=0.35,
                        help="entropy threshold for entropy bounded sampling")
    parser.add_argument("--stopping_criteria_idx", type=int, nargs="+", default=None,
                        help="List of token IDs that stop generation (e.g. eos_token_id)")

    parser.add_argument("--device", type=str, default="cuda",)
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16"],)
    args = parser.parse_args()
    if args.remasking_strategy == "low_confidence_dynamic" and args.confidence_threshold is None:
        parser.error(
            "--confidence_threshold is required when --remasking_strategy=low_confidence_dynamic"
        )
    if args.remasking_strategy == "entropy_bounded" and args.eb_threshold is None:
        parser.error(
            "--eb_threshold is required when --remasking_strategy=entropy_bounded"
        )
    return args


if __name__ == "__main__":
    args = parse_args()

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=args.dtype,
        device_map=args.device
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=args.trust_remote_code,
    )

    if args.mask_id is None:
        args.mask_id = tokenizer(tokenizer.mask_token)['input_ids'][0]
    if args.stopping_criteria_idx is None:
        gen_cfg = GenerationConfig.from_pretrained(args.model_dir,)
        args.stopping_criteria_idx = gen_cfg.eos_token_id
    if isinstance(args.stopping_criteria_idx, int):
        args.stopping_criteria_idx = [args.stopping_criteria_idx,]
    args.stop_words = tokenizer.convert_ids_to_tokens(
        args.stopping_criteria_idx)
    print(f"Your Arguments: {args}")

    origin_prompt = [
        # dict(role="user", content="Given the function $f(x) = \\frac{4x^2 - 4x + 4}{x^2 + 2x + 4}$, where $x \\in \\mathbb{R}$, determine its minimum value.\nPlease reason step by step, and put your final answer within \\boxed{}.\n"),
        dict(role="user", content="If the domain of the function $\\log x^2$ is $x < a$ or $x > b$, for some $a$ and $b$, find $a + b$.\nPlease reason step by step, and put your final answer within \\boxed{}.\n")
    ]

    messages = tokenizer.apply_chat_template(
        origin_prompt, add_generation_prompt=True, tokenize=False)
    tokenize_kwargs = dict(
        return_tensors='pt',
        padding=True,
        truncation=True,
        add_special_tokens=False,
        max_length=args.prompt_length
    )

    tokens = tokenizer.batch_encode_plus([messages], **tokenize_kwargs)
    tokens = {k: v.to(model.device) for k, v in tokens.items()}

    output_ids = block_diffusion_generate(
        model,
        prompt=tokens,
        mask_id=args.mask_id,
        gen_length=args.gen_length,
        block_length=args.block_length,
        denoising_steps=args.denoising_steps,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        remasking_strategy=args.remasking_strategy,
        confidence_threshold=args.confidence_threshold,
        eb_threshold=args.eb_threshold,
        stopping_criteria_idx=args.stopping_criteria_idx
    )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    cleaned_text = output_text.replace('<|MASK|>', '')
    print(cleaned_text)

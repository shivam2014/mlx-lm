"""
MTP Acceptance Rate Test — Qwen3.6-27B
Loads quantized model + injects MTP weights from raw HF checkpoint.
"""
import argparse
import time
import json
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import ArraysCache, make_prompt_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Quantized MLX model path")
    parser.add_argument("--mtp-weights", type=str, required=True, help="MTP weights safetensors file")
    parser.add_argument("--prompt", type=str, default="Write a haiku about speculative decoding:")
    parser.add_argument("--num-tokens", type=int, default=128)
    args = parser.parse_args()

    # Load quantized model (MTP auto-disabled since weights aren't in checkpoint)
    print(f"Loading quantized model...")
    t0 = time.time()
    model, tokenizer = load(args.model)
    print(f"Loaded in {time.time() - t0:.1f}s")

    # Check MTP module exists
    if not hasattr(model, "mtp"):
        print("❌ Model doesn't have MTP module — config may not have mtp_num_hidden_layers")
        return

    # Load MTP weights and inject into model
    print(f"Loading MTP weights from {args.mtp_weights}...")
    mtp_weights = mx.load(args.mtp_weights)
    mtp_weights_list = list(mtp_weights.items())
    
    # Fixup RMSNorm weights: HF stores as offsets (actual = 1 + stored).
    # We detect this by checking mean < 0.5 and add +1.0 if needed.
    _MTP_NORM_SUFFIXES = (
        '.input_layernorm.weight', '.post_attention_layernorm.weight',
        '.q_norm.weight', '.k_norm.weight',
        '.pre_fc_norm_hidden.weight', '.pre_fc_norm_embedding.weight',
        '.norm.weight',
    )
    for key, val in mtp_weights.items():
        if val.ndim == 1 and any(key.endswith(s) for s in _MTP_NORM_SUFFIXES):
            if val.mean().item() < 0.5:
                mtp_weights[key] = val + 1.0
    mtp_weights_list = list(mtp_weights.items())
    
    # Load weights into the MTP module (non-strict to allow partial matches)
    model.mtp.load_weights(mtp_weights_list, strict=False)
    # load_weights on sub-modules may not update RMSNorm params.
    # Directly assign fixupped norm weights.
    for k, v in mtp_weights.items():
        if v.ndim == 1:
            parts = k.split(".")
            target = model.mtp
            for p in parts[:-1]:
                if p.isdigit():
                    target = target[int(p)]
                else:
                    target = getattr(target, p)
            setattr(target, parts[-1], v)
    print(f"✅ Injected {len(mtp_weights)} MTP weight tensors")

    # Tokenize
    prompt_tokens = mx.array([tokenizer.encode(args.prompt)])
    prompt_len = prompt_tokens.shape[1]
    print(f"Prompt: {prompt_len} tokens, testing {args.num_tokens} tokens\n")

    # Prefill with backbone cache
    backbone_cache = make_prompt_cache(model)
    logits, h = model(prompt_tokens, cache=backbone_cache, return_hidden=True)
    mx.eval(logits, h)
    
    # First token
    backbone_token = mx.argmax(logits[:, -1, :], axis=-1)
    generated = [int(backbone_token.item())]
    last_hidden = h[:, -1:, :]
    mtp_cache = model.make_mtp_cache()
    # Align MTP cache RoPE offset with backbone position (Tom Turney bug #3)
    # The MTP decoder attention uses cache.offset for RoPE. It must match
    # the backbone position (current generated token count + prompt_len).
    _first_offset = next(
        c.offset for c in backbone_cache
        if hasattr(c, "offset") and not isinstance(c, ArraysCache)
    )
    for c in mtp_cache:
        c.offset = int(_first_offset)

    mtp_correct = 0
    mtp_total = 0

    for step in range(args.num_tokens):
        # MTP prediction (draft)
        last_token_arr = mx.array([[generated[-1]]], dtype=mx.uint32)
        mtp_logits = model.mtp_forward(last_hidden, last_token_arr, cache=mtp_cache)
        mx.eval(mtp_logits)
        mtp_next = mx.argmax(mtp_logits[:, -1, :], axis=-1)

        # Backbone forward (verify)
        logits, h = model(last_token_arr, cache=backbone_cache, return_hidden=True)
        mx.eval(logits, h)
        backbone_next = mx.argmax(logits[:, -1, :], axis=-1)

        mtp_total += 1
        if int(mtp_next.item()) == int(backbone_next.item()):
            mtp_correct += 1

        generated.append(int(backbone_next.item()))
        last_hidden = h[:, -1:, :]

        if (step + 1) % 32 == 0 or step == 0:
            acc = (mtp_correct / mtp_total) * 100
            print(f"  Step {step+1:>4}/{args.num_tokens}  "
                  f"acceptance: {acc:>5.1f}%  ({mtp_correct}/{mtp_total})")

    final_acc = (mtp_correct / mtp_total) * 100
    print(f"\n{'='*50}")
    print(f"MTP ACCEPTANCE RATE: {final_acc:.1f}% ({mtp_correct}/{mtp_total})")
    print(f"Generated: {tokenizer.decode(generated)[:200]}...")

    if final_acc > 70:
        print(f"\n✅ ≥70% — MTP speculative decoding would give ~2-4x speedup!")
    elif final_acc > 50:
        print(f"\n👍 50-70% — would give ~1.5-2x speedup.")
    elif final_acc > 30:
        print(f"\n⚠️ 30-50% — marginal gains.")
    else:
        print(f"\n❌ Below 30% — MTP heads may need investigation.")


if __name__ == "__main__":
    main()

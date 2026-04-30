"""
MTP Acceptance Rate Test

Measures how often the built-in MTP head predicts the same next token 
as the backbone on Qwen 3.6 models.

Usage:
    python test_mtp.py --model Qwen/Qwen3.6-35B-A3B
    python test_mtp.py --model mlx-community/Qwen3.6-27B-4bit
"""
import argparse
import time
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models.cache import make_prompt_cache


def main():
    parser = argparse.ArgumentParser(description="Test MTP acceptance rate")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--prompt", type=str,
                        default="Write a haiku about speculative decoding:")
    parser.add_argument("--num-tokens", type=int, default=128,
                        help="Number of tokens to test")
    parser.add_argument("--mtp-layers", type=int, default=1,
                        help="Number of MTP layers (from model config)")
    args = parser.parse_args()

    # Load model
    print(f"Loading {args.model}...")
    t0 = time.time()
    model, tokenizer = load(
        args.model,
        model_config={"mtp_num_hidden_layers": args.mtp_layers},
    )
    print(f"Loaded in {time.time() - t0:.1f}s")

    # Check if MTP module was created
    if hasattr(model, "mtp"):
        print(f"✅ MTP module active ({args.mtp_layers} layer(s))")
    else:
        print("⚠️ MTP module not created — model may not have MTP heads.")
        return

    # Tokenize
    prompt_tokens = mx.array([tokenizer.encode(args.prompt)])
    prompt_len = prompt_tokens.shape[1]
    print(f"Prompt: {prompt_len} tokens, testing {args.num_tokens} tokens\n")

    # Prefill — create persistent caches for backbone
    backbone_cache = make_prompt_cache(model)
    
    # Full prefill
    logits, h = model(prompt_tokens, cache=backbone_cache, return_hidden=True)
    mx.eval(logits, h)
    
    # First token
    backbone_token = mx.argmax(logits[:, -1, :], axis=-1)
    generated = [int(backbone_token.item())]
    last_hidden = h[:, -1:, :]
    
    # Initialize MTP cache
    mtp_cache = model.make_mtp_cache()

    mtp_correct = 0
    mtp_total = 0

    for step in range(args.num_tokens):
        # --- MTP prediction (draft) ---
        last_token_arr = mx.array([[generated[-1]]], dtype=mx.uint32)
        mtp_logits = model.mtp_forward(last_hidden, last_token_arr, cache=mtp_cache)
        mx.eval(mtp_logits)
        mtp_next = mx.argmax(mtp_logits[:, -1, :], axis=-1)

        # --- Backbone forward (verify + get next hidden) ---
        logits, h = model(last_token_arr, cache=backbone_cache, return_hidden=True)
        mx.eval(logits, h)
        backbone_next = mx.argmax(logits[:, -1, :], axis=-1)

        # --- Compare ---
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
        print(f"\n⚠️ 30-50% — marginal gains, likely won't beat baseline.")
    else:
        print(f"\n❌ Below 30% — MTP heads may need investigation.")


if __name__ == "__main__":
    main()

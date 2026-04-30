"""
MTP Acceptance Rate Test — Qwen 3.6 models with built-in MTP heads.

Measures how often the MTP head predicts the same next token as the backbone,
without needing a separate draft model.

Usage:
    python test_mtp.py --model Qwen3.6-27B-UD-MLX-4bit
    python test_mtp.py --model Qwen3.6-35B-A3B-ConfigI-MLX --mtp-shards 25,26
"""
import argparse
import time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


def _inject_mtp_weights(model, weights):
    """Set MTP weights via direct attribute assignment (load_weights key format doesn't match)."""
    key_map = {
        "fc.weight": ("fc", "weight"),
        "norm.weight": ("norm", "weight"),
        "pre_fc_norm_hidden.weight": ("pre_fc_norm_hidden", "weight"),
        "pre_fc_norm_embedding.weight": ("pre_fc_norm_embedding", "weight"),
        "layers.0.input_layernorm.weight": ("layers.0.input_layernorm", "weight"),
        "layers.0.post_attention_layernorm.weight": ("layers.0.post_attention_layernorm", "weight"),
        "layers.0.self_attn.q_proj.weight": ("layers.0.self_attn.q_proj", "weight"),
        "layers.0.self_attn.k_proj.weight": ("layers.0.self_attn.k_proj", "weight"),
        "layers.0.self_attn.v_proj.weight": ("layers.0.self_attn.v_proj", "weight"),
        "layers.0.self_attn.o_proj.weight": ("layers.0.self_attn.o_proj", "weight"),
        "layers.0.self_attn.q_norm.weight": ("layers.0.self_attn.q_norm", "weight"),
        "layers.0.self_attn.k_norm.weight": ("layers.0.self_attn.k_norm", "weight"),
        "layers.0.mlp.gate_proj.weight": ("layers.0.mlp.gate_proj", "weight"),
        "layers.0.mlp.down_proj.weight": ("layers.0.mlp.down_proj", "weight"),
        "layers.0.mlp.up_proj.weight": ("layers.0.mlp.up_proj", "weight"),
    }
    for hf_key, (param_path, attr) in key_map.items():
        if hf_key not in weights:
            continue
        v = weights[hf_key]
        parts = param_path.split(".")
        obj = model.mtp
        for p in parts:
            obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
        setattr(obj, attr, v)
    return len([k for k in key_map if k in weights])


def main():
    parser = argparse.ArgumentParser(description="Test MTP acceptance rate")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--mtp-weights", type=str, default=None,
                        help="Path to pre-extracted MTP weights safetensors")
    parser.add_argument("--raw-hf-dir", type=str, default=None,
                        help="Path to raw HuggingFace checkpoint for MTP weight extraction")
    parser.add_argument("--prompt", type=str, default="Write a haiku about speculative decoding:")
    parser.add_argument("--num-tokens", type=int, default=128)
    args = parser.parse_args()

    print(f"Loading model...")
    t0 = time.time()
    model, tokenizer = load(args.model)
    print(f"Loaded in {time.time() - t0:.1f}s")

    if not hasattr(model, "mtp"):
        print("❌ Model doesn't have MTP module")
        return

    # Load/inject MTP weights
    mtp_weights = None
    if args.mtp_weights:
        mtp_weights = mx.load(args.mtp_weights)
    elif args.raw_hf_dir:
        import json
        idx = json.load(open(f"{args.raw_hf_dir}/model.safetensors.index.json"))
        mtp_shards = set()
        for k, v in idx["weight_map"].items():
            if "mtp." in k.lower():
                mtp_shards.add(v)
        mtp_weights = {}
        for shard in sorted(mtp_shards):
            w = mx.load(f"{args.raw_hf_dir}/{shard}")
            for k, v in w.items():
                if "mtp." in k.lower():
                    new_k = k[4:]  # Strip "mtp." prefix
                    mtp_weights[new_k] = v
        # Apply norm shifts (all except norm.weight which is not delta format)
        norm_suffixes = (
            ".input_layernorm.weight", ".post_attention_layernorm.weight",
            ".q_norm.weight", ".k_norm.weight",
            ".pre_fc_norm_hidden.weight", ".pre_fc_norm_embedding.weight",
        )
        for k in list(mtp_weights.keys()):
            if any(k.endswith(s) for s in norm_suffixes) and mtp_weights[k].ndim == 1:
                mtp_weights[k] = (mtp_weights[k].astype(mx.float32) + 1.0).astype(mx.bfloat16)

    if mtp_weights is None:
        print("⚠️ No MTP weights provided. Use --mtp-weights or --raw-hf-dir")
        return

    n = _inject_mtp_weights(model, mtp_weights)
    print(f"Injected {n} MTP weights")

    # Run test
    tokens = mx.array([tokenizer.encode(args.prompt)])
    cache = make_prompt_cache(model)
    logits, h = model(tokens, cache=cache, return_hidden=True)
    mx.eval(logits, h)
    backbone_token = mx.argmax(logits[:, -1, :], axis=-1)
    generated = [int(backbone_token.item())]
    last_hidden = h[:, -1:, :]
    mtp_cache = model.make_mtp_cache()
    mtp_correct = 0

    for step in range(args.num_tokens):
        last_arr = mx.array([[generated[-1]]], dtype=mx.uint32)
        mtp_logits = model.mtp_forward(last_hidden, last_arr, cache=mtp_cache)
        mx.eval(mtp_logits)
        mtp_next = mx.argmax(mtp_logits[:, -1, :], axis=-1)

        logits, h = model(last_arr, cache=cache, return_hidden=True)
        mx.eval(logits, h)
        bb_next = mx.argmax(logits[:, -1, :], axis=-1)

        mtp_correct += int(mtp_next.item()) == int(bb_next.item())
        generated.append(int(bb_next.item()))
        last_hidden = h[:, -1:, :]

        if (step + 1) % 32 == 0:
            print(f"  Step {step+1:>4}  acceptance: {mtp_correct/(step+1)*100:.1f}%")

    final = mtp_correct / args.num_tokens * 100
    print(f"\n{'='*50}")
    print(f"MTP ACCEPTANCE RATE: {final:.1f}% ({mtp_correct}/{args.num_tokens})")
    print(f"Generated preview: {tokenizer.decode(generated)[:200]}")
    if final > 70:
        print(f"\n✅ ≥70% — MTP speculative decoding would give ~2-4x speedup!")
    elif final > 50:
        print(f"\n👍 50-70% — ~1.5-2x speedup.")
    else:
        print(f"\n⚠️ Below 50% — marginal.")


if __name__ == "__main__":
    main()

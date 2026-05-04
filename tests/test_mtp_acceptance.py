"""TDD: MTP acceptance rate test — validates MTP weight loading + forward pass.

RED:   This test is written first. It should FAIL with the current code
       (MTP produces 0-3% acceptance).
GREEN: After fixing MTP weight loading and forward pass, this test should PASS
       (MTP acceptance rate >= 30%).
"""
import time
import mlx.core as mx
import pytest
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


def _load_model_with_mtp(model_path, mtp_weights_path=None):
    """Load a model and verify MTP module exists."""
    model, tokenizer = load(model_path)

    # Check MTP module exists
    if not hasattr(model, "mtp"):
        pytest.skip("Model doesn't have MTP module")
        return None, None

    # For models with separate MTP weights file (e.g. UD format)
    if mtp_weights_path is not None:
        import mlx.core as _mx
        w = _mx.load(mtp_weights_path)
        # Fixup RMSNorm weights (HF offset format: stored = actual - 1.0)
        suffixes = (
            '.input_layernorm.weight', '.post_attention_layernorm.weight',
            '.q_norm.weight', '.k_norm.weight',
            '.pre_fc_norm_hidden.weight', '.pre_fc_norm_embedding.weight',
            '.norm.weight',
        )
        for k, v in w.items():
            if v.ndim == 1 and any(k.endswith(s) for s in suffixes):
                if v.mean().item() < 0.5:
                    w[k] = v + 1.0
        # Load onto MTP module (load_weights may skip norms, so assign directly)
        model.mtp.load_weights(list(w.items()), strict=False)
        for k, v in w.items():
            if v.ndim == 1:
                parts = k.split('.')
                target = model.mtp
                for p in parts[:-1]:
                    if p.isdigit():
                        target = target[int(p)]
                    else:
                        target = getattr(target, p)
                setattr(target, parts[-1], v)

    return model, tokenizer


def _run_mtp_acceptance(model, tokenizer, prompt="Hello world", num_tokens=16):
    """Run MTP forward pass and measure acceptance rate.

    Returns: (acceptance_rate, num_tested, num_correct)
    """
    p = mx.array([tokenizer.encode(prompt)])
    bc = make_prompt_cache(model)
    logits, h = model(p, cache=bc, return_hidden=True)
    mx.eval(logits, h)

    first = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    last_h = h[:, -1:, :]
    mc = model.make_mtp_cache()

    # Align MTP cache offset with backbone position
    first_offset = next(
        c.offset for c in bc if hasattr(c, "offset") and not isinstance(c, type(bc[0]) if "ArraysCache" not in type(c).__name__ else type(c))
    )
    for c in mc:
        c.offset = int(first_offset)

    correct = 0
    for step in range(num_tokens):
        last_arr = mx.array([[first if step == 0 else int(last_bt.item())]], dtype=mx.uint32)
        if step > 0:
            b, h2 = model(last_arr, cache=bc, return_hidden=True)
            mx.eval(b, h2)
            last_h = h2[:, -1:, :]
            last_bt = mx.argmax(b[:, -1, :], axis=-1)
        else:
            last_bt = mx.array([first])

        m = model.mtp_forward(last_h, last_arr, cache=mc)
        mx.eval(m)
        mt = int(mx.argmax(m[0, -1, :], axis=-1).item())
        bt = int(last_bt.item())

        if mt == bt:
            correct += 1

    return correct / num_tokens, num_tokens, correct


# ---------------------------------------------------------------------------
# ConfigI model — MTP weights embedded in main checkpoint
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not __import__("os").path.exists(
    "/Users/shivam94/.cache/huggingface/hub/Qwen3.6-27B-ConfigI-MLX/config.json"
), reason="ConfigI model not cached locally")
class TestMTPConfigI:
    """MTP acceptance test using ConfigI model (weights in main checkpoint)."""

    MODEL_PATH = "/Users/shivam94/.cache/huggingface/hub/Qwen3.6-27B-ConfigI-MLX"

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model, self.tokenizer = _load_model_with_mtp(self.MODEL_PATH)
        if self.model is None:
            pytest.skip("MTP not available")

    def test_mtp_module_exists(self):
        """MTP module should be present and have loaded weights."""
        from mlx.utils import tree_flatten
        params = list(tree_flatten(self.model.mtp.parameters()))
        assert len(params) == 15, f"Expected 15 MTP params, got {len(params)}"

        # Linear weights should have non-zero values (loaded from checkpoint)
        fc = self.model.mtp.fc.weight
        assert abs(mx.mean(fc).item()) > 1e-6, "fc.weight not loaded (mean near zero)"

    def test_return_hidden_works(self):
        """Model should support return_hidden with correct shape."""
        p = mx.array([self.tokenizer.encode("Hello")])
        bc = make_prompt_cache(self.model)
        logits, h = self.model(p, cache=bc, return_hidden=True)
        mx.eval(logits, h)
        assert h.shape[-1] == self.model.language_model.args.hidden_size

    def test_mtp_forward_runs(self):
        """MTP forward should produce valid logits without crashing."""
        p = mx.array([self.tokenizer.encode("Hello")])
        bc = make_prompt_cache(self.model)
        logits, h = self.model(p, cache=bc, return_hidden=True)
        mx.eval(logits, h)
        first = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        mc = self.model.make_mtp_cache()
        for c in mc:
            c.offset = int(next(
                c2.offset for c2 in bc if hasattr(c2, "offset")
            ))
        la = mx.array([[first]], dtype=mx.uint32)
        m = self.model.mtp_forward(h[:, -1:, :], la, cache=mc)
        mx.eval(m)
        assert m.shape == (1, 1, self.model.language_model.args.vocab_size)

    def test_mtp_acceptance_above_30_percent(self):
        """MTP acceptance rate should be >= 30%."""
        rate, total, correct = _run_mtp_acceptance(
            self.model, self.tokenizer, prompt="Hello world", num_tokens=16
        )
        print(f"\n  MTP acceptance: {correct}/{total} = {rate*100:.1f}%")
        assert rate >= 0.30, (
            f"MTP acceptance too low: {rate*100:.1f}% ({correct}/{total}). "
            f"Expected >= 30%. MTP weights may not be loaded correctly."
        )


# ---------------------------------------------------------------------------
# UD model — MTP weights in separate file
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not __import__("os").path.exists(
    "/Users/shivam94/.cache/huggingface/hub/Qwen3.6-27B-UD-MLX-4bit/config.json"
), reason="UD model not cached locally")
class TestMTPUD:
    """MTP acceptance test using UD model (separate mtp_weights.safetensors)."""

    MODEL_PATH = "/Users/shivam94/.cache/huggingface/hub/Qwen3.6-27B-UD-MLX-4bit"
    MTP_W_PATH = MODEL_PATH + "/mtp_weights.safetensors"

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model, self.tokenizer = _load_model_with_mtp(
            self.MODEL_PATH, mtp_weights_path=self.MTP_W_PATH
        )
        if self.model is None:
            pytest.skip("MTP not available")

    def test_mtp_module_exists(self):
        from mlx.utils import tree_flatten
        params = list(tree_flatten(self.model.mtp.parameters()))
        assert len(params) == 15

    def test_mtp_acceptance_above_30_percent(self):
        rate, total, correct = _run_mtp_acceptance(
            self.model, self.tokenizer, prompt="Hello world", num_tokens=16
        )
        print(f"\n  MTP acceptance: {correct}/{total} = {rate*100:.1f}%")
        assert rate >= 0.30


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

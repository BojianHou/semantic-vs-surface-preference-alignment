"""Unit tests for Support-Augmented KL-Dr.DPO (length_pref.py).

Plain stdlib ``unittest`` (the venv has no pytest). Run from the project root:

    .venv/bin/python -m unittest -v test_counterfactual_kl_dro

Everything is CPU-only and self-contained: the model used by the trainer-level
tests is a randomly-initialized 2-layer Qwen2 with a 64-token vocab, so no
download and no GPU are needed.

Coverage (one test per requirement):
    1  backward compatibility (flag off == pre-change dpo / drdpo losses)
    2  equal-loss identity           A = c when ell^0 = ell^L = ell^LS = c
    3  large-tau limit               A -> sum_k p_k ell^k
    4  small-tau limit               A -> max_k ell^k
    5  hard-example weighting        raising ell^k raises q^k and A
    6  probability / temperature validation
    7  gradient flow into all three rejected branches
    8  missing counterfactual columns -> clear error
    9  numerical stability (huge losses, tiny tau, bf16/fp16)
   10  smoke test: forward + backward + optimizer step
"""

import unittest

import torch
import torch.nn.functional as F

from length_pref import (
    CF_BRANCHES,
    CF_KEYS,
    CF_SUPPORTED_METHODS,
    LengthPrefDataCollator,
    LengthPrefTrainer,
    counterfactual_kl_aggregate,
    counterfactual_probs,
    validate_counterfactual_config,
)

P_DEFAULT = (0.85, 0.10, 0.05)  # eps_L=0.10, eps_LS=0.05


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def tiny_model(vocab=64, seed=0):
    """A 2-layer randomly-initialized Qwen2 — same architecture family as the real
    runs, small enough to train on CPU in a test."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        pad_token_id=0,
    )
    return Qwen2ForCausalLM(cfg)


def synthetic_batch(batch_size, n_blocks, seq_len=12, prompt_len=5, vocab=64, seed=1):
    """A role-blocked [n_blocks*B, L] batch with response-only loss masks."""
    g = torch.Generator().manual_seed(seed)
    n = n_blocks * batch_size
    ids = torch.randint(1, vocab, (n, seq_len), generator=g)
    am = torch.ones_like(ids)
    lm = torch.zeros_like(ids)
    lm[:, prompt_len:] = 1
    return {"input_ids": ids, "attention_mask": am, "loss_mask": lm,
            "is_tie": torch.zeros(batch_size, dtype=torch.bool)}


def desync_policy(trainer, seed=0, scale=0.05):
    """Perturb the policy AFTER the reference snapshot was taken.

    LengthPrefTrainer clones the initial policy as the frozen reference, so at
    construction time policy == reference and EVERY margin is exactly 0 — which
    makes ell^0 = ell^L = ell^LS = log 2 and hides any branch-dependent behaviour.
    A small deterministic perturbation puts the policy off the reference, the way
    it is after a few optimizer steps in a real run.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in trainer.model.parameters():
            p.add_(torch.randn(p.shape, generator=g) * scale)
    return trainer


def make_trainer(lp_cfg, model=None, tmpdir="/tmp/cf_kl_dro_test"):
    """A LengthPrefTrainer wired to a tiny model, no dataset, no reporting."""
    from transformers import TrainingArguments

    args = TrainingArguments(
        output_dir=tmpdir, per_device_train_batch_size=2, report_to=[],
        logging_strategy="no", save_strategy="no", use_cpu=True,
    )
    return LengthPrefTrainer(model=model if model is not None else tiny_model(),
                             args=args, lp_cfg=lp_cfg)


def base_cfg(method="drdpo", **over):
    cfg = dict(method=method, beta=0.1, rdpo_alpha=0.05, drdpo_beta_prime=1.0,
               drdpo_stratify_length=False, drdpo_length_penalty=False, drdpo_buffer=0,
               lambda_len_inv=0.0, lmpo_gamma_beta_ratio=0.55, lmpo_lambda=0.2,
               lmpo_k=5.0, tie_weight=1.0, tie_len_tol=5, label_smoothing=0.0, seed=42)
    cfg.update(over)
    return cfg


def cf_cfg(method="drdpo", eps_l=0.10, eps_ls=0.05, tau=1.0, **over):
    return base_cfg(method, use_counterfactual_kl_dro=True, cf_epsilon_length=eps_l,
                    cf_epsilon_length_syntax=eps_ls, cf_kl_temperature=tau, **over)


# --------------------------------------------------------------------------- #
# 2-5, 9: the aggregation math (pure function, float64 so limits are exact)
# --------------------------------------------------------------------------- #
class TestAggregateMath(unittest.TestCase):

    def test_equal_loss_identity(self):
        """(2) ell^0 = ell^L = ell^LS = c  =>  A = c exactly (sum p_k = 1)."""
        for c in (0.0, 0.3, 2.5, 17.0):
            ell = torch.full((4, 3), c, dtype=torch.float64)
            A, q = counterfactual_kl_aggregate(ell, P_DEFAULT, 1.0)
            self.assertTrue(torch.allclose(A, torch.full((4,), c, dtype=torch.float64),
                                           atol=1e-12), f"A={A} for c={c}")
            # with equal losses the adversary keeps the nominal distribution
            self.assertTrue(torch.allclose(q[0], torch.tensor(P_DEFAULT, dtype=torch.float64),
                                           atol=1e-12))

    def test_large_temperature_limit(self):
        """(3) tau_v -> inf  =>  A -> p_0 ell^0 + p_L ell^L + p_LS ell^LS."""
        ell = torch.tensor([[0.2, 1.4, 3.1], [2.0, 0.1, 0.9]], dtype=torch.float64)
        nominal = (ell * torch.tensor(P_DEFAULT, dtype=torch.float64)).sum(-1)
        prev = float("inf")
        for tau in (1e2, 1e4, 1e6):
            A, _ = counterfactual_kl_aggregate(ell, P_DEFAULT, tau)
            err = float((A - nominal).abs().max())
            self.assertLess(err, prev)          # monotonically approaching
            prev = err
        self.assertLess(prev, 1e-6)

    def test_small_temperature_limit(self):
        """(4) tau_v -> 0  =>  A -> max_k ell^k."""
        ell = torch.tensor([[0.2, 1.4, 3.1], [2.0, 0.1, 0.9]], dtype=torch.float64)
        worst = ell.max(dim=-1).values
        for tau, tol in ((1e-1, 5e-1), (1e-2, 5e-2), (1e-3, 5e-3)):
            A, q = counterfactual_kl_aggregate(ell, P_DEFAULT, tau)
            self.assertTrue(torch.allclose(A, worst, atol=tol), f"tau={tau} A={A}")
            self.assertLessEqual(float(A.max() - worst.max()), 1e-9)  # A <= max always
        # at tau=1e-3 essentially all mass sits on the argmax branch
        self.assertGreater(float(q[0, 2]), 0.999)
        self.assertGreater(float(q[1, 0]), 0.999)

    def test_hard_example_weighting(self):
        """(5) raising one counterfactual's loss raises its q AND raises A."""
        ell = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)
        A0, q0 = counterfactual_kl_aggregate(ell, P_DEFAULT, 1.0)
        harder = ell.clone()
        harder[0, 1] += 2.0                      # the length counterfactual gets harder
        A1, q1 = counterfactual_kl_aggregate(harder, P_DEFAULT, 1.0)
        self.assertGreater(float(q1[0, 1]), float(q0[0, 1]))   # its weight grows
        self.assertLess(float(q1[0, 0]), float(q0[0, 0]))      # others shrink
        self.assertGreater(float(A1), float(A0))               # robust loss grows
        # ...and monotonically so
        prev_q, prev_A = float(q0[0, 1]), float(A0)
        for bump in (0.5, 1.0, 2.0, 4.0):
            e = ell.clone()
            e[0, 1] += bump
            A, q = counterfactual_kl_aggregate(e, P_DEFAULT, 1.0)
            self.assertGreater(float(q[0, 1]), prev_q)
            self.assertGreater(float(A), prev_A)
            prev_q, prev_A = float(q[0, 1]), float(A)

    def test_vanishing_counterfactual_mass(self):
        """(limit 5) eps -> 0 => A -> the original pair's loss."""
        ell = torch.tensor([[0.7, 5.0, 4.0]], dtype=torch.float64)
        prev_err, prev_mass = float("inf"), float("inf")
        for eps in (1e-2, 1e-4, 1e-8):
            probs = counterfactual_probs(eps, eps)
            A, q = counterfactual_kl_aggregate(ell, probs, 1.0)
            err = abs(float(A) - 0.7)
            mass = float(q[0, 1] + q[0, 2])
            self.assertLess(err, prev_err)          # converges to the original loss
            self.assertLess(mass, prev_mass)        # counterfactual mass vanishes
            prev_err, prev_mass = err, mass
        # residual at eps=1e-8 is eps*(e^5+e^4)/e^0.7 ~ 1e-6, as the expansion predicts
        self.assertAlmostEqual(float(A), 0.7, places=5)
        self.assertLess(mass, 1e-5)

    def test_bounds(self):
        """A is always between the nominal mean and the max (both inclusive)."""
        torch.manual_seed(0)
        ell = torch.rand(64, 3, dtype=torch.float64) * 10
        p = torch.tensor(P_DEFAULT, dtype=torch.float64)
        for tau in (0.05, 0.5, 1.0, 5.0):
            A, _ = counterfactual_kl_aggregate(ell, P_DEFAULT, tau)
            self.assertTrue(torch.all(A >= (ell * p).sum(-1) - 1e-9))
            self.assertTrue(torch.all(A <= ell.max(-1).values + 1e-9))

    def test_numerical_stability(self):
        """(9) huge losses, tiny tau and half precision stay finite."""
        big = torch.tensor([[1e3, 5e3, 2e3], [0.0, 1e4, 1e-8]], dtype=torch.float32)
        for tau in (1e-3, 1e-2, 1.0):
            A, q = counterfactual_kl_aggregate(big, P_DEFAULT, tau)
            self.assertTrue(torch.isfinite(A).all(), f"A={A} tau={tau}")
            self.assertTrue(torch.isfinite(q).all())
            self.assertTrue(torch.allclose(q.sum(-1), torch.ones(2, dtype=q.dtype), atol=1e-9))
            # exp(5e3/1e-3) would overflow; logsumexp must give ~the max instead.
            # A = max + tau*log(p_argmax) + O(tau*exp(-gap/tau)), so the offset from
            # the max is bounded by tau*|log p_min|.
            self.assertLessEqual(float(A[0]), 5e3 + 1e-9)
            self.assertGreaterEqual(float(A[0]), 5e3 - tau * 3.0)
        for dtype in (torch.bfloat16, torch.float16):
            ell = torch.tensor([[10.0, 400.0, 30.0]], dtype=dtype)
            A, q = counterfactual_kl_aggregate(ell, P_DEFAULT, 0.5)
            self.assertEqual(A.dtype, dtype)          # cast back to the training dtype
            self.assertTrue(torch.isfinite(A.float()).all())
            self.assertTrue(torch.isfinite(q.float()).all())

    def test_gradient_reaches_every_branch(self):
        """(7) all three branches get finite, non-zero gradient."""
        ell = torch.tensor([[0.5, 1.5, 0.9], [2.0, 0.3, 0.4]],
                           dtype=torch.float64, requires_grad=True)
        A, _ = counterfactual_kl_aggregate(ell, P_DEFAULT, 1.0)
        A.sum().backward()
        self.assertTrue(torch.isfinite(ell.grad).all())
        self.assertTrue(torch.all(ell.grad > 0), ell.grad)
        # dA/dell^k == q^k
        _, q = counterfactual_kl_aggregate(ell.detach(), P_DEFAULT, 1.0)
        self.assertTrue(torch.allclose(ell.grad, q, atol=1e-10))

    def test_q_is_detached(self):
        ell = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
        _, q = counterfactual_kl_aggregate(ell, P_DEFAULT, 1.0)
        self.assertFalse(q.requires_grad)


# --------------------------------------------------------------------------- #
# 6: configuration validation
# --------------------------------------------------------------------------- #
class TestValidation(unittest.TestCase):

    def test_valid_defaults(self):
        probs, tau = validate_counterfactual_config(0.10, 0.05, 1.0)
        self.assertAlmostEqual(sum(probs), 1.0)
        self.assertEqual(probs, (0.85, 0.10, 0.05))
        self.assertEqual(tau, 1.0)

    def test_negative_epsilon(self):
        for bad in ((-0.1, 0.05), (0.1, -0.05)):
            with self.assertRaises(ValueError):
                counterfactual_probs(*bad)

    def test_zero_epsilon_rejected(self):
        """Every enabled support point must carry strictly positive mass."""
        for bad in ((0.0, 0.05), (0.1, 0.0)):
            with self.assertRaises(ValueError):
                counterfactual_probs(*bad)

    def test_epsilon_sum_ge_one(self):
        for bad in ((0.6, 0.4), (0.9, 0.5), (1.0, 0.1)):
            with self.assertRaises(ValueError):
                counterfactual_probs(*bad)

    def test_bad_temperature(self):
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                validate_counterfactual_config(0.10, 0.05, bad)

    def test_trainer_rejects_bad_config_before_training(self):
        with self.assertRaises(ValueError):
            make_trainer(cf_cfg(eps_l=-0.1))
        with self.assertRaises(ValueError):
            make_trainer(cf_cfg(tau=0.0))

    def test_unsupported_method_rejected(self):
        self.assertEqual(tuple(CF_SUPPORTED_METHODS), ("dpo", "rdpo", "drdpo"))
        for method in ("sampo", "lmpo", "tie"):
            with self.assertRaises(ValueError):
                make_trainer(cf_cfg(method=method))

    def test_train_py_arg_validation(self):
        """train.py fails fast on an invalid cf config / method."""
        import argparse

        from train import _validate_counterfactual_args

        def mk(**over):
            d = dict(use_counterfactual_kl_dro=True, method="drdpo", cf_epsilon_length=0.1,
                     cf_epsilon_length_syntax=0.05, cf_kl_temperature=1.0,
                     drdpo_beta_prime=1.0)
            d.update(over)
            return argparse.Namespace(**d)

        _validate_counterfactual_args(mk())                       # valid
        _validate_counterfactual_args(mk(use_counterfactual_kl_dro=False, method="simpo"))
        for bad in (dict(method="simpo"), dict(method="sampo"), dict(cf_epsilon_length=0.0),
                    dict(cf_epsilon_length_syntax=1.5), dict(cf_kl_temperature=-1.0)):
            with self.assertRaises(ValueError):
                _validate_counterfactual_args(mk(**bad))


# --------------------------------------------------------------------------- #
# 8: data-format errors
# --------------------------------------------------------------------------- #
class _StubTokenizer:
    """Minimal chat-template tokenizer so the collator can run without a download."""

    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(ch) % 60 + 1 for ch in text]}


class TestMissingColumns(unittest.TestCase):

    def _examples(self, keys):
        return [{"prompt": "p", **{k: f"text-{k}" for k in keys}} for _ in range(2)]

    def test_missing_counterfactual_columns(self):
        """(8) flag on + counterfactual columns absent => clear, named error."""
        coll = LengthPrefDataCollator(tokenizer=_StubTokenizer(), counterfactual=True)
        with self.assertRaises(ValueError) as ctx:
            coll(self._examples(["chosen", "rejected"]))
        msg = str(ctx.exception)
        self.assertIn("rejected_double_prime", msg)
        self.assertIn("rejected_prime", msg)
        self.assertIn("use_counterfactual_kl_dro", msg)

    def test_partially_missing_column_named(self):
        coll = LengthPrefDataCollator(tokenizer=_StubTokenizer(), counterfactual=True)
        with self.assertRaises(ValueError) as ctx:
            coll(self._examples(["chosen", "rejected", "rejected_prime"]))
        self.assertIn("rejected_double_prime", str(ctx.exception))
        self.assertNotIn("rejected_prime'", str(ctx.exception).split("missing")[-1])

    def test_four_blocks_when_enabled(self):
        coll = LengthPrefDataCollator(tokenizer=_StubTokenizer(), counterfactual=True)
        out = coll(self._examples(CF_KEYS))
        self.assertEqual(out["input_ids"].shape[0], 8)          # 4 blocks x B=2
        # same tokenizer/padding/masking rules for every block
        self.assertEqual(out["attention_mask"].shape, out["input_ids"].shape)
        self.assertEqual(out["loss_mask"].shape, out["input_ids"].shape)

    def test_two_blocks_when_disabled(self):
        """Flag off: the extra columns are simply never read."""
        coll = LengthPrefDataCollator(tokenizer=_StubTokenizer(), counterfactual=False)
        out = coll(self._examples(["chosen", "rejected"]))
        self.assertEqual(out["input_ids"].shape[0], 4)          # 2 blocks x B=2


# --------------------------------------------------------------------------- #
# 1: backward compatibility
# --------------------------------------------------------------------------- #
class TestBackwardCompatibility(unittest.TestCase):
    """Flag off => bit-for-bit the pre-change loss.

    The expectation is recomputed here from an INDEPENDENT implementation of the
    original formulas (plain DPO mean and the Dr.DPO dual), so the test would fail
    if the refactor into _outer_aggregate/_pair_loss changed anything.
    """

    def _expected(self, trainer, batch, method):
        from length_pref import _forward_logps

        with torch.no_grad():
            pol, counts, _, _ = _forward_logps(trainer.model, batch["input_ids"],
                                               batch["attention_mask"], batch["loss_mask"])
            ref, _, _, _ = _forward_logps(trainer.ref_model, batch["input_ids"],
                                          batch["attention_mask"], batch["loss_mask"])
        pw, pl = pol.chunk(2)
        rw, rl = ref.chunk(2)
        nw, nl = counts.chunk(2)
        beta = 0.1
        margin = (pw - rw) - (pl - rl)
        if method == "dpo":
            return -F.logsigmoid(beta * margin).mean()
        if method == "rdpo":
            return -F.logsigmoid(beta * margin - 0.05 * (nw.float() - nl.float())).mean()
        per_ex = -F.logsigmoid(beta * margin)               # drdpo
        n = per_ex.shape[0]
        return -1.0 * (torch.logsumexp(-per_ex / 1.0, dim=0) - torch.log(torch.tensor(float(n))))

    def test_flag_off_matches_original(self):
        batch = synthetic_batch(batch_size=4, n_blocks=2)
        for method in ("dpo", "rdpo", "drdpo"):
            with self.subTest(method=method):
                # desync first: with policy == reference every margin is 0 and every
                # formula collapses to log 2, which would make this test vacuous.
                trainer = desync_policy(make_trainer(base_cfg(method),
                                                     model=tiny_model(seed=3)))
                self.assertFalse(trainer.cf_on)
                got = trainer.compute_loss(trainer.model, batch)
                want = self._expected(trainer, batch, method)
                self.assertNotAlmostEqual(float(want), 0.6931471, places=4)
                self.assertAlmostEqual(float(got), float(want), places=6)

    def test_flag_absent_from_cfg_is_off(self):
        """Old lp_cfg dicts (no cf keys at all) still work — checkpoint/config compat."""
        cfg = base_cfg("drdpo")
        self.assertNotIn("use_counterfactual_kl_dro", cfg)
        trainer = make_trainer(cfg, model=tiny_model(seed=3))
        loss = trainer.compute_loss(trainer.model, synthetic_batch(4, 2))
        self.assertTrue(torch.isfinite(loss))

    def test_drdpo_variants_unchanged(self):
        """The stratified / buffered / length-penalty Dr.DPO arms still run."""
        batch = synthetic_batch(batch_size=4, n_blocks=2)
        for over in (dict(drdpo_stratify_length=True), dict(drdpo_buffer=16),
                     dict(drdpo_length_penalty=True), dict(lambda_len_inv=5e-4)):
            with self.subTest(**over):
                trainer = desync_policy(make_trainer(base_cfg("drdpo", **over),
                                                     model=tiny_model(seed=3)))
                self.assertTrue(torch.isfinite(trainer.compute_loss(trainer.model, batch)))

    def test_original_pair_recovered_as_eps_vanishes(self):
        """cf ON with eps->0 reproduces the flag-OFF loss on the same pair.

        Uses a 4-block batch whose first two blocks are the plain (chosen, rejected)
        pair, so the flag-off loss on those two blocks is the reference value.
        """
        batch4 = synthetic_batch(batch_size=3, n_blocks=4, seed=7)
        two = {k: (v[:6] if k != "is_tie" else v) for k, v in batch4.items()}
        off = desync_policy(make_trainer(base_cfg("drdpo"), model=tiny_model(seed=5)))
        want = float(off.compute_loss(off.model, two))
        on = desync_policy(make_trainer(cf_cfg("drdpo", eps_l=1e-9, eps_ls=1e-9),
                                        model=tiny_model(seed=5)))
        got = float(on.compute_loss(on.model, batch4))
        self.assertAlmostEqual(got, want, places=5)


# --------------------------------------------------------------------------- #
# 4 / 7 / 10: trainer-level counterfactual behaviour
# --------------------------------------------------------------------------- #
class TestTrainerCounterfactual(unittest.TestCase):

    def test_loss_is_finite_and_metrics_present(self):
        trainer = desync_policy(make_trainer(cf_cfg("drdpo"), model=tiny_model(seed=11)))
        loss = trainer.compute_loss(trainer.model, synthetic_batch(3, 4))
        self.assertTrue(torch.isfinite(loss))
        m = trainer.cf_metrics_extra
        for name in CF_BRANCHES:
            for prefix in ("loss", "margin", "weight", "accuracy"):
                self.assertIn(f"cf/{prefix}_{name}", m)
        self.assertIn("cf/counterfactual_mass", m)
        self.assertIn("cf/robust_local_loss", m)
        self.assertIn("cf/worst_orbit_accuracy", m)
        # weights are a probability distribution and mass adds up
        w = sum(m[f"cf/weight_{n}"] for n in CF_BRANCHES)
        self.assertAlmostEqual(w, 1.0, places=5)
        self.assertAlmostEqual(m["cf/counterfactual_mass"],
                               m["cf/weight_length"] + m["cf/weight_length_syntax"],
                               places=6)
        # worst-orbit accuracy can never exceed any single-branch accuracy
        for name in CF_BRANCHES:
            self.assertLessEqual(m["cf/worst_orbit_accuracy"], m[f"cf/accuracy_{name}"] + 1e-9)

    def test_gradient_flows_to_all_three_rejected_blocks(self):
        """(7) each rejected branch receives finite, non-zero gradient."""
        model = tiny_model(seed=13)
        trainer = desync_policy(make_trainer(cf_cfg("drdpo"), model=model), seed=3)
        batch = synthetic_batch(2, 4, seed=17)
        emb = model.get_input_embeddings()

        grads = {}

        def hook(_module, _gin, gout):
            grads["emb"] = gout[0].detach().clone()

        handle = emb.register_full_backward_hook(hook)
        loss = trainer.compute_loss(model, batch)
        loss.backward()
        handle.remove()

        g = grads["emb"]                       # [4B, L, H]
        self.assertTrue(torch.isfinite(g).all())
        blocks = g.chunk(4)
        for name, blk in zip(CF_KEYS, blocks):
            self.assertGreater(float(blk.abs().sum()), 0.0, f"no gradient into {name}")
        # and the parameters themselves got a finite, non-zero gradient
        total = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
        self.assertGreater(total, 0.0)
        self.assertTrue(all(torch.isfinite(p.grad).all()
                            for p in model.parameters() if p.grad is not None))

    def test_small_tau_emphasises_worst_pair(self):
        """(4, trainer level) tiny tau_v => loss ~ the WORST branch, not the easiest.

        Guards against a sign inversion: with tau_v -> 0 the per-example robust loss
        must approach max_k ell^k (hardest transformation), never min_k.
        """
        trainer = desync_policy(make_trainer(cf_cfg("dpo", tau=1e-3),
                                             model=tiny_model(seed=19)), seed=2)
        batch = synthetic_batch(4, 4, seed=23)
        loss = float(trainer.compute_loss(trainer.model, batch))
        m = trainer.cf_metrics_extra
        branch_losses = [m[f"cf/loss_{n}"] for n in CF_BRANCHES]
        self.assertGreater(max(branch_losses) - min(branch_losses), 1e-6,
                           "degenerate batch: the three branches must differ")
        # dpo's outer aggregation is the plain mean, so mean_i(max_k) >= max_k(mean_i)
        self.assertGreaterEqual(loss + 1e-6, max(branch_losses))
        self.assertGreater(loss, min(branch_losses))

    def test_large_tau_matches_nominal_mixture(self):
        """(3, trainer level) huge tau_v => loss ~ nominal weighted mean of branches."""
        trainer = desync_policy(make_trainer(cf_cfg("dpo", tau=1e6),
                                             model=tiny_model(seed=19)), seed=2)
        loss = float(trainer.compute_loss(trainer.model, synthetic_batch(4, 4, seed=23)))
        m = trainer.cf_metrics_extra
        branch_losses = [m[f"cf/loss_{n}"] for n in CF_BRANCHES]
        self.assertGreater(max(branch_losses) - min(branch_losses), 1e-6)
        nominal = sum(p * m[f"cf/loss_{n}"] for p, n in zip(P_DEFAULT, CF_BRANCHES))
        self.assertAlmostEqual(loss, nominal, places=5)

    def test_chosen_block_used_once(self):
        """Efficiency: exactly one policy forward and one reference forward per step."""
        trainer = make_trainer(cf_cfg("drdpo"), model=tiny_model(seed=29))
        calls = {"policy": 0, "ref": 0}
        pol_fwd, ref_fwd = trainer.model.forward, trainer.ref_model.forward

        def count_pol(*a, **k):
            calls["policy"] += 1
            return pol_fwd(*a, **k)

        def count_ref(*a, **k):
            calls["ref"] += 1
            return ref_fwd(*a, **k)

        trainer.model.forward, trainer.ref_model.forward = count_pol, count_ref
        trainer.compute_loss(trainer.model, synthetic_batch(2, 4))
        self.assertEqual(calls, {"policy": 1, "ref": 1})

    def test_reference_is_no_grad(self):
        trainer = make_trainer(cf_cfg("drdpo"), model=tiny_model(seed=31))
        trainer.compute_loss(trainer.model, synthetic_batch(2, 4)).backward()
        self.assertTrue(all(p.grad is None for p in trainer.ref_model.parameters()))

    def test_outer_aggregation_identical_given_same_per_example_loss(self):
        """The outer Dr.DPO dual is untouched: feeding it A_i vs ell_i is the only change."""
        trainer = make_trainer(cf_cfg("drdpo"), model=tiny_model(seed=37))
        per_ex = torch.tensor([0.2, 1.0, 3.0, 0.5])
        want = trainer._dro_dual(per_ex, 1.0)
        got = trainer._outer_aggregate("drdpo", per_ex, torch.tensor([True, False, True, False]))
        self.assertAlmostEqual(float(got), float(want), places=9)

    def test_smoke_forward_backward_step(self):
        """(10) a full training step on a tiny synthetic batch."""
        model = tiny_model(seed=41)
        trainer = desync_policy(make_trainer(cf_cfg("drdpo"), model=model), seed=4)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = [p.detach().clone() for p in model.parameters()]
        losses = []
        for step in range(2):
            loss = trainer.compute_loss(model, synthetic_batch(2, 4, seed=100 + step))
            self.assertTrue(torch.isfinite(loss))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        moved = any(not torch.equal(b, p) for b, p in zip(before, model.parameters()))
        self.assertTrue(moved, "optimizer step did not change any parameter")
        self.assertTrue(all(t == t for t in losses))          # no NaN

    def test_mixed_precision_step(self):
        """(9, trainer level) bf16 autocast forward stays finite end to end."""
        model = tiny_model(seed=43)
        trainer = make_trainer(cf_cfg("drdpo"), model=model)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            loss = trainer.compute_loss(model, synthetic_batch(2, 4, seed=47))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all()
                            for p in model.parameters() if p.grad is not None))


if __name__ == "__main__":
    unittest.main(verbosity=2)

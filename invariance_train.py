"""Length/syntax-invariant preference alignment — search harness for I2 & I3.

Reference-based DPO (sum-logp, sigmoid, β) on the 4-way decomposed set, with knobs:

  --neg_key {rejected, rejected_prime, rejected_double_prime}   # I2: length(+syntax)-matched negative
  --lambda_cons FLOAT   --cons_variants {rpp,rp,both}           # I3: counterfactual consistency
  --group_dro {off,worst}                                        # (R6 baseline; kept for completeness)

Margin (DPO logratio diff):  m(a,b) = (logπθ(a)−logπref(a)) − (logπθ(b)−logπref(b))
  base preference : -logσ(β · m(chosen, NEG))
  I3 consistency  : λ · mean_v ( m(chosen,rejected) − m(chosen, v) )²   for v in cons variants
     — forces the chosen-vs-rejected preference to be INVARIANT to swapping the
       negative for its length/syntax-matched twin (decision-level invariance).

Reuses sipo.SIPODataCollator (encodes all 4 responses) + sipo._forward_logps +
train.DecomposedLogpCallback. Ref model = frozen base policy.
"""
import argparse
import os

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

import train  # noqa: F401  (tokenizer workaround + eval callback)
from sipo import SIPODataCollator, _forward_logps

KEYS = ("chosen", "rejected_prime", "rejected_double_prime", "rejected")  # collator role order


class InvarianceTrainer(Trainer):
    def __init__(self, *a, ref_model=None, cfg=None, **k):
        super().__init__(*a, **k)
        self.ref_model = ref_model
        self.cfg = cfg
        self._leninv_ema = {}  # running per-env mean-margin estimates for the EMA len_inv variant

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        c = self.cfg
        ii, am, lm = inputs["input_ids"], inputs["attention_mask"], inputs["loss_mask"]
        pol, counts = _forward_logps(model, ii, am, lm)[:2]   # one policy forward (with grad)
        if self.ref_model.device != ii.device:
            self.ref_model.to(ii.device)
        with torch.no_grad():
            ref = _forward_logps(self.ref_model, ii, am, lm)[0]
        B = pol.shape[0] // 4
        # role blocks: chosen, rejected_prime, rejected_double_prime, rejected
        r = {k: (pol[i * B:(i + 1) * B] - ref[i * B:(i + 1) * B]) for i, k in enumerate(KEYS)}

        def margin(a, b):
            return r[a] - r[b]

        per_ex = -F.logsigmoid(c["beta"] * margin("chosen", c["neg_key"]))  # [B]

        cc, cr = counts[0:B], counts[3 * B:4 * B]   # chosen / rejected response token counts
        chosen_longer = cc > cr
        if c["group_dro"] == "worst":
            a = per_ex[chosen_longer].mean() if chosen_longer.any() else None
            b = per_ex[~chosen_longer].mean() if (~chosen_longer).any() else None
            base = torch.maximum(a, b) if (a is not None and b is not None) else (a if b is None else b)
        else:
            base = per_ex.mean()

        loss = base
        if c["lambda_cons"] > 0:
            variants = {"rpp": ["rejected_double_prime"], "rp": ["rejected_prime"],
                        "both": ["rejected_double_prime", "rejected_prime"]}[c["cons_variants"]]
            m_ref = margin("chosen", "rejected")
            # smooth_l1 (Huber): margins are sum-logp-scale (~±20 nats), so a squared
            # penalty explodes (grad-norm ~1e3+) and hijacks the update; Huber bounds it.
            cons = sum(F.smooth_l1_loss(margin("chosen", v), m_ref, beta=1.0) for v in variants) / len(variants)
            loss = loss + c["lambda_cons"] * cons

        # I4: length-invariance. The eval ranks RAW chosen_logp - rejected_logp; that raw
        # margin is systematically shifted by the length direction (chosen-longer subset gets
        # a lower margin, so acc_chosen_longer < acc_rejected_longer). I2/I3 on policy-ref
        # diffs cannot fix this because the ref's own length gap dominates g3. Here we
        # penalize the DIFFERENCE IN MEAN RAW POLICY MARGIN between the two length
        # environments -> forces the decision boundary to sit the same way regardless of
        # which response is longer, closing length_acc_gap without touching semantics.
        if c["lambda_len_inv"] > 0:
            pol_margin = pol[0:B] - pol[3 * B:4 * B]           # raw policy sum-logp: chosen - rejected
            if c["len_inv_ema"] > 0:
                # EMA-smoothed variant: keep a running (detached) estimate of each env's mean
                # margin, then pull EACH present env's current-batch mean toward the stop-grad
                # MIDPOINT of the two EMAs. Lower-variance target than an 8-example per-batch
                # difference, and every batch contributes (even single-env ones), so a weak
                # lambda can close the gap without the mean-eroding noise of the per-batch form.
                d = c["len_inv_ema"]
                terms = []
                for present, key in ((chosen_longer, "cl"), (~chosen_longer, "rl")):
                    if present.any():
                        cur = pol_margin[present].mean()
                        prev = self._leninv_ema.get(key)
                        self._leninv_ema[key] = cur.detach() if prev is None else d * prev + (1 - d) * cur.detach()
                        terms.append((key, cur))
                if self._leninv_ema.get("cl") is not None and self._leninv_ema.get("rl") is not None:
                    mid = ((self._leninv_ema["cl"] + self._leninv_ema["rl"]) / 2).detach()
                    if terms:
                        len_inv = sum((cur - mid) ** 2 for _, cur in terms) / len(terms)
                        loss = loss + c["lambda_len_inv"] * len_inv
            elif chosen_longer.any() and (~chosen_longer).any():  # per-batch form: need both envs
                len_inv = (pol_margin[chosen_longer].mean() - pol_margin[~chosen_longer].mean()) ** 2
                loss = loss + c["lambda_len_inv"] * len_inv

        # I5: IRMv1 environment-invariance. Treat the two length groups as environments and a
        # scalar dummy classifier w applied to the DPO logit z = beta * margin. The IRMv1
        # penalty sum_e (d/dw mean_e[-logsigmoid(w*z)] |_{w=1})^2 is minimized when w=1 is a
        # stationary point of EACH env's risk — i.e. the SAME preference classifier is optimal
        # regardless of which response is longer. Hand-derived: d/dw softplus(-w z)|w=1 = -z*sig(-z),
        # so g_e = mean_e[-z*sigmoid(-z)]. Unlike len_inv (which equates raw-margin MEANS via a bulk
        # logp shift that distorts g3), IRM only drives each env's boundary gradient to 0 — a targeted
        # push on the misclassified tail — so it should close length_acc_gap with less g3 distortion.
        if c["lambda_irm"] > 0 and chosen_longer.any() and (~chosen_longer).any():
            m_irm = (pol[0:B] - pol[3 * B:4 * B]) if c["irm_margin"] == "raw" else margin("chosen", c["neg_key"])
            z = c["beta"] * m_irm
            g_env = -z * torch.sigmoid(-z)   # per-example dummy-classifier gradient contribution
            pen = g_env[chosen_longer].mean() ** 2 + g_env[~chosen_longer].mean() ** 2
            loss = loss + c["lambda_irm"] * pen

        # I5: V-REx — penalize the VARIANCE of per-length-env DPO risk (equalize per-env LOSS, a
        # closer proxy to per-env accuracy than raw-margin means). With two envs, Var ∝ (R_cl - R_rl)^2.
        if c["lambda_vrex"] > 0 and chosen_longer.any() and (~chosen_longer).any():
            vrex = (per_ex[chosen_longer].mean() - per_ex[~chosen_longer].mean()) ** 2
            loss = loss + c["lambda_vrex"] * vrex
        return (loss, {"m": margin("chosen", "rejected")}) if return_outputs else loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--neg_key", choices=list(KEYS), default="rejected")
    p.add_argument("--lambda_cons", type=float, default=0.0)
    p.add_argument("--cons_variants", choices=["rpp", "rp", "both"], default="rpp")
    p.add_argument("--group_dro", choices=["off", "worst"], default="off")
    p.add_argument("--lambda_len_inv", type=float, default=0.0,
                   help="I4: penalize mean raw-margin gap between chosen-longer / rejected-longer envs")
    p.add_argument("--len_inv_ema", type=float, default=0.0,
                   help="I4: EMA decay (e.g. 0.9) for a lower-variance len_inv target; 0 = per-batch form")
    p.add_argument("--lambda_irm", type=float, default=0.0,
                   help="I5: IRMv1 penalty — force the DPO classifier (w=1) to be locally optimal in "
                        "BOTH length envs (sum_e (dR_e/dw|w=1)^2). Balances per-env accuracy without a "
                        "raw-margin mean-shift, so it should close length_acc_gap with less g3 distortion.")
    p.add_argument("--irm_margin", choices=["raw", "ref"], default="raw",
                   help="I5: which margin feeds the IRM logit. raw = pol(chosen)-pol(neg) (what eval ranks); "
                        "ref = ref-adjusted DPO margin (what base loss uses).")
    p.add_argument("--lambda_vrex", type=float, default=0.0,
                   help="I5: V-REx penalty — variance of per-length-env DPO risk (R_cl - R_rl)^2. "
                        "Equalizes per-env LOSS (accuracy proxy) rather than raw-margin means.")
    p.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--train_dataset", default="Bojian92/tldr_preference_decomposed")
    p.add_argument("--train_split", default="train")
    p.add_argument("--eval_dataset", default="Bojian92/tldr_preference_decomposed")
    p.add_argument("--eval_split", default="validation")
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=256)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--prompt_truncation_side", default="left")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--eval_every", type=int, default=20)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--result_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_save", action="store_true")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16)
    ref = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16).eval()
    for pm in ref.parameters():
        pm.requires_grad_(False)

    train_ds = load_dataset(args.train_dataset, split=args.train_split)
    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    eval_ds = load_dataset(args.eval_dataset, split=args.eval_split)
    if args.max_eval_samples:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))
    cfg = dict(neg_key=args.neg_key, lambda_cons=args.lambda_cons, cons_variants=args.cons_variants,
               group_dro=args.group_dro, beta=args.beta, lambda_len_inv=args.lambda_len_inv,
               len_inv_ema=args.len_inv_ema, lambda_irm=args.lambda_irm, irm_margin=args.irm_margin,
               lambda_vrex=args.lambda_vrex)
    print(f"[invariance] cfg={cfg} train_n={len(train_ds)} eval_n={len(eval_ds)}")

    ta = TrainingArguments(
        output_dir=args.output_dir, per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs, lr_scheduler_type="cosine", warmup_ratio=0.1,
        logging_steps=args.eval_every, save_strategy="no", bf16=True, seed=args.seed,
        remove_unused_columns=False, report_to=[],
    )
    callback = train.DecomposedLogpCallback(
        tokenizer=tok, eval_dataset=eval_ds, eval_every=args.eval_every,
        result_dir=os.path.join(args.result_dir, "dpo"), method="dpo", max_len=args.max_len,
        eval_batch_size=args.eval_batch_size, prompt_truncation_side=args.prompt_truncation_side,
        eval_at_start=True, use_wandb=False,
    )
    trainer = InvarianceTrainer(
        model=model, args=ta, train_dataset=train_ds, processing_class=tok,
        data_collator=SIPODataCollator(tokenizer=tok, max_len=args.max_len,
                                       prompt_truncation_side=args.prompt_truncation_side),
        ref_model=ref, cfg=cfg,
    )
    trainer.add_callback(callback)
    trainer.train()
    if not args.no_save:
        trainer.save_model(os.path.join(args.output_dir, "final"))
        tok.save_pretrained(os.path.join(args.output_dir, "final"))
    print(f"Done. -> {args.result_dir}/dpo")


if __name__ == "__main__":
    main()

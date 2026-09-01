"""Reconstruct Dr.DPO's KL-DRO pair weights and ask what they actually down-weight.

Dr.DPO (length_pref.py:226-233) replaces the batch-mean DPO loss with the KL-DRO
dual  -b' log E[exp(-L_i/b')].  Differentiating gives each pair an effective
gradient weight

    w_i  =  exp(-L_i / b')  /  sum_j exp(-L_j / b')          (within a micro-batch)

so LOW-loss pairs are up-weighted and HIGH-loss pairs are down-weighted (the
noise-robust direction, opposite to CVaR-style DRO). REPORT §9.9 finding 1
conjectured this "incidentally trims length-exploiting pairs". This script tests
that conjecture directly by measuring, on real training pairs:

  1. how peaked the weighting is        -> effective sample size (ESS / B)
  2. what it down-weights               -> mean weight for chosen-longer vs
                                           rejected-longer pairs
  3. whether loss tracks length at all  -> corr(L_i, |y_w| - |y_l|)

and repeats (1)-(3) over a grid of b' so the round-7 sweep can be predicted
before it finishes.

CAVEAT: losses are computed under a FIXED checkpoint (default: Dr.DPO's own
converged policy), not under the evolving policy that produced them during
training. This characterises the weighting regime at convergence, not the whole
trajectory.

Usage (from project root):
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/drdpo_weights.py
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/drdpo_weights.py \
        --policy checkpoints/dpo-v3/final --label DPO_v3
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train  # noqa: E402,F401  (import applies the transformers Mistral-regex workaround)
from length_pref import _encode, _forward_logps  # noqa: E402  (exact training encoding)

BETA_PRIME_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]


def pair_losses(policy, ref, tok, rows, beta, max_len, device, bs=8):
    """Per-pair DPO loss L_i and length difference (|y_w| - |y_l|)."""
    losses, len_diff = [], []
    for start in range(0, len(rows), bs):
        chunk = rows[start:start + bs]
        enc = [_encode(tok, r["prompt"], r[k], max_len, "left")
               for k in ("chosen", "rejected") for r in chunk]
        width = max(len(e["input_ids"]) for e in enc)
        pad = tok.pad_token_id
        ids, am, lm = [], [], []
        for e in enc:
            x = e["input_ids"]
            n = width - len(x)
            ids.append(x + [pad] * n)
            am.append([1] * len(x) + [0] * n)
            m = [0] * width
            for j in range(e["prompt_len"], len(x)):
                m[j] = 1
            lm.append(m)
        ids = torch.tensor(ids, device=device)
        am = torch.tensor(am, device=device)
        lm = torch.tensor(lm, device=device)
        with torch.no_grad():
            pol, counts, _, _ = _forward_logps(policy, ids, am, lm)
            rf, _, _, _ = _forward_logps(ref, ids, am, lm)
        pol_w, pol_l = pol.chunk(2)
        ref_w, ref_l = rf.chunk(2)
        n_w, n_l = counts.chunk(2)
        margin = (pol_w - ref_w) - (pol_l - ref_l)
        losses.append((-F.logsigmoid(beta * margin)).float().cpu().numpy())
        len_diff.append((n_w - n_l).float().cpu().numpy())
    return np.concatenate(losses), np.concatenate(len_diff)


def report(losses, len_diff, bs):
    """ESS and length-conditional mean weight, per beta'."""
    cl = len_diff > 0      # chosen longer  (the hard subset)
    rl = len_diff < 0      # rejected longer (the easy subset)
    print(f"\n  pairs={len(losses)}  chosen_longer={cl.sum()}  rejected_longer={rl.sum()}")
    print(f"  per-pair DPO loss L_i: mean={losses.mean():.4f} sd={losses.std():.4f} "
          f"max={losses.max():.3f}")
    r = float(np.corrcoef(losses, len_diff)[0, 1])
    print(f"  corr(L_i, |y_w|-|y_l|) = {r:+.4f}   "
          f"mean L[chosen_longer]={losses[cl].mean():.4f}  "
          f"mean L[rejected_longer]={losses[rl].mean():.4f}")

    print(f"\n  {'beta_prime':>10s} {'ESS/B':>7s} {'w[cl]/w_unif':>13s} "
          f"{'w[rl]/w_unif':>13s} {'ratio cl:rl':>12s}")
    print("  " + "-" * 60)
    out = []
    for bp in BETA_PRIME_GRID:
        w = np.zeros_like(losses)
        ess = []
        # Weights are normalised WITHIN each micro-batch, exactly as in training.
        for s in range(0, len(losses), bs):
            L = losses[s:s + bs]
            e = np.exp(-(L - L.min()) / bp)   # shift for numerical stability
            wi = e / e.sum()
            w[s:s + bs] = wi * len(L)          # 1.0 == uniform weight
            ess.append(1.0 / np.sum(wi ** 2) / len(L))
        w_cl, w_rl = w[cl].mean(), w[rl].mean()
        print(f"  {bp:10.2f} {np.mean(ess):7.3f} {w_cl:13.3f} {w_rl:13.3f} "
              f"{w_cl / w_rl:12.3f}")
        out.append((bp, np.mean(ess), w_cl, w_rl))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="checkpoints/round6/drdpo/final")
    p.add_argument("--ref", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--label", default="drdpo")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_len", type=int, default=1024)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.ref)
    ds = load_dataset("trl-lib/tldr-preference", split="train")
    idx = np.random.default_rng(42).choice(len(ds), size=args.n, replace=False)
    rows = [ds[int(i)] for i in idx]

    print(f"\n=== Dr.DPO KL-DRO weight reconstruction — policy={args.label} ===")
    policy = AutoModelForCausalLM.from_pretrained(
        args.policy, torch_dtype=torch.bfloat16).cuda().eval()
    ref = AutoModelForCausalLM.from_pretrained(
        args.ref, torch_dtype=torch.bfloat16).cuda().eval()
    device = next(policy.parameters()).device

    losses, len_diff = pair_losses(policy, ref, tok, rows, args.beta,
                                   args.max_len, device, args.batch_size)
    report(losses, len_diff, args.batch_size)

    os.makedirs("metrics", exist_ok=True)
    out = f"metrics/drdpo_weights_{args.label}.csv"
    np.savetxt(out, np.column_stack([losses, len_diff]), delimiter=",",
               header="dpo_loss,len_diff", comments="")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

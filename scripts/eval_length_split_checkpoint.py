"""Evaluate a saved checkpoint on the FULL decomposed test/validation set, split by
response-length direction (chosen-longer vs rejected-longer).

Reuses train.py's exact encoding (chat template, max_len 1024, left truncation) so
numbers are directly comparable to the in-training callback + round-3 §9.4b.

Usage (from project root):
  .venv/bin/python scripts/eval_length_split_checkpoint.py \
    --checkpoint checkpoints/dpo-v3/final --label "DPO v3"
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # import train from root
import train  # noqa: E402  (applies the tokenizer network-hang workaround)
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from datasets import load_dataset  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--label", default=None)
    p.add_argument("--eval_dataset", default="Bojian92/tldr_preference_decomposed")
    p.add_argument("--split", default="validation")
    p.add_argument("--max_eval_samples", type=int, default=None, help="None = full split (2000)")
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--prompt_truncation_side", default="left")
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--out_csv", default=None)
    # v2 eval set: a local parquet rather than a hub dataset, and (optionally) the extra
    # syntax variants rejected_{prime,double_prime}_v2 used to measure per-example variance.
    p.add_argument("--data_files", default=None,
                   help="local file(s); when set, --eval_dataset is the builder name, e.g. 'parquet'")
    p.add_argument("--response_keys", default=None,
                   help="comma-separated response columns to score (default: train.RESPONSE_KEYS)")
    args = p.parse_args()
    label = args.label or args.checkpoint

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16).cuda().eval()
    device = next(model.parameters()).device

    ds = (load_dataset(args.eval_dataset, data_files=args.data_files, split=args.split)
          if args.data_files else load_dataset(args.eval_dataset, split=args.split))
    response_keys = (tuple(k.strip() for k in args.response_keys.split(","))
                     if args.response_keys else train.RESPONSE_KEYS)
    missing = [k for k in ("prompt", *response_keys) if k not in ds.column_names]
    if missing:
        raise ValueError(f"eval set missing {missing}; has {ds.column_names}")
    if args.max_eval_samples:
        ds = ds.select(range(min(args.max_eval_samples, len(ds))))
    n = len(ds)
    print(f"[{label}] eval {args.eval_dataset}[{args.split}] n={n}")

    rows = []
    for start in range(0, n, args.eval_batch_size):
        end = min(start + args.eval_batch_size, n)
        batch = ds[start:end]
        encoded, owner = [], []
        for i in range(end - start):
            for key in response_keys:
                encoded.append(train._encode_example(tok, batch["prompt"][i], batch[key][i],
                                                     args.max_len, args.prompt_truncation_side))
                owner.append((start + i, key))
        sums, means, counts = train._batch_logps(model, tok, encoded, device)
        per = {}
        for (idx, key), s, m, c in zip(owner, sums, means, counts):
            d = per.setdefault(idx, {})
            d[f"{key}_logp"] = float(s); d[f"{key}_mean_logp"] = float(m); d[f"{key}_tokens"] = int(c)
        for idx, d in per.items():
            d["index"] = idx; rows.append(d)

    df = pd.DataFrame(rows).sort_values("index").reset_index(drop=True)
    out = args.out_csv or f"metrics/length_split_fulltest_{label.replace(' ', '_').replace('/', '_')}.csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out, index=False)

    def report(d, name):
        acc_sum = ((d.chosen_logp - d.rejected_logp) > 0).mean()
        acc_mean = ((d.chosen_mean_logp - d.rejected_mean_logp) > 0).mean()
        return (name, len(d), acc_sum, acc_mean,
                (d.chosen_logp - d.rejected_logp).mean(), (d.chosen_mean_logp - d.rejected_mean_logp).mean())

    cl = df[df.chosen_tokens > df.rejected_tokens]
    rl = df[df.chosen_tokens < df.rejected_tokens]
    print(f"\n{label} — length-direction split on FULL {args.split} (n={n})")
    print(f"{'subset':16s} {'n':>5s} {'acc(sum)':>9s} {'acc(mean)':>10s} {'sumC-R':>8s} {'meanC-R':>8s}")
    for d, name in [(cl, "chosen_longer"), (rl, "rejected_longer"), (df, "all")]:
        _, nn, a_s, a_m, s_cr, m_cr = report(d, name)
        print(f"{name:16s} {nn:>5d} {a_s:>9.3f} {a_m:>10.3f} {s_cr:>8.2f} {m_cr:>8.3f}")
    gap_sum = abs(((cl.chosen_logp - cl.rejected_logp) > 0).mean() - ((rl.chosen_logp - rl.rejected_logp) > 0).mean())
    gap_mean = abs(((cl.chosen_mean_logp - cl.rejected_mean_logp) > 0).mean() - ((rl.chosen_mean_logp - rl.rejected_mean_logp) > 0).mean())
    print(f"\nlength_acc_gap: sum-ranking={gap_sum:.3f}  mean-ranking={gap_mean:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

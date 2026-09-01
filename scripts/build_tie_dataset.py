"""Build the Tie-Training mixed dataset (2605.11134) for the TL;DR length setting.

Tie Training mixes STRICT preference pairs (chosen ≻ rejected) with equal-utility
"tie" pairs whose two responses have ~equal quality but differ in the spurious
feature (here: LENGTH), labeled with a random 50/50 winner. Standard DPO on this
mixture adds curvature only in the length direction, shrinking length reliance.

Tie source: the decomposed train split's (rejected_double_prime, rejected). By
construction rejected_double_prime is a CHOSEN-LENGTH paraphrase of rejected's
content (REPORT §9.5: corr(rej″, chosen len) 0.96 train), so the pair is
near-equal in content/quality but differs in length — a natural length tie.

Mixing: alpha = strict fraction = n_strict / (n_strict + n_tie). Tie budget
k = round(n_strict * (1-alpha)/alpha), sampled WITH replacement from the tie pool
(the decomposed train pool is small — ~3000 — so ties are upsampled; this is a
length-direction regularizer, so repetition is acceptable but limits tie
diversity; noted in REPORT §10).

Usage (from project root):
  .venv/bin/python scripts/build_tie_dataset.py --alpha 0.5 --out data/tie_mix_a0.5.parquet
"""
import argparse
import os
import random

from datasets import Dataset, load_dataset


def _text(v):
    if isinstance(v, str):
        return v
    if isinstance(v, list) and v and isinstance(v[-1], dict) and "content" in v[-1]:
        return v[-1]["content"]
    raise ValueError(f"Unrecognized field type: {type(v)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strict_dataset", default="trl-lib/tldr-preference")
    p.add_argument("--strict_split", default="train")
    p.add_argument("--tie_dataset", default="Bojian92/tldr_preference_decomposed")
    p.add_argument("--tie_split", default="train")
    p.add_argument("--alpha", type=float, default=0.5, help="strict fraction n_strict/(n_strict+n_tie)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/tie_mix.parquet")
    args = p.parse_args()
    rng = random.Random(args.seed)

    strict = load_dataset(args.strict_dataset, split=args.strict_split)
    n_strict = len(strict)
    rows = {"prompt": [], "chosen": [], "rejected": [], "is_tie": []}
    for ex in strict:
        rows["prompt"].append(_text(ex["prompt"]))
        rows["chosen"].append(_text(ex["chosen"]))
        rows["rejected"].append(_text(ex["rejected"]))
        rows["is_tie"].append(False)

    tie_src = load_dataset(args.tie_dataset, split=args.tie_split)
    pool = [( _text(e["prompt"]), _text(e["rejected_double_prime"]), _text(e["rejected"]) )
            for e in tie_src]
    k = round(n_strict * (1.0 - args.alpha) / args.alpha)  # tie budget
    print(f"strict={n_strict}  tie_pool={len(pool)}  tie_budget(k)={k}  alpha={args.alpha}")
    for _ in range(k):
        prompt, a, b = pool[rng.randrange(len(pool))]  # sample WITH replacement
        # random 50/50 winner label (the tie mechanism); symmetric loss is label-invariant
        if rng.random() < 0.5:
            w, l = a, b
        else:
            w, l = b, a
        rows["prompt"].append(prompt); rows["chosen"].append(w)
        rows["rejected"].append(l); rows["is_tie"].append(True)

    ds = Dataset.from_dict(rows).shuffle(seed=args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ds.to_parquet(args.out)
    n_tie = sum(rows["is_tie"])
    print(f"wrote {args.out}: total={len(ds)} strict={n_strict} tie={n_tie} "
          f"(effective alpha={n_strict/(n_strict+n_tie):.3f})")


if __name__ == "__main__":
    main()

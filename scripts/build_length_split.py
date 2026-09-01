"""Round 3 — split the FULL trl-lib/tldr-preference train set by response-length direction.

Measures each pair's chosen/rejected response length using the SAME token
definition as the decomposed eval (chat template applied, response portion only,
Qwen2.5 tokenizer, pre-truncation). Partitions into:

  chosen_longer   : chosen_tokens >  rejected_tokens
  rejected_longer : chosen_tokens <  rejected_tokens
  ties            : dropped

Then subsamples both directional subsets to an EQUAL N = min(len) with a fixed
seed, so the only difference between the two training arms is length direction
(not data volume). Writes the original-index lists to a JSON that train.py
consumes via --train_length_subset. The full (mixed) baseline stays untouched —
the existing results/core_v2_aggressive/simpo run is its control.

Usage:
    python build_length_split.py                       # default settings
    python build_length_split.py --seed 42 --out results/length_split/split_indices.json
"""

import argparse
import json
import os

from datasets import load_dataset
from transformers import AutoTokenizer

# Reuse train.py's network-hang workaround for the Qwen tokenizer load.
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def _skip_mistral_regex_probe(cls, tokenizer, *args, **kwargs):
    return tokenizer


PreTrainedTokenizerBase._patch_mistral_regex = classmethod(_skip_mistral_regex_probe)


def response_token_len(tok, prompt, response):
    """Response token count with the chat template applied (pre-truncation).

    Matches evaluate/train encoding: the prompt is a user turn with a generation
    prompt, the response is the assistant turn; we count only the response tokens.
    The fixed assistant wrapper is identical for chosen and rejected, so the
    length *direction* is exactly the direction of the scored response tokens.
    """
    pm = [{"role": "user", "content": prompt}]
    ptext = tok.apply_chat_template(pm, tokenize=False, add_generation_prompt=True)
    ftext = tok.apply_chat_template(
        pm + [{"role": "assistant", "content": response}], tokenize=False
    )
    rtext = ftext[len(ptext):]
    return len(tok(rtext, add_special_tokens=False)["input_ids"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--train_dataset", default="trl-lib/tldr-preference")
    p.add_argument("--train_split", default="train")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/length_split/split_indices.json")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset(args.train_dataset, split=args.train_split)
    n = len(ds)
    print(f"Loaded {args.train_dataset}[{args.train_split}] : {n} pairs")

    chosen_longer, rejected_longer, ties = [], [], []
    cho_tok_sum = rej_tok_sum = 0
    for i in range(n):
        ex = ds[i]
        c = response_token_len(tok, ex["prompt"], ex["chosen"])
        r = response_token_len(tok, ex["prompt"], ex["rejected"])
        cho_tok_sum += c
        rej_tok_sum += r
        if c > r:
            chosen_longer.append(i)
        elif c < r:
            rejected_longer.append(i)
        else:
            ties.append(i)
        if (i + 1) % 10000 == 0:
            print(f"  {i + 1}/{n}  chosen_longer={len(chosen_longer)} "
                  f"rejected_longer={len(rejected_longer)} ties={len(ties)}")

    nc, nr, nt = len(chosen_longer), len(rejected_longer), len(ties)
    print("\n=== length-direction distribution (full train) ===")
    print(f"  chosen_longer  : {nc} ({nc / n:.1%})")
    print(f"  rejected_longer: {nr} ({nr / n:.1%})")
    print(f"  ties           : {nt} ({nt / n:.1%})")
    print(f"  mean tokens chosen={cho_tok_sum / n:.2f} rejected={rej_tok_sum / n:.2f} "
          f"(mean diff {(cho_tok_sum - rej_tok_sum) / n:+.2f})")

    # Equal-N subsample (seeded) so the two arms differ only in length direction.
    import random

    rng = random.Random(args.seed)
    N = min(nc, nr)
    cl = sorted(rng.sample(chosen_longer, N))
    rl = sorted(rng.sample(rejected_longer, N))
    print(f"\nEqual-N per arm: N={N} (seed={args.seed})")

    payload = {
        "train_dataset": args.train_dataset,
        "train_split": args.train_split,
        "model_name_or_path": args.model_name_or_path,
        "seed": args.seed,
        "N_per_arm": N,
        "counts_full": {"chosen_longer": nc, "rejected_longer": nr, "ties": nt, "total": n},
        "mean_tokens": {"chosen": cho_tok_sum / n, "rejected": rej_tok_sum / n},
        "indices": {"chosen_longer": cl, "rejected_longer": rl},
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f)
    print(f"Wrote split indices -> {args.out}")


if __name__ == "__main__":
    main()

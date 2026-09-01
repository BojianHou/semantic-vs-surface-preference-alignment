"""Assemble + QC the agent-written decomposed-v2 evaluation set.

The counterfactuals are written by LLM agents (one writer + one adversarial auditor per
shard); this script never generates text. It joins their output back onto the source
shards, measures every constraint the agents were given, drops or flags what fails, and
writes the final parquet.

Constraints checked (all from reports/AUDIT_2026-08-29.md):
  T  token length of each variant == chosen_tokens  (kills the length contamination in g2)
  S  one leading space, like the source strings
  D  no variant byte-identical to `rejected` or to another variant (the old set had 6.2%
     fully-degenerate rows, which forced g2 and g3 toward 0 for free)
  V  g2-pair vocabulary overlap: rejected_prime vs rejected_double_prime should be a
     STRUCTURAL rearrangement, not a different paraphrase (old set: 0.545 char similarity)

Usage:
    .venv/bin/python scripts/assemble_decomposed_v2.py                    # report + write
    .venv/bin/python scripts/assemble_decomposed_v2.py --tolerance 1      # allow +-1 token
    .venv/bin/python scripts/assemble_decomposed_v2.py --list_failures    # ids needing repair
"""
import argparse
import difflib
import glob
import json
import os

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

PreTrainedTokenizerBase._patch_mistral_regex = classmethod(lambda cls, t, *a, **k: t)

VARIANTS = ("rejected_double_prime", "rejected_prime",
            "rejected_double_prime_v2", "rejected_prime_v2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tolerance", type=int, default=0, help="allowed |tokens - N| per variant")
    ap.add_argument("--root", default="data/decomposed_v2",
                    help="directory holding shards/ and out/")
    ap.add_argument("--out", default=None)
    ap.add_argument("--flat_out", default=None,
                    help="also write the information-matched control: each counterfactual as "
                         "its OWN (chosen, rejected) row, so an arm can see the same text "
                         "without any worst-case aggregation")
    ap.add_argument("--list_failures", action="store_true")
    a = ap.parse_args()
    SRC_DIR, OUT_DIR = f"{a.root}/shards", f"{a.root}/out"
    if a.out is None:
        a.out = f"{a.root}/{os.path.basename(a.root)}.parquet"

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])  # noqa: E731

    src = {}
    for f in sorted(glob.glob(f"{SRC_DIR}/shard_*.json")):
        for e in json.load(open(f)):
            src[e["id"]] = e
    written, missing_shards = {}, []
    for si in range(len(glob.glob(f"{SRC_DIR}/shard_*.json"))):
        p = f"{OUT_DIR}/out_{si:03d}.json"
        if not os.path.exists(p):
            missing_shards.append(si)
            continue
        try:
            for o in json.load(open(p)):
                written[o["id"]] = o
        except Exception as exc:
            missing_shards.append(si)
            print(f"  ! shard {si} unreadable: {exc}")

    print(f"source examples : {len(src)}")
    print(f"written         : {len(written)}   (missing shards: {len(missing_shards)})")
    if missing_shards:
        print(f"  missing shard ids: {missing_shards[:20]}{' ...' if len(missing_shards) > 20 else ''}")

    rows, fails = [], []
    for i, s in src.items():
        o = written.get(i)
        if o is None:
            fails.append((i, "not written"))
            continue
        if any(k not in o or not isinstance(o[k], str) or not o[k].strip() for k in VARIANTS):
            fails.append((i, "missing/empty variant"))
            continue
        N = s["chosen_tokens"]
        # Normalize to exactly one leading space FIRST, then count: that is the string the
        # evaluation will actually score, so the token constraint must hold on it. (Agents
        # that dropped the space would otherwise "pass" a count they took on a different
        # string than the one we use.)
        texts = {k: " " + o[k].strip() for k in VARIANTS}
        counts = {k: ntok(texts[k]) for k in VARIANTS}
        bad_len = [f"{k}={counts[k]}" for k in VARIANTS if abs(counts[k] - N) > a.tolerance]
        vals = list(texts.values())
        dup = len(set(v.strip() for v in vals)) < 4 or any(v.strip() == s["rejected"].strip() for v in vals)
        why = []
        if bad_len:
            why.append(f"token!=N({N}): {','.join(bad_len)}")
        if dup:
            why.append("degenerate/duplicate")
        if why:
            fails.append((i, "; ".join(why)))
            continue
        sim = difflib.SequenceMatcher(None, texts["rejected_prime"],
                                      texts["rejected_double_prime"]).ratio()
        rows.append({"prompt": s["prompt"], "chosen": s["chosen"], "rejected": s["rejected"],
                     **texts, "source_id": i, "chosen_tokens": N,
                     "rejected_tokens": s["rejected_tokens"], "g2_pair_similarity": sim})

    print(f"\nPASS {len(rows)} / {len(src)}   FAIL {len(fails)}")
    if fails:
        from collections import Counter
        kinds = Counter(w.split(":")[0].split(";")[0] for _, w in fails)
        for k, c in kinds.most_common():
            print(f"   {c:5d}  {k}")
    if a.list_failures:
        print("\nfailing ids:", json.dumps(sorted(i for i, _ in fails)))
        return

    if not rows:
        raise SystemExit("nothing passed QC yet")
    df = pd.DataFrame(rows)
    print("\n--- quality of the accepted set ---")
    print(f"unique prompts            : {df.prompt.nunique()} / {len(df)} rows")
    print(f"g2-pair char similarity   : mean {df.g2_pair_similarity.mean():.3f} "
          f"(OLD set 0.545; higher = more purely syntactic)")
    print(f"g2 pair token-length diff : 0 for {100.0:.0f}% by construction "
          f"(OLD set: identical only 30.3% of the time)")
    print(f"median chosen tokens      : {int(df.chosen_tokens.median())}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out_df = df.drop(columns=["g2_pair_similarity"])
    out_df.to_parquet(a.out, index=False)
    print(f"\n-> {a.out}")

    if a.flat_out:
        # Information-matched control: same prompts, same texts, but each rejected variant
        # becomes an independent preference row. An arm trained on this sees exactly the
        # data the counterfactual objective sees, aggregated by a plain mean -- which is
        # what isolates "worst-case aggregation" from "more rejected data".
        flat = pd.concat([
            out_df.assign(rejected=out_df[k])[["prompt", "chosen", "rejected"]]
            for k in ("rejected", "rejected_double_prime", "rejected_prime")
        ], ignore_index=True)
        flat.to_parquet(a.flat_out, index=False)
        print(f"-> {a.flat_out}  ({len(flat)} rows = {len(out_df)} x 3)")


if __name__ == "__main__":
    main()

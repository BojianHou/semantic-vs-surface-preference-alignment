"""Disk-state status of the agent-written v2 shards — the resume mechanism.

The counterfactual sets are written by LLM agents one shard-file at a time, so a killed
session (closed laptop, dropped connection, TaskStop) never loses completed shards. This
script re-derives what is still outstanding by actually QC-ing every shard on disk, and
prints the args to hand back to the workflow.

    CLEAN    every example passes QC                      -> skip
    DIRTY    file exists but some example fails QC        -> auditor repair pass
    MISSING  no file / unparseable / examples absent      -> writer + auditor

Usage:
    .venv/bin/python scripts/v2_shard_status.py                          # both roots, summary
    .venv/bin/python scripts/v2_shard_status.py --root data/decomposed_v2 --args
        -> {"root": ..., "missing": [...], "dirty": [...]}  paste into Workflow(args=...)
"""
import argparse
import glob
import json
import os

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

PreTrainedTokenizerBase._patch_mistral_regex = classmethod(lambda cls, t, *a, **k: t)

VARIANTS = ("rejected_double_prime", "rejected_prime",
            "rejected_double_prime_v2", "rejected_prime_v2")
ROOTS = ("data/decomposed_v2", "data/decomposed_v2_train")

_TOK = None


def ntok(s):
    global _TOK
    if _TOK is None:
        _TOK = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    return len(_TOK(s, add_special_tokens=False)["input_ids"])


def classify(root, tolerance=0):
    n_shards = len(glob.glob(f"{root}/shards/shard_*.json"))
    missing, dirty, clean = [], [], []
    for si in range(n_shards):
        src = {e["id"]: e for e in json.load(open(f"{root}/shards/shard_{si:03d}.json"))}
        op = f"{root}/out/out_{si:03d}.json"
        if not os.path.exists(op):
            missing.append(si)
            continue
        try:
            out = {o["id"]: o for o in json.load(open(op))}
        except Exception:
            missing.append(si)
            continue
        bad = 0
        for i, s in src.items():
            o = out.get(i)
            if not o or any(not isinstance(o.get(k), str) or not o.get(k, "").strip()
                            for k in VARIANTS):
                bad += 1
                continue
            N = s["chosen_tokens"]
            t = {k: " " + o[k].strip() for k in VARIANTS}
            if any(abs(ntok(t[k]) - N) > tolerance for k in VARIANTS):
                bad += 1
                continue
            vals = [x.strip() for x in t.values()]
            if len(set(vals)) < 4 or any(v == s["rejected"].strip() for v in vals):
                bad += 1
        (clean if bad == 0 else dirty).append(si)
    return {"clean": clean, "dirty": dirty, "missing": missing, "n_shards": n_shards}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--tolerance", type=int, default=0)
    ap.add_argument("--args", action="store_true", help="print Workflow args JSON for --root")
    a = ap.parse_args()

    roots = [a.root] if a.root else list(ROOTS)
    for root in roots:
        st = classify(root, a.tolerance)
        if a.args and a.root:
            print(json.dumps({"root": root, "missing": st["missing"], "dirty": st["dirty"]}))
            return
        done = len(st["clean"])
        print(f"=== {root} ===")
        print(f"  CLEAN   {done:3d} / {st['n_shards']}   ({done * 16} examples ready)")
        print(f"  DIRTY   {len(st['dirty']):3d}   {st['dirty'][:12]}")
        print(f"  MISSING {len(st['missing']):3d}")
        json.dump(st, open(f"{root}/todo.json", "w"))


if __name__ == "__main__":
    main()

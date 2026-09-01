"""Token counter for the decomposed-v2 rewriting agents.

The counterfactuals must be TOKEN-length matched under the exact tokenizer the
evaluation uses (Qwen2.5). Word counts are not a reliable proxy, so the writing
agents call this to check their own drafts and revise until the count matches.

This is a measuring instrument for the agents — it never writes or edits text.

Usage:
    # count one string
    .venv/bin/python scripts/count_tokens.py --text "  some candidate summary."

    # count many at once (JSON list on stdin) -> JSON list of counts
    echo '["first text","second text"]' | .venv/bin/python scripts/count_tokens.py

    # check a draft against a target, get the delta you still need to close
    .venv/bin/python scripts/count_tokens.py --text "..." --target 34
"""
import argparse
import json
import sys

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# Same offline workaround train.py uses (Qwen vocab > 100k triggers a network probe).
PreTrainedTokenizerBase._patch_mistral_regex = classmethod(lambda cls, t, *a, **k: t)

_TOK = None


def tok():
    global _TOK
    if _TOK is None:
        _TOK = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    return _TOK


def count(s):
    return len(tok()(s, add_special_tokens=False)["input_ids"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--target", type=int, default=None)
    a = p.parse_args()

    if a.text is not None:
        n = count(a.text)
        if a.target is None:
            print(n)
        else:
            d = n - a.target
            verdict = "MATCH" if d == 0 else ("TOO LONG by %d" % d if d > 0 else "TOO SHORT by %d" % -d)
            print(json.dumps({"tokens": n, "target": a.target, "delta": d, "verdict": verdict}))
        return

    data = json.load(sys.stdin)
    if isinstance(data, dict):
        print(json.dumps({k: count(v) for k, v in data.items()}))
    else:
        print(json.dumps([count(s) for s in data]))


if __name__ == "__main__":
    main()

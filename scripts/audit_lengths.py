"""Length audit of the decomposed TL;DR dataset: rejected'' vs rejected vs chosen.

Batched tokenization (fast). Tests the task-2 hypothesis premise that
rejected'' is longer than rejected.
"""
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

tok = AutoTokenizer.from_pretrained("checkpoints/simpo-strong/final")
ds = load_dataset("Bojian92/tldr_preference_decomposed")
print("splits:", {k: len(v) for k, v in ds.items()})


def lens(col):
    col = [x if isinstance(x, str) else ("" if x is None else str(x)) for x in col]
    enc = tok(col, add_special_tokens=False)["input_ids"]
    return np.array([len(x) for x in enc])


for split in ds:
    d = ds[split]
    c, r = lens(d["chosen"]), lens(d["rejected"])
    rp, rpp = lens(d["rejected_prime"]), lens(d["rejected_double_prime"])
    print(f"--- {split} (n={len(d)}) tokens ---")
    print(f"  mean: chosen={c.mean():.1f} rej'={rp.mean():.1f} rej''={rpp.mean():.1f} rejected={r.mean():.1f}")
    print(f"  rpp>r:   {100*(rpp>r).mean():.1f}%  ties {int((rpp==r).sum())}  mean(rpp-r)={(rpp-r).mean():+.2f}  median {int(np.median(rpp-r))}")
    print(f"  chosen>r: {100*(c>r).mean():.1f}%  mean(c-r)={(c-r).mean():+.2f}")
    print(f"  rpp vs chosen: mean(rpp-c)={(rpp-c).mean():+.2f}  rpp==c {100*(rpp==c).mean():.1f}%  corr={np.corrcoef(rpp,c)[0,1]:.2f}")

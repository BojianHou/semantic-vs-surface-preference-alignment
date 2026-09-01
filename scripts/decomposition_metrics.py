"""The corrected g1/g2/g3 metric (see reports/AUDIT_2026-08-29.md).

Every round before the 2026-08-29 audit scored the decomposition with a single number
per gap: the MEAN summed-logp difference. That statistic is wrong for the research
question in three ways, all fixed here.

1. LENGTH ARITHMETIC. A summed log-prob falls by ~2.5 nats per extra token no matter
   how good the text is, so g2/g3 can never reach 0 while the paired responses differ
   in token count. Each gap is split exactly:

       gap_i = (mean_a - mean_b) * nbar_i    <- CONTENT  (what we want)
             + (n_a - n_b)       * mbar_i    <- LENGTH   (pure arithmetic)

   All statistics below are computed on the CONTENT component.

2. CANCELLATION. mean(g) answers "is there a systematic bias?" It does NOT answer
   "does the model respond to this perturbation?", because large per-example effects
   of both signs average to ~0. We therefore report three distinct things:

       bias        mean(g)        systematic preference          -> want 0
       sensitivity mean(|g|)      per-example responsiveness     -> want 0
       consistency mean(g)/sd(g)  how directional the effect is  -> want ~0 for g2/g3,
                                                                     LARGE for g1

3. CLUSTERING. The legacy eval set has only 195 unique prompts across 2000 rows, so
   row-level error bars are ~2x too narrow. `bootstrap_ci` resamples PROMPTS.

The headline robustness number is `pct_surface_beats_semantics`: the share of examples
where a meaning-preserving syntax (or length) rewrite moves the score MORE than the real
semantic difference does. That is the quantity the research question actually asks about.
"""
import numpy as np
import pandas as pd

PAIRS = {
    "g1": ("chosen", "rejected_prime"),                     # semantics -> want LARGE
    "g2": ("rejected_prime", "rejected_double_prime"),      # syntax    -> want 0
    "g3": ("rejected_double_prime", "rejected"),            # length    -> want 0
}


def components(df, pairs=PAIRS):
    """Per-example CONTENT and LENGTH components of each gap.

    Requires columns <obj>_logp, <obj>_mean_logp, <obj>_tokens for each object.
    Returns {gap: {"content": [...], "length": [...], "total": [...]}}.
    """
    out = {}
    for g, (a, b) in pairs.items():
        na = df[f"{a}_tokens"].to_numpy(float)
        nb = df[f"{b}_tokens"].to_numpy(float)
        ma = df[f"{a}_mean_logp"].to_numpy(float)
        mb = df[f"{b}_mean_logp"].to_numpy(float)
        nbar, mbar = (na + nb) / 2.0, (ma + mb) / 2.0
        content, length = (ma - mb) * nbar, (na - nb) * mbar
        out[g] = {"content": content, "length": length, "total": content + length}
    return out


def bootstrap_ci(values, clusters=None, nboot=4000, seed=0, stat=np.mean):
    """95% CI for `stat`, resampling CLUSTERS (prompts) when given, else rows."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, float)
    if clusters is None:
        idx = np.arange(len(values))
        draws = [stat(values[rng.integers(0, len(idx), len(idx))]) for _ in range(nboot)]
    else:
        clusters = np.asarray(clusters)
        by = {c: np.where(clusters == c)[0] for c in np.unique(clusters)}
        keys = np.array(list(by))
        draws = []
        for _ in range(nboot):
            pick = rng.choice(keys, len(keys), replace=True)
            draws.append(stat(values[np.concatenate([by[k] for k in pick])]))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def scorecard(df, clusters=None, ci=True, seed=0):
    """The corrected per-method scorecard. `df` is one eval CSV; `clusters` = prompt ids."""
    comp = components(df)
    g1, g2, g3 = (comp[g]["content"] for g in ("g1", "g2", "g3"))
    row = {
        "n": len(df),
        "n_clusters": int(len(np.unique(clusters))) if clusters is not None else len(df),
        "acc": float((df.chosen_logp > df.rejected_logp).mean()),
    }
    for name, v in (("g1", g1), ("g2", g2), ("g3", g3)):
        row[f"{name}_bias"] = float(v.mean())
        row[f"{name}_sens"] = float(np.abs(v).mean())
        row[f"{name}_consistency"] = float(v.mean() / v.std()) if v.std() > 0 else np.nan
        row[f"{name}_length_term"] = float(comp[name]["length"].mean())
        if ci:
            lo, hi = bootstrap_ci(v, clusters, seed=seed)
            row[f"{name}_bias_lo"], row[f"{name}_bias_hi"] = lo, hi
    # the headline robustness numbers
    row["pct_syntax_beats_semantics"] = float(100 * np.mean(np.abs(g2) > np.abs(g1)))
    row["pct_length_beats_semantics"] = float(100 * np.mean(np.abs(g3) > np.abs(g1)))
    row["pct_g1_positive"] = float(100 * np.mean(g1 > 0))
    return row


def paired_delta(df_a, df_b, clusters=None, seed=0):
    """Paired change from model B (reference, e.g. untrained) to model A, same examples.

    Returns per-gap dicts with the mean change and its clustered CI, for both the bias
    and the sensitivity. Paired comparison is far more powerful than comparing two
    independent scorecards, because example-level variance cancels.
    """
    ca, cb = components(df_a), components(df_b)
    out = {}
    for g in PAIRS:
        va, vb = ca[g]["content"], cb[g]["content"]
        d_bias, d_sens = va - vb, np.abs(va) - np.abs(vb)
        lo1, hi1 = bootstrap_ci(d_bias, clusters, seed=seed)
        lo2, hi2 = bootstrap_ci(d_sens, clusters, seed=seed)
        out[g] = {
            "d_bias": float(d_bias.mean()), "d_bias_ci": (lo1, hi1),
            "d_sens": float(d_sens.mean()), "d_sens_ci": (lo2, hi2),
            "bias_sig": lo1 > 0 or hi1 < 0,
            "sens_sig": lo2 > 0 or hi2 < 0,
        }
    return out


def noise_floor(df):
    """Irreducible wording noise — measurable only on the v2 eval set.

    v2 carries a SECOND independent variant built to the same syntax target and the same
    token length (`rejected_prime_v2`, `rejected_double_prime_v2`). The gap between two
    such twins is not a syntax effect at all — it is just "different words, different
    probability". That is the floor every other gap must be judged against:

        g2 / noise ~ 1   ->  the model is NOT syntax sensitive; the apparent g2 is wording
        g2 / noise >> 1  ->  the model genuinely responds to syntactic structure

    Without it the per-example |g2| is uninterpretable, because any paraphrase pair moves
    the score even when nothing systematic is being measured. On a 285-example preview both
    the untrained model and Dr.DPO sat at g1 ~ 3.9x noise and g2 ~ 2.0x noise — i.e. the
    syntax effect is real, and training scaled every gap up without changing the ratio.
    """
    need = ("rejected_prime_v2", "rejected_double_prime_v2")
    if any(f"{k}_logp" not in df.columns for k in need):
        return None
    twins = np.concatenate([
        components(df, {"n": ("rejected_prime", "rejected_prime_v2")})["n"]["content"],
        components(df, {"n": ("rejected_double_prime", "rejected_double_prime_v2")})["n"]["content"],
    ])
    comp = components(df)
    g1, g2, g3 = (comp[g]["content"] for g in ("g1", "g2", "g3"))
    floor = float(np.abs(twins).mean())
    return {
        "noise_floor": floor,
        "noise_bias": float(twins.mean()),
        "g1_over_noise": float(np.abs(g1).mean() / floor),
        "g2_over_noise": float(np.abs(g2).mean() / floor),
        "g3_over_noise": float(np.abs(g3).mean() / floor),
        # the headline: how much more the model responds to meaning than to surface form
        "semantics_over_syntax": float(np.abs(g1).mean() / np.abs(g2).mean()),
    }


def prompt_clusters(eval_dataset="Bojian92/tldr_preference_decomposed", split="validation",
                    index=None):
    """Cluster id per eval row = which unique prompt it came from."""
    from datasets import load_dataset
    d = load_dataset(eval_dataset, split=split)
    _, cid = np.unique(np.array(d["prompt"]), return_inverse=True)
    return cid if index is None else cid[np.asarray(index)]

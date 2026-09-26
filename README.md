# Syntactic Shadowing: Why Length Penalties Do Not Fix Preference Optimization

Code, benchmark and per-example scores for ICLR 2027 submission. This repository is anonymized for double-blind review.

## Summary

We decompose the alignment gap `G = log π(y_w) − log π(y_l)` of a preference pair into a **length**, a residual **syntactic** and a residual **semantic** gap. The rejected summary is rewritten along a meaning-preserving path that changes one property per step (the chosen summary stays fixed), and every gap is read against the score discrepancy between two rewrites produced under the same specification (*twins*).

On 25 models of Qwen2.5-1.5B-Instruct, length penalties move length sensitivity `L` in both directions but none raises directional semantic correctness `C` above plain DPO; no method at its usual strength moves syntactic sensitivity `S` (about 1.5× the twin scale, trained or not); training on surface-matched pairs as ordinary pairs raises `C` by 4–6 points.

## Benchmark

| file | contents |
|---|---|
| `data/decomposed_v2/decomposed_v2_validation.parquet` | 1,979 evaluation families, one per unique prompt |
| `data/decomposed_v2_train/decomposed_v2_train.parquet` | 1,998 disjoint training prompts with the same rewrites |
| `data/decomposed_v2_train/decomposed_v2_train_flat.parquet` | the training set as 5,994 ordinary pairs |
| `metrics/*_v2.csv` | per-family scores of the 37 checkpoints behind the main table |

Columns: `chosen` (y_w), `rejected` (y_l), `rejected_double_prime` (y″₁: meaning of y_l, exactly y_w's token count), `rejected_prime` (y′₁: additionally y_w's sentence pattern), and their twins `rejected_double_prime_v2`, `rejected_prime_v2`. Source pairs: [`trl-lib/tldr-preference`](https://huggingface.co/datasets/trl-lib/tldr-preference). Model checkpoints are not included.

## Reproduce the main table (CPU, no checkpoints)

```python
import pandas as pd

def metrics(csv, base="metrics/length_split_fulltest_BASE_untrained_v2.csv"):
    d, b = pd.read_csv(csv).sort_values("index"), pd.read_csv(base).sort_values("index")
    s = lambda k: d[f"{k}_logp"]
    g1 = s("chosen") - s("rejected_prime")                                    # semantic
    g2 = s("rejected_prime") - s("rejected_double_prime")                     # syntactic
    g3 = (d.rejected_double_prime_tokens + d.rejected_tokens) / 2 * (
        d.rejected_double_prime_mean_logp - d.rejected_mean_logp)             # length (content part)
    nu = 0.5 * ((s("rejected_prime") - s("rejected_prime_v2")).abs()
                + (s("rejected_double_prime") - s("rejected_double_prime_v2")).abs()).mean()
    return pd.Series({"acc": (s("chosen") > s("rejected")).mean(), "M": g1.abs().mean() / nu,
                      "S": g2.abs().mean() / nu, "L": g3.abs().mean() / nu, "C": (g1 > 0).mean(),
                      "g1_bar": g1.mean() / nu, "disp": s("chosen").mean() - b.chosen_logp.mean()})

print(metrics("metrics/length_split_fulltest_DPO_v3_v2.csv").round(3))
# acc 0.580  M 3.535  S 1.508  L 1.551  C 0.790  g1_bar 2.726  disp -14.655
```

Files are named by launcher label (e.g. `round6_rdpo_fulltest_v2.csv` for R-DPO); counterfactual-data controls (`round9_<arm>_s{1,2,3}_v2.csv`) are reported as three-seed means.

## Score and train

```bash
pip install -r requirements.txt

# score a checkpoint on the benchmark
python scripts/eval_length_split_checkpoint.py --checkpoint <path-or-hub-id> \
  --eval_dataset parquet --data_files data/decomposed_v2/decomposed_v2_validation.parquet --split train \
  --response_keys chosen,rejected_prime,rejected_double_prime,rejected,rejected_prime_v2,rejected_double_prime_v2 \
  --out_csv metrics/mymodel_v2.csv

# train (base model Qwen/Qwen2.5-1.5B-Instruct)
bash scripts/run_round6.sh <label>        # SOTA methods: orpo lddpo rdpo_robust rdpo drdpo sampo lmpo tie
bash scripts/run_round7.sh <label>        # Dr. DPO variants: drdpo_bp025 ... drdpo_i4 drdpo_i4b
bash scripts/run_round9.sh <arm> <seed>   # controls: base_drdpo cf_dpo cf_drdpo cf_uniform flat_dpo flat_sm
```

## Citation

```bibtex
@inproceedings{anonymous2027syntactic,
  title     = {Syntactic Shadowing: Why Length Penalties Do Not Fix Preference Optimization},
  author    = {Anonymous},
  booktitle = {Submitted to the International Conference on Learning Representations},
  year      = {2027},
  note      = {Under review}
}
```

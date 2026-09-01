"""length_pref — length-robust / spurious-correlation preference losses not in TRL.

A single self-contained pairwise (chosen, rejected) trainer, mirroring the
`sipo.py` pattern (custom collator + `transformers.Trainer` subclass + a shared
differentiable forward). Implements, selected by ``lp_cfg["method"]``:

    dpo    reference DPO (sigmoid, summed logps)  -- VALIDATION MODE ONLY:
           smoke-tested to match trl.DPOTrainer's loss on identical batches so the
           custom variants below inherit that faithfulness (REPORT §10 authenticity).
    rdpo   R-DPO (2403.19159, "Disentangling Length from Quality"): subtract an
           explicit length penalty alpha*(|y_w|-|y_l|) from the reward margin.
    drdpo  Dr.DPO (2407.07880, Distributionally-Robust DPO): replace the batch mean
           E[L_i] with the KL-DRO dual -beta' log E[exp(-L_i/beta')], which
           DOWN-weights high-loss (noisy / length-exploiting) pairs.
    sampo  SamPO (2406.10957, "Down-Sampled KL"): down-sample each response's
           per-token log-ratios to T_m=min(|y_w|,|y_l|) positions before summing,
           removing the length term from the implicit-reward difference.
    lmpo   LMPO (2502.14643): reference-free, length-averaged SimPO reward with an
           extra length-scaled probability margin (z-normalized, detached).
    tie    Tie Training (2605.11134): DPO plus an equal-utility "tie" regularizer
           on length-matched pairs to break the length spurious correlation.

Reward scale by method (matches each paper): dpo/rdpo/drdpo/sampo/tie use SUMMED
token log-probs with a small beta (~0.1); lmpo uses length-AVERAGED (mean) logps
with a large beta (~2.5). rdpo/drdpo/sampo/tie/dpo are reference-based (a frozen
clone of the initial policy is the reference, exactly as TRL's DPO with
ref_model=None); lmpo is reference-free.

===========================================================================
Support-Augmented KL-Dr.DPO  (opt-in, ``use_counterfactual_kl_dro``)
===========================================================================
OFF BY DEFAULT — everything above is untouched when the flag is false.

When on, each preference example carries two extra *rejected* counterfactuals
that keep the rejected semantics but control surface form:

    y_w    = chosen
    y_l^0  = rejected                  (original)
    y_l^L  = rejected_double_prime     (length matched to chosen)
    y_l^LS = rejected_prime            (length AND syntax matched to chosen)

All three carry the SAME preference label (chosen > each). With the usual DPO
score s(x,y) = log pi(y|x) - log pi_ref(y|x) and margin
m^k = beta*(s(x,y_w) - s(x,y_l^k)), the per-pair loss is the same
loss-MINIMIZATION quantity this file already uses everywhere,
ell^k = -log sigmoid(m^k) = softplus(-m^k) >= 0.

The three pairs are aggregated per example by a KL-REGULARIZED (KL-penalized)
DRO inner problem with a FIXED temperature tau_v — NOT a constrained DRO with
an explicit KL radius, and no dual variable is optimized:

    p       = (1 - eps_L - eps_LS,  eps_L,  eps_LS)          nominal support
    A_i     = tau_v * logsumexp_k( log p_k + ell_i^k / tau_v )
    q_i^k   = softmax_k( log p_k + ell_i^k / tau_v )         (monitoring only)

Note the PLUS sign on ell/tau_v: this is the worst-case (max-side) smoothing,
so A_i -> max_k ell_i^k as tau_v -> 0 and A_i -> sum_k p_k ell_i^k as
tau_v -> inf. A_i is used directly as the training loss (autograd differentiates
the logsumexp); q is only detached for logging.

A_i then REPLACES the per-example loss ell_i fed to the existing outer
aggregation, which is used verbatim:

    (ell^0, ell^L, ell^LS) -> A_i -> _outer_aggregate({A_i})   # unchanged

Two levels, two different sign conventions — deliberately:
  * INNER (new, ``counterfactual_kl_aggregate``): +ell/tau_v, so the HARDEST
    transformation of one example dominates.
  * OUTER (existing Dr.DPO ``_dro_dual``): -L/beta', i.e. w_i = softmax(-L_i/b'),
    so high-loss EXAMPLES are DOWN-weighted. That is the published Dr.DPO
    (2407.07880) noise-robust dual and is left exactly as it was.
The three counterfactuals are never flattened into the outer batch, which would
triple each prompt's mass and mix within- with between-example robustness.
"""

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import Trainer

# Reference-based methods need a frozen copy of the initial policy.
_REF_BASED = {"dpo", "rdpo", "drdpo", "sampo", "tie"}

# ---- Support-Augmented KL-Dr.DPO --------------------------------------------- #
# Dataset columns, in the fixed row-block order the collator emits and
# compute_loss chunks: chosen first, then the three rejected variants.
CF_KEYS = ("chosen", "rejected", "rejected_double_prime", "rejected_prime")
# Support-point names, aligned with the probability vector p = (p_0, p_L, p_LS)
# and with the cf/* metric suffixes.
CF_BRANCHES = ("original", "length", "length_syntax")
# Only the reference-based, sigmoid-margin pair losses have a well-defined
# per-pair ell^k to aggregate. sampo (per-token down-sampling), lmpo
# (reference-free) and tie (mixed tie labels) are out of scope for v1.
CF_SUPPORTED_METHODS = ("dpo", "rdpo", "drdpo")


def counterfactual_probs(eps_length, eps_length_syntax):
    """Validate the support and return the nominal probabilities (p_0, p_L, p_LS).

    Raises ValueError on any invalid configuration; every enabled support point
    must carry STRICTLY positive nominal mass (a zero would make log p = -inf and
    silently drop that counterfactual from the orbit).
    """
    eps_L, eps_LS = float(eps_length), float(eps_length_syntax)
    if not (eps_L > 0.0):
        raise ValueError(f"cf_epsilon_length must be > 0, got {eps_L}")
    if not (eps_LS > 0.0):
        raise ValueError(f"cf_epsilon_length_syntax must be > 0, got {eps_LS}")
    if not (eps_L + eps_LS < 1.0):
        raise ValueError(
            "cf_epsilon_length + cf_epsilon_length_syntax must be < 1, got "
            f"{eps_L} + {eps_LS} = {eps_L + eps_LS}"
        )
    return (1.0 - eps_L - eps_LS, eps_L, eps_LS)


def validate_counterfactual_config(eps_length, eps_length_syntax, kl_temperature):
    """Fail fast on an invalid (eps_L, eps_LS, tau_v); return (probs, tau_v)."""
    probs = counterfactual_probs(eps_length, eps_length_syntax)
    tau = float(kl_temperature)
    if not (tau > 0.0) or not math.isfinite(tau):
        raise ValueError(f"cf_kl_temperature (tau_v) must be finite and > 0, got {tau}")
    return probs, tau


def counterfactual_kl_aggregate(losses, probs, tau):
    """Local KL-regularized DRO over one example's counterfactual orbit.

        A_i = tau * logsumexp_k( log p_k + ell_i^k / tau )
        q_i = softmax_k( log p_k + ell_i^k / tau )          (detached, logging only)

    ``losses``  [B, K] per-pair losses in loss-MINIMIZATION convention (larger =
                worse); K must match ``probs``.
    ``probs``   nominal support probabilities, summing to 1.
    ``tau``     counterfactual KL temperature tau_v > 0.

    Returns ``(A [B], q [B, K])``. The logsumexp is evaluated in float64 and A is
    cast back to the caller's dtype, so an fp16/bf16 batch can neither overflow
    exp(ell/tau) nor lose the limit behaviour. float64 (not merely float32) is
    needed for LARGE tau_v: there logits = log p_k + ell/tau_v collapse onto
    log p_k and A = tau_v * logsumexp(...) multiplies the rounding error by tau_v,
    so at tau_v=1e6 float32 misses the nominal-mean limit by ~0.08 while float64 is
    exact to ~1e-10. The tensor is only [B, K], so the cost is negligible.
    Gradients flow through A only — q is detached.
    """
    if losses.ndim != 2:
        raise ValueError(f"counterfactual losses must be [B, K], got {tuple(losses.shape)}")
    if losses.shape[-1] != len(probs):
        raise ValueError(
            f"got {losses.shape[-1]} counterfactual losses but {len(probs)} probabilities"
        )
    if not (float(tau) > 0.0):
        raise ValueError(f"tau_v must be > 0, got {tau}")
    # Everything below runs in float64; A is cast back to the training dtype.
    work = torch.float64
    ell = losses.to(work)
    log_p = torch.log(torch.as_tensor(probs, dtype=work, device=ell.device))
    logits = log_p + ell / tau                       # log p_k + ell^k / tau_v
    A = tau * torch.logsumexp(logits, dim=-1)        # [B]
    q = torch.softmax(logits.detach(), dim=-1)       # [B, K], monitoring only
    return A.to(losses.dtype), q


def _encode(tokenizer, prompt, response, max_len, prompt_truncation_side):
    """Tokenize (prompt, response) with the chat template; return ids + prompt_len.

    Self-contained copy of train._encode_example (avoids a circular import), so the
    encoding matches the in-training decomposed-eval callback exactly.
    """
    pm = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(pm, tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(
        pm + [{"role": "assistant", "content": response}], tokenize=False
    )
    if not full_text.startswith(prompt_text):
        raise ValueError("Chat template full text does not start with the generation prompt.")
    response_text = full_text[len(prompt_text):]

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
    if not response_ids:
        raise ValueError("Empty response after tokenization.")
    if len(prompt_ids) + len(response_ids) > max_len:
        max_prompt_len = max_len - len(response_ids)
        if max_prompt_len > 0:
            prompt_ids = (prompt_ids[-max_prompt_len:] if prompt_truncation_side == "left"
                          else prompt_ids[:max_prompt_len])
        else:
            prompt_ids = prompt_ids[:1]
            response_ids = response_ids[: max_len - 1]
    return {"input_ids": prompt_ids + response_ids, "prompt_len": len(prompt_ids)}


@dataclass
class LengthPrefDataCollator:
    """Raw {prompt, chosen, rejected} -> one role-blocked [2B, L] padded batch.

    Rows are [chosen(B), rejected(B)]; compute_loss splits with .chunk(2). Accepts
    either conversational (list-of-messages, as train._to_conversational produces)
    or plain-string chosen/rejected.

    With ``counterfactual=True`` (Support-Augmented KL-Dr.DPO) the batch instead
    carries the four ``CF_KEYS`` blocks — [chosen(B), rejected(B),
    rejected_double_prime(B), rejected_prime(B)], i.e. [4B, L] — and compute_loss
    splits with .chunk(4). Same tokenizer, truncation, padding and response-only
    masking for every block; chosen is encoded ONCE and reused by all three pairs.
    """

    tokenizer: object
    max_len: int = 1024
    max_prompt_len: int = 768  # unused directly; _encode handles truncation via max_len
    prompt_truncation_side: str = "left"
    counterfactual: bool = False

    @staticmethod
    def _text(v):
        """Accept a plain string or a conversational [{'role','content'}] list."""
        if isinstance(v, str):
            return v
        if isinstance(v, list) and v and isinstance(v[-1], dict) and "content" in v[-1]:
            return v[-1]["content"]
        raise ValueError(f"Unrecognized response field type: {type(v)}")

    def _keys(self, examples):
        if not self.counterfactual:
            return ("chosen", "rejected")
        missing = sorted({k for k in CF_KEYS for ex in examples if k not in ex})
        if missing:
            raise ValueError(
                "use_counterfactual_kl_dro=True requires the counterfactual columns "
                f"{list(CF_KEYS)}; the batch is missing {missing}. Train on a dataset "
                "with rejected_double_prime / rejected_prime (e.g. "
                "Bojian92/tldr_preference_decomposed)."
            )
        return CF_KEYS

    def __call__(self, examples):
        pad_id = self.tokenizer.pad_token_id
        encoded = []  # role-major: all chosen, then all rejected, ...
        for key in self._keys(examples):
            for ex in examples:
                encoded.append(_encode(self.tokenizer, self._text(ex["prompt"]),
                                       self._text(ex[key]), self.max_len,
                                       self.prompt_truncation_side))
        max_len = max(len(e["input_ids"]) for e in encoded)
        input_ids, attention_mask, loss_mask = [], [], []
        for e in encoded:
            ids = e["input_ids"]
            pad = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)
            lm = [0] * len(ids) + [0] * pad
            for j in range(e["prompt_len"], len(ids)):
                lm[j] = 1  # score response tokens only
            loss_mask.append(lm)
        out = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.long),
        }
        # Tie Training: per-example flag marking equal-utility (length-tie) rows.
        out["is_tie"] = torch.tensor([bool(ex.get("is_tie", False)) for ex in examples],
                                     dtype=torch.bool)
        return out


def _forward_logps(model, input_ids, attention_mask, loss_mask):
    """One forward pass -> (logp_sum [N], counts [N], per_token [N,L-1], lm [N,L-1]).

    per_token[:, k] = log p(token_{k+1} | tokens_<=k), zeroed outside the response.
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = logits[:, :-1, :].float()  # fp32 log-softmax for stability
    labels = input_ids[:, 1:].clone()
    lm = loss_mask[:, 1:].bool()
    labels[~lm] = 0
    per_token = torch.gather(logits.log_softmax(-1), 2, labels.unsqueeze(2)).squeeze(2)
    per_token = per_token * lm
    counts = lm.sum(-1).clamp(min=1)
    return per_token.sum(-1), counts, per_token, lm


class LengthPrefTrainer(Trainer):
    """HF Trainer implementing the length-robust losses (see module docstring)."""

    def __init__(self, *args, lp_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lp_cfg = lp_cfg
        # Support-Augmented KL-Dr.DPO: validated once, before the first step.
        self.cf_on = bool(lp_cfg.get("use_counterfactual_kl_dro", False))
        self.cf_probs, self.cf_tau = (None, None)
        self._cf_metric_sums, self._cf_metric_n = {}, 0
        if self.cf_on:
            if lp_cfg["method"] not in CF_SUPPORTED_METHODS:
                raise ValueError(
                    f"use_counterfactual_kl_dro is only supported for methods "
                    f"{list(CF_SUPPORTED_METHODS)}, got {lp_cfg['method']!r}"
                )
            self.cf_probs, self.cf_tau = validate_counterfactual_config(
                lp_cfg["cf_epsilon_length"],
                lp_cfg["cf_epsilon_length_syntax"],
                lp_cfg["cf_kl_temperature"],
            )
            print(f"[cf-kl-dro] p=(p_0={self.cf_probs[0]:.4f}, p_L={self.cf_probs[1]:.4f}, "
                  f"p_LS={self.cf_probs[2]:.4f})  tau_v={self.cf_tau}  "
                  f"outer={lp_cfg['method']}")
        self.ref_model = None
        if lp_cfg["method"] in _REF_BASED:
            # Frozen clone of the INITIAL policy = the reference (matches TRL DPO
            # ref_model=None). Snapshotted here, before any optimizer step.
            self.ref_model = copy.deepcopy(self.model).eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)
        self._ref_on_device = False
        self._loss_buf = None  # (E) FIFO of recent detached exp(-L/b') terms

    def _ref_logps(self, input_ids, attention_mask, loss_mask):
        """Summed + per-token reference log-probs (no grad)."""
        if not self._ref_on_device:
            self.ref_model.to(input_ids.device)
            self._ref_on_device = True
        with torch.no_grad():
            ref_sum, _, ref_pt, _ = _forward_logps(self.ref_model, input_ids, attention_mask, loss_mask)
        return ref_sum, ref_pt

    @staticmethod
    def _dro_dual(per_ex, bp):
        """KL-DRO dual  -b' log E[exp(-L_i/b')]  over a set of per-pair losses.

        Its gradient is the weighted sum  sum_i w_i dL_i,  w_i = softmax(-L_i/b'),
        so LOW-loss pairs are up-weighted and high-loss pairs down-weighted.
        """
        n = per_ex.shape[0]
        log_mean_exp = torch.logsumexp(-per_ex / bp, dim=0) - torch.log(
            torch.tensor(float(n), device=per_ex.device, dtype=per_ex.dtype))
        return -bp * log_mean_exp

    def _dro_dual_buffered(self, per_ex, bp, cap):
        """(E) DRO weights normalised over a FIFO buffer of recent losses, not the batch.

        The per-micro-batch dual renormalises inside every group of 8, so its weights
        sum to 1 whether that batch is globally easy or globally hard — the "robust"
        comparison is only ever local. Here the normaliser spans the last ``cap``
        losses, so a globally-easy pair keeps w>1 and a globally-hard one w<1.

        Uses the detached-weight form  mean_i(w_i * L_i)  with
        w_i = exp(-L_i/b') / mean_j exp(-L_j/b'). This reproduces the dual's gradient
        direction while keeping E[w] ~ 1, so the effective learning rate stays
        comparable to the other arms (a raw buffered logsumexp would shrink the
        current batch's gradient mass to ~B/cap and silently act as an LR cut).
        """
        with torch.no_grad():
            cur = torch.exp(-per_ex / bp)
            pool = torch.cat([cur] + ([self._loss_buf.to(cur.device)] if self._loss_buf is not None else []))
            w = cur / pool.mean().clamp(min=1e-8)
            keep = torch.cat([cur, self._loss_buf.to(cur.device)]) if self._loss_buf is not None else cur
            self._loss_buf = keep[-cap:].detach().cpu()
        return (w * per_ex).mean()

    def _outer_aggregate(self, method, per_ex, chosen_longer):
        """Batch-level aggregation of per-EXAMPLE losses (unchanged semantics).

        Extracted verbatim from the original inline drdpo branch so the
        Support-Augmented path can reuse it byte-for-byte: only the *input*
        changes (A_i instead of ell_i), never the formula. Non-drdpo methods
        aggregate with the plain batch mean, as before.
        """
        if method != "drdpo":
            return per_ex.mean()
        c = self.lp_cfg
        bp = c["drdpo_beta_prime"]
        if c.get("drdpo_stratify_length"):
            # (B) Round-7: apply the KL-DRO dual WITHIN each length environment and
            # weight the two environments equally. Rationale (round-7 §A): the plain
            # dual's weighting is empirically length-BLIND (corr(L_i, |y_w|-|y_l|)
            # = -0.04), so it cannot trim length-exploiting pairs as §9.9 conjectured.
            # Stratifying makes the robustness length-aware and simultaneously
            # de-skews the 55.7/40.7 train imbalance. Envs match invariance_train.py:
            # chosen_longer = n_w > n_l (ties fall in the complement).
            parts = [per_ex[m] for m in (chosen_longer, ~chosen_longer) if m.any()]
            return sum(self._dro_dual(p, bp) for p in parts) / len(parts)
        if c.get("drdpo_buffer", 0) > 0:
            # (E) Global normalizer: judge each pair against the recent global loss
            # distribution instead of only the 7 others in its micro-batch.
            return self._dro_dual_buffered(per_ex, bp, int(c["drdpo_buffer"]))
        return self._dro_dual(per_ex, bp)

    def _pair_loss(self, beta, margin, len_w, len_l):
        """Per-pair DPO loss ell = -log sigmoid(beta*margin [- alpha*(len_w-len_l)]).

        The optional explicit length penalty is R-DPO's, applied inside the sigmoid
        for method=rdpo and for drdpo with ``drdpo_length_penalty`` — identical to
        the original inline expressions.
        """
        c = self.lp_cfg
        logits = beta * margin
        if c["method"] == "rdpo" or (c["method"] == "drdpo" and c.get("drdpo_length_penalty")):
            logits = logits - c["rdpo_alpha"] * (len_w - len_l)
        return -F.logsigmoid(logits), logits

    def _counterfactual_loss(self, beta, pol, ref, counts):
        """Support-Augmented KL-Dr.DPO inner problem for a [4B, ...] batch.

        ``pol``/``ref``/``counts`` are the summed policy log-probs, reference
        log-probs and response-token counts of the four CF_KEYS blocks. The chosen
        block is sliced ONCE and reused by all three margins (no extra forward).

        Returns ``(A [B], margins [B,3], ells [B,3], q [B,3], n_w, n_l0)``.
        """
        pol_w, *pol_ks = pol.chunk(4)
        ref_w, *ref_ks = ref.chunk(4)
        n_w, *n_ks = counts.chunk(4)
        len_w = n_w.float()
        score_w = pol_w - ref_w                      # s_theta(x, y_w), computed once

        margins, ells = [], []
        for pol_k, ref_k, n_k in zip(pol_ks, ref_ks, n_ks):
            margin_k = score_w - (pol_k - ref_k)     # s(x,y_w) - s(x,y_l^k)
            ell_k, _ = self._pair_loss(beta, margin_k, len_w, n_k.float())
            margins.append(beta * margin_k)          # m^k (reward scale)
            ells.append(ell_k)                       # ell^k = softplus(-m^k) >= 0
        margins = torch.stack(margins, dim=-1)       # [B, 3]
        ells = torch.stack(ells, dim=-1)             # [B, 3]
        A, q = counterfactual_kl_aggregate(ells, self.cf_probs, self.cf_tau)
        return A, margins, ells, q, n_w, n_ks[0]

    @staticmethod
    def _downsample_sum(logratio_pt, lm, other_len, generator):
        """SamPO: per row, sum log-ratios over T_m=min(own_len,other_len) uniformly
        sampled response positions (without replacement). Same positions serve the
        policy and reference since ``logratio_pt`` already = policy - reference.
        """
        B = logratio_pt.shape[0]
        out = logratio_pt.new_zeros(B)
        for i in range(B):
            idx = lm[i].nonzero(as_tuple=True)[0]
            own = idx.numel()
            tm = int(min(own, int(other_len[i].item())))
            if tm <= 0:
                continue
            if tm >= own:
                sel = idx  # sampling all -> plain sum of this (shorter) response
            else:
                perm = torch.randperm(own, device=logratio_pt.device, generator=generator)[:tm]
                sel = idx[perm]
            out[i] = logratio_pt[i, sel].sum()
        return out

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        c = self.lp_cfg
        method = c["method"]
        beta = c["beta"]
        ids, am, lm_in = inputs["input_ids"], inputs["attention_mask"], inputs["loss_mask"]

        logp_sum, counts, per_token, lm = _forward_logps(model, ids, am, lm_in)

        # ---- Support-Augmented KL-Dr.DPO (opt-in): [4B] batch, inner KL-DRO ----
        if self.cf_on:
            # ONE policy forward and ONE reference forward over all four blocks;
            # chosen appears once in each and is reused by the three margins.
            ref_sum, _ = self._ref_logps(ids, am, lm_in)
            A, margins, ells, q, n_w, n_l0 = self._counterfactual_loss(
                beta, logp_sum, ref_sum, counts)
            # A_i replaces ell_i as the per-EXAMPLE loss; the outer aggregation
            # (batch mean for dpo/rdpo, Dr.DPO's KL dual for drdpo) is unchanged.
            loss = self._outer_aggregate(method, A, n_w > n_l0)
            self._log_counterfactual(A, margins, ells, q)
            pol_w = logp_sum.chunk(4)[0]
            pol_l = logp_sum.chunk(4)[1]
            len_w, len_l = n_w.float(), n_l0.float()
            margin = margins[:, 0] / beta          # original-pair DPO margin
            if c.get("lambda_len_inv", 0.0) > 0:
                loss = self._length_invariance(loss, pol_w, pol_l, n_w, n_l0)
            self._log_extra(method, loss, pol_w, pol_l, len_w, len_l,
                            margin=margin, beta=beta)
            return (loss, {}) if return_outputs else loss

        pol_w, pol_l = logp_sum.chunk(2)          # summed policy logps
        n_w, n_l = counts.chunk(2)                 # response token counts (lengths)
        len_w, len_l = n_w.float(), n_l.float()

        # ---- reference-free: LMPO ----
        if method == "lmpo":
            mean_w, mean_l = pol_w / len_w, pol_l / len_l
            avg_len = (len_w + len_l) / 2.0
            p_w = torch.exp(mean_w * avg_len / len_w)
            p_l = torch.exp(mean_l * avg_len / len_l)
            m = (1.0 - p_w) * (1.0 - (p_w - p_l) ** c["lmpo_k"]) / 2.0
            margin_term = (c["lmpo_lambda"] * (m - m.mean()) / (m.std() + 1e-8)).detach()
            gamma = beta * c["lmpo_gamma_beta_ratio"]
            gap = beta * (mean_w - mean_l) - margin_term
            loss = -F.logsigmoid(gap - gamma).mean()
            self._log_extra(method, loss, pol_w, pol_l, len_w, len_l)
            return (loss, {}) if return_outputs else loss

        # ---- reference-based: dpo / rdpo / drdpo / sampo / tie ----
        ref_sum, ref_pt = self._ref_logps(ids, am, lm_in)
        ref_w, ref_l = ref_sum.chunk(2)
        logratio_w = pol_w - ref_w
        logratio_l = pol_l - ref_l
        margin = logratio_w - logratio_l  # standard DPO reward-margin / beta

        if method == "dpo":
            loss = -F.logsigmoid(beta * margin).mean()

        elif method == "rdpo":
            # R-DPO: subtract alpha*(|y_w|-|y_l|) from the reward margin (inside sigmoid).
            logits = beta * margin - c["rdpo_alpha"] * (len_w - len_l)
            loss = -F.logsigmoid(logits).mean()

        elif method == "drdpo":
            # Dr.DPO: KL-DRO dual over per-pair DPO losses (down-weights high-loss pairs).
            # (C) --drdpo_length_penalty puts R-DPO's explicit length penalty INSIDE the
            # dual (handled by _pair_loss) so the robust weighting sees length-corrected
            # losses. Batch-level aggregation lives in _outer_aggregate.
            per_ex, _ = self._pair_loss(beta, margin, len_w, len_l)  # [B], per-pair DPO loss
            loss = self._outer_aggregate(method, per_ex, n_w > n_l)

        elif method == "sampo":
            # Down-sample per-token log-ratios of each response to T_m=min(len_w,len_l).
            gen = torch.Generator(device=ids.device)
            gen.manual_seed(c["seed"] + int(self.state.global_step))
            lr_pt = (per_token - ref_pt)                    # per-token log-ratio [2B, L-1]
            lm_w, lm_l = lm.chunk(2)
            lr_w, lr_l = lr_pt.chunk(2)
            r_w = self._downsample_sum(lr_w, lm_w, n_l, gen)
            r_l = self._downsample_sum(lr_l, lm_l, n_w, gen)
            loss = -F.logsigmoid(beta * (r_w - r_l)).mean()

        elif method == "tie":
            # Tie Training (2605.11134): strict rows use standard DPO; equal-utility
            # "tie" rows (equal content, different length) use the symmetric term
            # -0.5 logσ(δ) - 0.5 logσ(-δ), the exact expectation over a random 50/50
            # winner label. This adds curvature only in the spurious (length)
            # direction, shrinking length reliance. Ties are mixed in at the data
            # level (see scripts/build_tie_dataset.py); tie_weight scales their term.
            is_tie = inputs["is_tie"].to(margin.device)
            delta = beta * margin
            pref = -F.logsigmoid(delta)
            tie = -0.5 * F.logsigmoid(delta) - 0.5 * F.logsigmoid(-delta)
            per_ex = torch.where(is_tie, c["tie_weight"] * tie, pref)
            loss = per_ex.mean()

        else:
            raise ValueError(f"Unknown length_pref method: {method}")

        if c.get("lambda_len_inv", 0.0) > 0:
            loss = self._length_invariance(loss, pol_w, pol_l, n_w, n_l)

        self._log_extra(method, loss, pol_w, pol_l, len_w, len_l, margin=margin, beta=beta)
        return (loss, {}) if return_outputs else loss

    def _length_invariance(self, loss, pol_w, pol_l, n_w, n_l):
        """(D) I4 length-invariance penalty — the round-5 winner term (REPORT §9.8
        follow-up 2, lambda=5e-4). Equalizes the MEAN RAW POLICY MARGIN across the two
        length environments, which is what the eval actually ranks on. Same per-batch
        form as invariance_train.py:104 so the two rounds stay comparable.
        """
        chosen_longer = n_w > n_l
        if chosen_longer.any() and (~chosen_longer).any():
            raw = pol_w - pol_l
            len_inv = (raw[chosen_longer].mean() - raw[~chosen_longer].mean()) ** 2
            loss = loss + self.lp_cfg["lambda_len_inv"] * len_inv
        return loss

    def _log_extra(self, method, loss, pol_w, pol_l, len_w, len_l, margin=None, beta=None):
        with torch.no_grad():
            extra = {
                f"{method}/loss": float(loss),
                f"{method}/logps_chosen": float(pol_w.mean()),
                f"{method}/logps_rejected": float(pol_l.mean()),
                f"{method}/len_chosen": float(len_w.mean()),
                f"{method}/len_rejected": float(len_l.mean()),
            }
            if margin is not None:
                extra[f"{method}/reward_margin"] = float((beta * margin).mean())
                extra[f"{method}/reward_acc"] = float((margin > 0).float().mean())
            self.log_history_extra = extra

    def _log_counterfactual(self, A, margins, ells, q):
        """Accumulate the cf/* batch metrics (all detached; q never gets gradient).

        ``margins`` [B,3] = m^k, ``ells`` [B,3] = ell^k, ``q`` [B,3] = adversarial
        probabilities, in CF_BRANCHES order (original, length, length_syntax).
        Batch values land in ``log_history_extra``; running means are flushed into
        the HF/wandb logs by ``log()`` at each logging interval (epoch metrics).
        """
        with torch.no_grad():
            m = {"cf/robust_local_loss": float(A.mean().float())}
            for j, name in enumerate(CF_BRANCHES):
                m[f"cf/loss_{name}"] = float(ells[:, j].mean().float())
                m[f"cf/margin_{name}"] = float(margins[:, j].mean().float())
                m[f"cf/weight_{name}"] = float(q[:, j].mean().float())
                m[f"cf/accuracy_{name}"] = float((margins[:, j] > 0).float().mean())
            # mass the inner adversary moves off the original pair onto the orbit
            m["cf/counterfactual_mass"] = float((q[:, 1] + q[:, 2]).mean().float())
            # an example is worst-orbit-correct only if ALL three margins are positive
            m["cf/worst_orbit_accuracy"] = float(
                (margins.min(dim=-1).values > 0).float().mean())
            self.cf_metrics_extra = m
            for k, v in m.items():
                self._cf_metric_sums[k] = self._cf_metric_sums.get(k, 0.0) + v
            self._cf_metric_n += 1

    def log(self, logs, *args, **kwargs):
        """Merge the running cf/* means into HF's periodic log (and wandb)."""
        if self._cf_metric_n:
            logs = {**logs, **{k: v / self._cf_metric_n
                               for k, v in self._cf_metric_sums.items()}}
            self._cf_metric_sums, self._cf_metric_n = {}, 0
        return super().log(logs, *args, **kwargs)

"""SIPO — Semantics-Isolating Preference Optimization.

Trains on the 4-way decomposition (`Bojian92/tldr_preference_decomposed`:
prompt, chosen, rejected_prime, rejected_double_prime, rejected). The chain
strips one attribute per stage (verified: rejected_prime/rejected_double_prime
are word-length-identical to chosen; only `rejected` differs in length):

    g1 = chosen        - rejected_prime          -> pure SEMANTICS   (want large)
    g2 = rejected_prime- rejected_double_prime    -> pure SYNTAX      (want ~0)
    g3 = rejected_double_prime - rejected         -> pure LENGTH      (want ~0)

===========================================================================
ITER-2 REBUILD (per the evaluator's restart directive)
===========================================================================
The iter-1 loss closed g2 with a reward-DIFFERENCE penalty:
`Inv(r_rp, r_rpp) = (Σ_t logp(rp_t) - Σ_t logp(rpp_t))^2`. This is
STRUCTURALLY STUCK: rp and rpp are same-length, near-identical token
sequences differing only in a few syntax tokens, so the *summed* log-prob
gradients ∇Σlogp(rp) ≈ ∇Σlogp(rpp) nearly cancel. The squared-scalar
gradient `2·(Σa-Σb)·(∇Σa-∇Σb)` therefore has a vanishing direction term and
g2 never moves off its init (~ -9.4) across 16 sweeps + 2 loss edits.

The fix is a PER-POSITION objective. rp and rpp share the *same prompt*, so
their response tokens start at the same absolute index; per-token log-probs
align position-by-position. Instead of squaring one summed difference we sum
squared PER-POSITION differences:

    L_syn = mean_t ( logp(rp_t | rp_<t) - logp(rpp_t | rpp_<t) )^2

Its gradient `2·Σ_t (a_t-b_t)(∇a_t-∇b_t)` gives every position its OWN
non-canceling coefficient, concentrating full-magnitude gradient exactly on
the positions where the two syntax variants differ. (`syn_mode="kl"` offers
the literal symmetric-KL-between-next-token-distributions variant; `perpos`
is the default because it targets the eval metric — a summed-logp gap —
directly.)

Separately, `length_acc_gap` (the |acc_chosen_longer − acc_rejected_longer|
metric) is a chosen-vs-rejected length bias that NO decomposition invariance
touches — SIPO never trains chosen vs rejected. We add an explicit
length-BALANCED chosen>rejected preference term on the SUMMED reward (the
scale the eval accuracy uses), equal-weighting the two length arms so the
under-performing chosen-longer arm gets pulled up:

    L_acc = 0.5·[ -logσ(β_acc·(Σlogp(chosen)-Σlogp(rejected)))|chosen_longer
                  -logσ(β_acc·(Σlogp(chosen)-Σlogp(rejected)))|rejected_longer ]

Full loss:

    L =  -logσ( β·(r_chosen - r_rp) - γ )      (mean scale)  # semantics  -> g1
       +  λ_syn · L_syn                         (per-position) # syntax    -> g2
       +  λ_len · Inv(r_rpp, r_rej)             (reward-diff)  # length    -> g3
       +  λ_acc · L_acc                         (length-bal.)  # acc + gap
       +  α_sft · (-mean logp(chosen))          (NLL anchor)   # hold g1

Everything is a config knob so the auto-research loop can sweep it. MEAN
base_reward is kept for the preference term (holds g1 in 15-20 with healthy
grad_norm — summed rewards explode the grad norm and collapse g1).
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import Trainer

# Chain order, most- to least-preferred (matches the eval callback).
SIPO_KEYS = ("chosen", "rejected_prime", "rejected_double_prime", "rejected")


def _encode(tokenizer, prompt, response, max_len, prompt_truncation_side):
    """Tokenize (prompt, response) with chat template; return ids + prompt_len.

    Self-contained copy of train._encode_example to avoid a circular import.
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
            prompt_ids = prompt_ids[-max_prompt_len:] if prompt_truncation_side == "left" else prompt_ids[:max_prompt_len]
        else:
            prompt_ids = prompt_ids[:1]
            response_ids = response_ids[: max_len - 1]
    return {"input_ids": prompt_ids + response_ids, "prompt_len": len(prompt_ids)}


@dataclass
class SIPODataCollator:
    """Turn raw 4-response examples into a single [4B, L] padded batch.

    Rows are role-blocked: [chosen(B), rejected_prime(B), rejected_double_prime(B),
    rejected(B)]. compute_loss reshapes to [4, B].

    CRITICAL for the per-position syntax term: all four responses of one example
    share the SAME prompt, and the prompt is tokenized identically for each, so
    their response tokens begin at the SAME absolute index. Per-token log-probs
    of rejected_prime and rejected_double_prime therefore align column-by-column,
    which is what makes the per-position invariance meaningful.
    """

    tokenizer: object
    max_len: int = 1024
    prompt_truncation_side: str = "left"

    def __call__(self, examples):
        pad_id = self.tokenizer.pad_token_id
        encoded = []  # role-major: all chosen, then all rejected_prime, ...
        for key in SIPO_KEYS:
            for ex in examples:
                encoded.append(_encode(self.tokenizer, ex["prompt"], ex[key],
                                       self.max_len, self.prompt_truncation_side))
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
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.long),
        }


def _forward_logps(model, input_ids, attention_mask, loss_mask, want_logsm=False):
    """One forward pass -> per-token & summed response log-probs (differentiable).

    Returns:
        logp_sum  [N]        summed response log-prob per row
        counts    [N]        response token count (clamped >=1)
        per_token [N, L-1]   log p(token_{k+1} | tokens_<=k), 0 outside response
        lm        [N, L-1]   bool response mask aligned to per_token
        logsm     [N, L-1, V] full log-softmax (only if want_logsm; else None)

    per_token[:, k] scores the token at absolute position k+1, so two rows whose
    responses start at the same absolute index are column-aligned over their
    overlapping response span.
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = logits[:, :-1, :].float()  # fp32 log-softmax for stability
    labels = input_ids[:, 1:].clone()
    lm = loss_mask[:, 1:].bool()
    labels[~lm] = 0
    logsm = logits.log_softmax(-1)
    per_token = torch.gather(logsm, 2, labels.unsqueeze(2)).squeeze(2)
    per_token = per_token * lm  # zero out non-response / pad positions
    counts = lm.sum(-1).clamp(min=1)
    logp_sum = per_token.sum(-1)
    return logp_sum, counts, per_token, lm, (logsm if want_logsm else None)


class SIPOTrainer(Trainer):
    """HF Trainer with the SIPO 4-way loss (see module docstring)."""

    def __init__(self, *args, sipo_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sipo_cfg = sipo_cfg  # dict: see build_trainer / module docstring

    # ----- invariance helpers ----- #
    def _inv_reward_diff(self, a, b):
        """Old reward-difference invariance: pull scalar reward logp(a)->logp(b).

        Retained for the LENGTH term (g3), where rpp and rejected have DIFFERENT
        lengths so their summed-logp gradients do NOT cancel and this moves fast.
        """
        c = self.sipo_cfg
        d = a - b
        if c["inv"] == "hinge":
            slack = (d.abs() - c["inv_eps"]).clamp(min=0.0)
            return (slack ** 2).mean()
        if c["inv"] == "l2":
            return (d ** 2).mean()
        return F.smooth_l1_loss(a, b, beta=1.0)  # smooth_l1 (Huber)

    @staticmethod
    def _perpos_logp_penalty(pt_a, lm_a, pt_b, lm_b, reduction="seqsum"):
        """Squared difference of column-aligned per-token log-probs of rp vs rpp.

        pt_a, pt_b are column-aligned per-token logps ([B, L-1]); lm_a, lm_b the
        response masks. We penalize only positions in the overlap (both are real
        response tokens) so the length mismatch (median 1 token) is simply the
        unpenalized tail. This carries a full-magnitude, non-canceling gradient:
        each position contributes 2·(a_t-b_t)(∇a_t-∇b_t) with its own coefficient.

        reduction:
          - "seqsum" (default): SUM the squared per-position diffs within each
            sequence, then mean over the batch. The syntax difference lives in a
            FEW positions; a token-count MEAN divides their gradient by the
            response length (~37) and lets g2 asymptote early. Summing within the
            sequence gives every syntax position its full, undiluted coefficient.
          - "mean": mean over all overlap response tokens (diluted; kept for
            ablation / back-compat with the first iter-2 runs).
        """
        both = lm_a & lm_b
        sq = ((pt_a - pt_b) * both) ** 2  # [B, L-1]
        if reduction == "mean":
            return sq.sum() / both.sum().clamp(min=1)
        # "seqsum": per-sequence sum, then batch mean.
        return sq.sum(-1).mean()

    @staticmethod
    def _perpos_kl_penalty(logsm_a, lm_a, logsm_b, lm_b):
        """Mean per-position symmetric KL between next-token distributions.

        Literal reading of the directive: match the policy's predictive
        distribution at aligned positions of rp vs rpp. logsm_* are [B, L-1, V]
        log-softmax tensors. Symmetric KL = KL(p||q)+KL(q||p) summed over vocab,
        averaged over overlap positions.
        """
        both = (lm_a & lm_b)
        p = logsm_a.exp()
        q = logsm_b.exp()
        sym_kl = (p * (logsm_a - logsm_b) + q * (logsm_b - logsm_a)).sum(-1)  # [B, L-1]
        sym_kl = sym_kl * both
        denom = both.sum().clamp(min=1)
        return sym_kl.sum() / denom

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        c = self.sipo_cfg
        want_kl = c.get("syn_mode", "perpos") == "kl" and c["lambda_syn"] > 0
        logp_sum, counts, per_token, lm, logsm = _forward_logps(
            model, inputs["input_ids"], inputs["attention_mask"], inputs["loss_mask"],
            want_logsm=want_kl,
        )
        r_mean = logp_sum / counts
        r_sum = logp_sum

        def _pick(which):
            return r_mean if which == "mean" else r_sum

        r_pref = _pick(c["base_reward"])            # preference scale (MEAN default)
        r_inv = _pick(c.get("inv_reward") or c["base_reward"])  # syntax reward-diff scale
        # DECOUPLED length scale: the SYNTAX reward-diff needs the SUM scale for
        # leverage on g2, but the LENGTH term (rpp vs rejected, DIFFERENT lengths)
        # is only stable on the MEAN scale — on the sum scale it fights the length
        # signal and g3 diverges (iter2g: g3 -> -8 at lr 1e-5). So length uses
        # `len_reward` (default = base_reward = mean, iter03's stable g3 regime).
        r_len = _pick(c.get("len_reward") or c["base_reward"])
        B = r_pref.shape[0] // 4

        # ----- role-blocked slices ([chosen, rp, rpp, rejected]) ----- #
        r_chosen, r_rp = r_pref[0:B], r_pref[B:2 * B]
        i_rpp, i_rej = r_len[2 * B:3 * B], r_len[3 * B:4 * B]

        # ----- (1) semantic preference: chosen > rejected_prime (mean scale) ----- #
        l_pref = -F.logsigmoid(c["beta"] * (r_chosen - r_rp) - c["gamma"]).mean()

        # ----- (2) syntax invariance: per-position rp vs rpp ----- #
        if c["lambda_syn"] > 0:
            if c.get("syn_mode", "perpos") == "kl":
                l_syn_raw = self._perpos_kl_penalty(
                    logsm[B:2 * B], lm[B:2 * B], logsm[2 * B:3 * B], lm[2 * B:3 * B]
                )
            elif c.get("syn_mode", "perpos") == "reward_diff":
                # Sequence-level syntax invariance: pull logp(rp) -> logp(rpp).
                # rpp is SHARED between g2 (=rp-rpp) and g3 (=rpp-rejected) with
                # opposite signs, so a SYMMETRIC penalty closes g2 by dragging rpp
                # down -> g3 diverges (iter2i/j: g3 -> -7). The target state is
                # rp ~= rpp ~= rejected with chosen >= rp+18, i.e. RAISE rp toward
                # a FIXED rpp (sft holds chosen above). syn_detach detaches the rpp
                # target so the gradient flows only into rp: g2 closes, rpp (hence
                # g3) is untouched, and g1 is held by the sft anchor.
                rp_r = r_inv[B:2 * B]
                rpp_r = r_inv[2 * B:3 * B]
                if c.get("syn_detach", True):
                    rpp_r = rpp_r.detach()
                l_syn_raw = self._inv_reward_diff(rp_r, rpp_r)
            else:  # "perpos" (default)
                # UNTRIED MECHANISM (iter5): honor syn_detach in the per-position
                # branch. Detaching the rpp per-token target makes the squared
                # penalty raise ONLY rp's differing positions toward a FIXED rpp,
                # so the gradient never drags rpp down (g3 untouched) and never
                # backprops the "lower rpp" direction that shares params with the
                # chosen head. Paired with sft_alpha (holds chosen -> g1), this is
                # the direct test of "can g2 close without collapsing g1?" that the
                # non-detached perpos runs (iter11/iter14) could not answer.
                pt_rp = per_token[B:2 * B]
                pt_rpp = per_token[2 * B:3 * B]
                if c.get("syn_detach", True):
                    pt_rpp = pt_rpp.detach()
                l_syn_raw = self._perpos_logp_penalty(
                    pt_rp, lm[B:2 * B], pt_rpp, lm[2 * B:3 * B],
                    reduction=c.get("syn_reduction", "seqsum"),
                )
            l_syn = c["lambda_syn"] * l_syn_raw
        else:
            l_syn = torch.zeros((), device=r_pref.device)

        # ----- (3) length invariance: rpp vs rejected (reward-diff, high leverage) ----- #
        l_len = c["lambda_len"] * self._inv_reward_diff(i_rpp, i_rej) if c["lambda_len"] > 0 \
            else torch.zeros((), device=r_pref.device)

        # ----- (4) chosen>rejected preference (attacks acc C4 + length_acc_gap C5) ----- #
        # The eval's acc / length_acc_gap compare SUMMED logp, which structurally
        # favors SHORTER responses. Training this margin on the SUM scale therefore
        # LEARNS the length shortcut (empirically acc_chosen_longer -> ~0). Instead
        # train a length-NORMALIZED (MEAN) chosen>rejected margin: it creates a
        # per-token quality gap delta = mean_chosen - mean_rejected > 0 that is
        # length-neutral and lifts BOTH sum-based arms symmetrically, shrinking the
        # gap. acc_reward="sum" keeps the legacy length-balanced arms for ablation.
        l_acc = torch.zeros((), device=r_pref.device)
        if c.get("lambda_acc", 0.0) > 0:
            if c.get("acc_reward", "mean") == "mean":
                margin = r_mean[0:B] - r_mean[3 * B:4 * B]        # per-token quality gap
                l_acc = c["lambda_acc"] * (-F.logsigmoid(c.get("beta_acc", 3.0) * margin)).mean()
            else:  # "sum" (legacy, length-biased) — equal-weighted length arms
                margin = r_sum[0:B] - r_sum[3 * B:4 * B]
                nll = -F.logsigmoid(c.get("beta_acc", 0.1) * margin)
                n_chosen, n_rej = counts[0:B], counts[3 * B:4 * B]
                cl, rl = n_chosen > n_rej, n_chosen < n_rej
                arm_losses = [nll[m].mean() for m in (cl, rl) if m.any()]
                if arm_losses:
                    l_acc = c["lambda_acc"] * (sum(arm_losses) / len(arm_losses))

        # ----- (5) SFT / NLL anchor on chosen (raise logp(chosen); hold g1) ----- #
        mean_logp_chosen = r_mean[0:B]
        l_sft = c.get("sft_alpha", 0.0) * (-mean_logp_chosen).mean()

        loss = l_pref + l_syn + l_len + l_acc + l_sft

        # Live gaps for monitoring (eval measures g2/g3 on SUMMED logp).
        with torch.no_grad():
            s_rp, s_rpp, s_rej_ = r_sum[B:2 * B], r_sum[2 * B:3 * B], r_sum[3 * B:4 * B]
            self.log_history_extra = {
                "sipo/l_pref": float(l_pref),
                "sipo/l_syn": float(l_syn),
                "sipo/l_len": float(l_len),
                "sipo/l_acc": float(l_acc),
                "sipo/l_sft": float(l_sft),
                "sipo/g1_semantics": float((r_sum[0:B] - s_rp).mean()),
                "sipo/g2_syntax": float((s_rp - s_rpp).mean()),
                "sipo/g3_length": float((s_rpp - s_rej_).mean()),
            }
        if return_outputs:
            return loss, {"r_chosen": r_chosen, "r_rejected": r_pref[3 * B:4 * B]}
        return loss

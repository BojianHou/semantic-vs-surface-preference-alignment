"""Train Qwen2.5-1.5B-Instruct on trl-lib/tldr-preference with SimPO or DPO.

SimPO is implemented via TRL's experimental CPO trainer with ``loss_type="simpo"``
(see https://github.com/huggingface/trl/blob/main/trl/experimental/cpo/cpo_trainer.py).
DPO uses ``trl.DPOTrainer`` (https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py).

Every ``--eval_every`` training steps the model is scored on the decomposed
preference set ``Bojian92/tldr_preference_decomposed`` (columns: prompt, chosen,
rejected_prime, rejected_double_prime, rejected). For each example we compute the
sum/mean response log-probabilities under the *current* policy and the pairwise
gaps between consecutive decomposition stages. Results are written to a per-step
CSV, printed to the console, and (optionally) logged to wandb.

Usage:
    python train.py --method simpo
    python train.py --method dpo
"""

import argparse
import json
import os

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

# --- Workaround for transformers 4.57.x ------------------------------------- #
# During tokenizer load, transformers runs `_patch_mistral_regex` for ANY
# tokenizer whose vocab > 100k (Qwen2.5 = 151,643). That helper calls
# huggingface_hub.model_info() over the network *unconditionally* (it ignores
# local_files_only / the local cache). Under restricted networking that HTTP
# call hangs indefinitely (online) or raises OfflineModeIsEnabled (offline),
# either of which breaks `AutoTokenizer.from_pretrained`. Qwen is not a Mistral
# model, so the probe is a no-op for us — skip it entirely and return the
# tokenizer unchanged.
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def _skip_mistral_regex_probe(cls, tokenizer, *args, **kwargs):
    return tokenizer


PreTrainedTokenizerBase._patch_mistral_regex = classmethod(_skip_mistral_regex_probe)
# ---------------------------------------------------------------------------- #

# Decomposition stages, ordered from most-preferred to least-preferred. The
# pairwise gaps below follow this ordering.
RESPONSE_KEYS = ("chosen", "rejected_prime", "rejected_double_prime", "rejected")


# --------------------------------------------------------------------------- #
# Decomposed log-prob evaluation (self-contained; chat-template aware).
# --------------------------------------------------------------------------- #
def _encode_example(tokenizer, prompt, response, max_len, prompt_truncation_side):
    """Tokenize a (prompt, response) pair with the chat template applied.

    Mirrors evaluate_decomposed_tldr_logps.py: the prompt is wrapped as a user
    turn with an assistant generation prompt, and the response is the assistant
    turn. Only the response tokens are later scored.
    """
    prompt_messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        prompt_messages + [{"role": "assistant", "content": response}], tokenize=False
    )
    if not full_text.startswith(prompt_text):
        raise ValueError("Chat template full text does not start with the generation prompt.")
    response_text = full_text[len(prompt_text):]

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
    if not response_ids:
        raise ValueError("Encountered an empty response after tokenization.")

    # Truncate to fit max_len, preferring to keep the full response.
    if len(prompt_ids) + len(response_ids) > max_len:
        max_prompt_len = max_len - len(response_ids)
        if max_prompt_len > 0:
            if prompt_truncation_side == "left":
                prompt_ids = prompt_ids[-max_prompt_len:]
            else:
                prompt_ids = prompt_ids[:max_prompt_len]
        else:
            prompt_ids = prompt_ids[:1]
            response_ids = response_ids[: max_len - 1]

    input_ids = prompt_ids + response_ids
    return {"input_ids": input_ids, "prompt_len": len(prompt_ids)}


def _batch_logps(model, tokenizer, encoded_batch, device):
    """Return (sum_logp, mean_logp, token_count) for each item in the batch."""
    max_len = max(len(item["input_ids"]) for item in encoded_batch)
    pad_id = tokenizer.pad_token_id
    input_ids, attention_mask, prompt_lens = [], [], []
    for item in encoded_batch:
        ids = item["input_ids"]
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
        prompt_lens.append(item["prompt_len"])

    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()

    # Shift so token t predicts token t+1.
    labels = input_ids[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = attention_mask.clone().bool()
    for mask, plen in zip(loss_mask, prompt_lens):
        mask[:plen] = False  # do not score prompt tokens
    loss_mask = loss_mask[:, 1:]
    labels[~loss_mask] = 0

    per_token = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)
    counts = loss_mask.sum(-1)
    logps_sum = (per_token * loss_mask).sum(-1)
    logps_mean = logps_sum / counts.clamp(min=1)
    return logps_sum.cpu(), logps_mean.cpu(), counts.cpu()


class DecomposedLogpCallback(TrainerCallback):
    """Score the decomposed preference set every ``eval_every`` steps."""

    SUMMARY_GAPS = [
        "logp_chosen_minus_rejected_prime",
        "logp_rejected_prime_minus_rejected_double_prime",
        "logp_rejected_double_prime_minus_rejected",
        "logp_chosen_minus_rejected",
    ]

    def __init__(
        self,
        tokenizer,
        eval_dataset,
        eval_every,
        result_dir,
        method,
        max_len,
        eval_batch_size,
        prompt_truncation_side,
        eval_at_start,
        use_wandb,
    ):
        self.tokenizer = tokenizer
        self.dataset = eval_dataset
        self.eval_every = eval_every
        self.result_dir = result_dir
        self.method = method
        self.max_len = max_len
        self.eval_batch_size = eval_batch_size
        self.prompt_truncation_side = prompt_truncation_side
        self.eval_at_start = eval_at_start
        self.use_wandb = use_wandb
        self.summary_rows = []
        os.makedirs(result_dir, exist_ok=True)

    # ----- TrainerCallback hooks ----- #
    def on_train_begin(self, args, state, control, **kwargs):
        if self.eval_at_start:
            self._evaluate(kwargs["model"], step=0)

    def on_step_end(self, args, state, control, **kwargs):
        if self.eval_every > 0 and state.global_step % self.eval_every == 0:
            self._evaluate(kwargs["model"], step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        # Final eval if the last step did not land on an eval boundary.
        if self.eval_every <= 0 or state.global_step % self.eval_every != 0:
            self._evaluate(kwargs["model"], step=state.global_step)
        self._write_summary()

    # ----- core ----- #
    def _evaluate(self, model, step):
        was_training = model.training
        model.eval()
        device = next(model.parameters()).device

        rows = []
        n = len(self.dataset)
        for start in range(0, n, self.eval_batch_size):
            end = min(start + self.eval_batch_size, n)
            batch = self.dataset[start:end]
            encoded, owner = [], []  # owner[i] -> (example_idx, response_key)
            for i in range(end - start):
                prompt = batch["prompt"][i]
                for key in RESPONSE_KEYS:
                    encoded.append(
                        _encode_example(
                            self.tokenizer,
                            prompt,
                            batch[key][i],
                            self.max_len,
                            self.prompt_truncation_side,
                        )
                    )
                    owner.append((start + i, key))

            sums, means, counts = _batch_logps(model, self.tokenizer, encoded, device)
            # Regroup the flat list back into one row per example.
            per_example = {}
            for (ex_idx, key), s, m, c in zip(owner, sums, means, counts):
                d = per_example.setdefault(ex_idx, {})
                d[f"{key}_logp"] = float(s)
                d[f"{key}_mean_logp"] = float(m)
                d[f"{key}_tokens"] = int(c)
            for ex_idx, d in per_example.items():
                d["index"] = ex_idx
                rows.append(d)

        df = pd.DataFrame(rows).sort_values("index").reset_index(drop=True)
        # Pairwise gaps along the decomposition chain (sum log-probs).
        df["logp_chosen_minus_rejected_prime"] = df["chosen_logp"] - df["rejected_prime_logp"]
        df["logp_rejected_prime_minus_rejected_double_prime"] = (
            df["rejected_prime_logp"] - df["rejected_double_prime_logp"]
        )
        df["logp_rejected_double_prime_minus_rejected"] = (
            df["rejected_double_prime_logp"] - df["rejected_logp"]
        )
        df["logp_chosen_minus_rejected"] = df["chosen_logp"] - df["rejected_logp"]
        df["component_gap_sum"] = (
            df["logp_chosen_minus_rejected_prime"]
            + df["logp_rejected_prime_minus_rejected_double_prime"]
            + df["logp_rejected_double_prime_minus_rejected"]
        )

        out_path = os.path.join(self.result_dir, f"decomposed_step{step:06d}.csv")
        df.to_csv(out_path, index=False)

        means = df[self.SUMMARY_GAPS].mean()
        # chosen>rejected accuracy = fraction with positive overall gap.
        accuracy = float((df["logp_chosen_minus_rejected"] > 0).mean())
        print(f"\n[decomposed-eval] step={step}  n={len(df)}  saved -> {out_path}")
        for col in self.SUMMARY_GAPS:
            print(f"    mean {col}: {means[col]:+.4f}")
        print(f"    chosen>rejected acc: {accuracy:.4f}")

        summary = {"step": step, "n": len(df), "chosen_gt_rejected_acc": accuracy}
        summary.update({f"mean_{c}": float(means[c]) for c in self.SUMMARY_GAPS})
        self.summary_rows.append(summary)

        if self.use_wandb:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(
                        {f"decomposed/{k}": v for k, v in summary.items() if k != "step"},
                        step=step,
                    )
            except Exception as exc:  # never let logging crash training
                print(f"[decomposed-eval] wandb log skipped: {exc!r}")

        if was_training:
            model.train()

    def _write_summary(self):
        if not self.summary_rows:
            return
        path = os.path.join(self.result_dir, "decomposed_summary.csv")
        pd.DataFrame(self.summary_rows).to_csv(path, index=False)
        print(f"[decomposed-eval] summary across {len(self.summary_rows)} evals -> {path}")


# --------------------------------------------------------------------------- #
# Data / model.
# --------------------------------------------------------------------------- #
def _to_conversational(example):
    """Convert standard (prompt/chosen/rejected) strings to chat format so the
    trainer applies the model's chat template (matching the eval encoding)."""
    return {
        "prompt": [{"role": "user", "content": example["prompt"]}],
        "chosen": [{"role": "assistant", "content": example["chosen"]}],
        "rejected": [{"role": "assistant", "content": example["rejected"]}],
    }


def _filter_kwargs(config_cls, kwargs):
    """Keep only kwargs that are valid dataclass fields for ``config_cls``."""
    fields = config_cls.__dataclass_fields__
    return {k: v for k, v in kwargs.items() if k in fields}


def _load_train_split(args, name):
    """Load a training split, allowing a LOCAL file via --train_data_files.

    The v2 counterfactual sets (scripts/assemble_decomposed_v2.py) are local parquet,
    so `--train_dataset parquet --train_data_files <path>` mirrors the tie-dataset path.
    """
    if args.train_data_files:
        return load_dataset(args.train_dataset, data_files=args.train_data_files,
                            split=args.train_split)
    return load_dataset(name, split=args.train_split)


def _validate_counterfactual_args(args):
    """Fail fast (before any data/model work) on an invalid cf configuration."""
    from length_pref import CF_SUPPORTED_METHODS, validate_counterfactual_config

    if not args.use_counterfactual_kl_dro:
        return
    if args.method not in CF_SUPPORTED_METHODS:
        raise ValueError(
            f"--use_counterfactual_kl_dro requires --method in {list(CF_SUPPORTED_METHODS)}, "
            f"got {args.method!r}. (sampo/lmpo/tie/simpo/sipo/orpo have no per-pair "
            "sigmoid-margin loss to aggregate over the counterfactual orbit.)"
        )
    probs, tau = validate_counterfactual_config(
        args.cf_epsilon_length, args.cf_epsilon_length_syntax, args.cf_kl_temperature
    )
    print(f"[cf-kl-dro] enabled: p_0={probs[0]:.4f} p_L={probs[1]:.4f} p_LS={probs[2]:.4f} "
          f"tau_v={tau} outer={args.method}"
          + (f" (beta'={args.drdpo_beta_prime})" if args.method == "drdpo" else ""))


def build_trainer(args, model, tokenizer, train_dataset):
    common = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.eval_every,
        save_strategy="no" if args.no_save else "steps",
        save_steps=args.save_steps,
        bf16=True,
        beta=args.beta,
        max_length=args.max_len,
        report_to=(["wandb"] if args.report_to == "wandb" else []),
        run_name=f"{args.method}-qwen2.5-1.5B-tldr",
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
    )

    # --use_counterfactual_kl_dro needs a 4-response batch, which TRL's DPOTrainer
    # cannot express, so `dpo` is routed to the local LengthPrefTrainer (whose `dpo`
    # mode is smoke-verified to match TRL's loss). Flag off => routing unchanged.
    cf_on = args.use_counterfactual_kl_dro

    if args.method == "dpo" and not cf_on:
        from trl import DPOConfig, DPOTrainer

        cfg_kwargs = dict(common)
        cfg_kwargs["loss_type"] = args.dpo_loss_type
        cfg_kwargs["max_prompt_length"] = args.max_prompt_len
        cfg_kwargs["label_smoothing"] = args.label_smoothing  # rDPO flip prob (loss_type=robust)
        cfg_kwargs["ld_alpha"] = args.ld_alpha                # LD-DPO tail down-weight (None = vanilla)
        config = DPOConfig(**_filter_kwargs(DPOConfig, cfg_kwargs))
        trainer = DPOTrainer(
            model=model,
            ref_model=None,  # trainer clones the policy as the reference
            args=config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )
    elif args.method == "simpo":
        os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
        from trl.experimental.cpo import CPOConfig, CPOTrainer

        cfg_kwargs = dict(common)
        cfg_kwargs["loss_type"] = "simpo"
        cfg_kwargs["cpo_alpha"] = args.cpo_alpha
        cfg_kwargs["simpo_gamma"] = args.simpo_gamma
        config = CPOConfig(**_filter_kwargs(CPOConfig, cfg_kwargs))

        # SimPO's defining feature is length-normalized (average) log-probs. To
        # ablate length bias we can disable that and use *summed* log-probs. TRL
        # hardcodes ``average_log_prob=self.loss_type in ["ipo", "simpo"]`` inside
        # ``concatenated_forward`` (cpo_trainer.py), so the only seam is the
        # ``get_batch_logps`` staticmethod it calls — override it to force sums.
        if args.simpo_average_log_prob:
            trainer_cls = CPOTrainer
        else:
            class _SimPOSumLogpTrainer(CPOTrainer):
                @staticmethod
                def get_batch_logps(logits, labels, average_log_prob=False, is_encoder_decoder=False):
                    # Ignore the caller's average_log_prob (True for simpo) and sum.
                    return CPOTrainer.get_batch_logps(
                        logits, labels, average_log_prob=False, is_encoder_decoder=is_encoder_decoder
                    )

            trainer_cls = _SimPOSumLogpTrainer

        trainer = trainer_cls(
            model=model,
            args=config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )
    elif args.method == "orpo":
        os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
        from trl.experimental.orpo import ORPOConfig, ORPOTrainer

        # ORPO is reference-free and monolithic (SFT + odds-ratio). Its ``beta`` is
        # the odds-ratio weight lambda; its log-likelihoods are length-averaged, so
        # length robustness rides on that normalization (like SimPO).
        cfg_kwargs = dict(common)
        cfg_kwargs["max_prompt_length"] = args.max_prompt_len
        config = ORPOConfig(**_filter_kwargs(ORPOConfig, cfg_kwargs))
        trainer = ORPOTrainer(
            model=model,
            args=config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )
    elif args.method in ("rdpo", "drdpo", "sampo", "lmpo", "tie") or (cf_on and args.method == "dpo"):
        # Custom length-robust / spurious-correlation losses not available in TRL.
        # Implemented in length_pref.py as a self-contained pairwise trainer whose
        # ``dpo`` mode is smoke-verified to match TRL's DPOTrainer loss.
        from transformers import TrainingArguments

        from length_pref import LengthPrefDataCollator, LengthPrefTrainer

        ta_kwargs = {k: v for k, v in common.items() if k not in ("beta", "max_length")}
        config = TrainingArguments(**_filter_kwargs(TrainingArguments, ta_kwargs))
        lp_cfg = dict(
            method=args.method,
            beta=args.beta,
            rdpo_alpha=args.rdpo_alpha,
            drdpo_beta_prime=args.drdpo_beta_prime,
            drdpo_stratify_length=args.drdpo_stratify_length,
            drdpo_length_penalty=args.drdpo_length_penalty,
            drdpo_buffer=args.drdpo_buffer,
            lambda_len_inv=args.lambda_len_inv,
            lmpo_gamma_beta_ratio=args.lmpo_gamma_beta_ratio,
            lmpo_lambda=args.lmpo_lambda,
            lmpo_k=args.lmpo_k,
            tie_weight=args.tie_weight,
            tie_len_tol=args.tie_len_tol,
            label_smoothing=args.label_smoothing,
            seed=args.seed,
            # Support-Augmented KL-Dr.DPO (off by default; see length_pref.py).
            use_counterfactual_kl_dro=cf_on,
            cf_epsilon_length=args.cf_epsilon_length,
            cf_epsilon_length_syntax=args.cf_epsilon_length_syntax,
            cf_kl_temperature=args.cf_kl_temperature,
        )
        print(f"[{args.method}] config: {lp_cfg}")
        trainer = LengthPrefTrainer(
            model=model,
            args=config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            data_collator=LengthPrefDataCollator(
                tokenizer=tokenizer,
                max_len=args.max_len,
                max_prompt_len=args.max_prompt_len,
                prompt_truncation_side=args.prompt_truncation_side,
                counterfactual=cf_on,
            ),
            lp_cfg=lp_cfg,
        )
    elif args.method == "sipo":
        from transformers import TrainingArguments

        from sipo import SIPODataCollator, SIPOTrainer

        # Plain HF TrainingArguments (no trl beta/max_length fields).
        ta_kwargs = {k: v for k, v in common.items() if k not in ("beta", "max_length")}
        config = TrainingArguments(**_filter_kwargs(TrainingArguments, ta_kwargs))
        sipo_cfg = dict(
            beta=args.beta,
            gamma=args.simpo_gamma,
            lambda_len=args.sipo_lambda_len,
            lambda_syn=args.sipo_lambda_syn,
            base_reward=args.sipo_base_reward,
            inv_reward=args.sipo_inv_reward,
            inv=args.sipo_inv,
            inv_eps=args.sipo_inv_eps,
            sft_alpha=args.sipo_sft_alpha,
            syn_mode=args.sipo_syn_mode,
            syn_reduction=args.sipo_syn_reduction,
            lambda_acc=args.sipo_lambda_acc,
            beta_acc=args.sipo_beta_acc,
            acc_reward=args.sipo_acc_reward,
            len_reward=args.sipo_len_reward,
            syn_detach=args.sipo_syn_detach,
        )
        print(f"[sipo] config: {sipo_cfg}")
        trainer = SIPOTrainer(
            model=model,
            args=config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            data_collator=SIPODataCollator(
                tokenizer=tokenizer,
                max_len=args.max_len,
                prompt_truncation_side=args.prompt_truncation_side,
            ),
            sipo_cfg=sipo_cfg,
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")
    return trainer


def main():
    args = parse_args()
    _validate_counterfactual_args(args)  # invalid cf config => raise before any work

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )

    if args.method == "sipo":
        # SIPO trains on the 4-way decomposition (raw columns; collator applies
        # the chat template). Default its train set to the decomposed data.
        sipo_train_dataset = args.train_dataset
        if sipo_train_dataset == "trl-lib/tldr-preference":  # generic default -> switch
            sipo_train_dataset = args.eval_dataset
        train_ds = load_dataset(sipo_train_dataset, split=args.train_split)
        missing = [k for k in ("prompt", "chosen", "rejected_prime",
                               "rejected_double_prime", "rejected") if k not in train_ds.column_names]
        if missing:
            raise ValueError(f"SIPO train set missing {missing}; has {train_ds.column_names}")
        if args.max_train_samples is not None:
            train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
        print(f"[sipo] train set {sipo_train_dataset}[{args.train_split}] n={len(train_ds)} (raw 4-way)")
    elif args.method == "tie":
        # Tie Training: prebuilt mixed dataset (strict + length-ties) with an
        # `is_tie` column. Keep raw strings (the LengthPrefDataCollator applies the
        # chat template) and DO NOT map to conversational (that would drop is_tie).
        train_ds = load_dataset("parquet", data_files=args.tie_dataset, split="train")
        if "is_tie" not in train_ds.column_names:
            raise ValueError(f"Tie dataset {args.tie_dataset} missing 'is_tie'; "
                             f"build it with scripts/build_tie_dataset.py")
        if args.max_train_samples is not None:
            train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
        n_tie = sum(train_ds["is_tie"])
        print(f"[tie] mixed set {args.tie_dataset} n={len(train_ds)} tie={n_tie} "
              f"strict={len(train_ds) - n_tie}")
    elif args.use_counterfactual_kl_dro:
        # Support-Augmented KL-Dr.DPO trains on the 4-way decomposed columns; keep
        # raw strings (the collator applies the chat template) and DO NOT map to
        # conversational, which would drop the counterfactual columns.
        from length_pref import CF_KEYS

        cf_train_dataset = args.train_dataset
        if cf_train_dataset == "trl-lib/tldr-preference" and not args.train_data_files:
            cf_train_dataset = args.eval_dataset  # generic default -> the decomposed set
        train_ds = _load_train_split(args, cf_train_dataset)
        missing = [k for k in ("prompt", *CF_KEYS) if k not in train_ds.column_names]
        if missing:
            raise ValueError(
                f"--use_counterfactual_kl_dro requires columns {missing} which "
                f"{cf_train_dataset}[{args.train_split}] does not have "
                f"(it has {train_ds.column_names}). Point --train_dataset at a "
                "decomposed set such as Bojian92/tldr_preference_decomposed."
            )
        if args.max_train_samples is not None:
            train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
        print(f"[cf-kl-dro] train set {cf_train_dataset}[{args.train_split}] "
              f"n={len(train_ds)} (chosen + 3 rejected variants)")
    else:
        # Standard preference -> conversational so chat template applies.
        train_ds = _load_train_split(args, args.train_dataset)
        # Round-3 length-direction arm: restrict to a precomputed length subset
        # (see build_length_split.py), applied on raw indices before other subsetting.
        if args.train_length_subset is not None:
            with open(args.length_split_file) as f:
                split = json.load(f)
            if split.get("train_dataset") != args.train_dataset or split.get("train_split") != args.train_split:
                raise ValueError(
                    f"Split file {args.length_split_file} was built for "
                    f"{split.get('train_dataset')}[{split.get('train_split')}], not "
                    f"{args.train_dataset}[{args.train_split}]."
                )
            idx = split["indices"][args.train_length_subset]
            print(f"[length-split] arm={args.train_length_subset} N={len(idx)} "
                  f"(seed={split.get('seed')}, source={args.length_split_file})")
            train_ds = train_ds.select(idx)
        if args.max_train_samples is not None:
            train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
        train_ds = train_ds.map(_to_conversational, desc="to-conversational")

    # Decomposed eval data (kept as raw strings; callback applies chat template).
    eval_ds = load_dataset(args.eval_dataset, split=args.eval_split)
    missing = [k for k in ("prompt", *RESPONSE_KEYS) if k not in eval_ds.column_names]
    if missing:
        raise ValueError(f"Eval dataset missing columns {missing}; has {eval_ds.column_names}")
    if args.max_eval_samples is not None:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))

    result_dir = os.path.join(args.result_dir, args.method)
    callback = DecomposedLogpCallback(
        tokenizer=tokenizer,
        eval_dataset=eval_ds,
        eval_every=args.eval_every,
        result_dir=result_dir,
        method=args.method,
        max_len=args.max_len,
        eval_batch_size=args.eval_batch_size,
        prompt_truncation_side=args.prompt_truncation_side,
        eval_at_start=args.eval_at_start,
        use_wandb=(args.report_to == "wandb"),
    )

    trainer = build_trainer(args, model, tokenizer, train_ds)
    trainer.add_callback(callback)

    trainer.train()

    if not args.no_save:
        trainer.save_model(os.path.join(args.output_dir, "final"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))
    print(f"Done. Decomposed-eval CSVs in {result_dir}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--method",
        choices=["simpo", "dpo", "sipo", "orpo", "rdpo", "drdpo", "sampo", "lmpo", "tie"],
        required=True,
        help="dpo/simpo/sipo/orpo use TRL (orpo=experimental); dpo also covers LD-DPO "
        "(--ld_alpha) and rDPO (--dpo_loss_type robust --label_smoothing). "
        "rdpo/drdpo/sampo/lmpo/tie are custom length-robust losses in length_pref.py.",
    )

    # model / data
    p.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--train_dataset", default="trl-lib/tldr-preference")
    p.add_argument("--train_split", default="train")
    p.add_argument("--eval_dataset", default="Bojian92/tldr_preference_decomposed")
    p.add_argument("--eval_split", default="validation")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--result_dir", default="results/_scratch")
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)
    # Round-3: train on one length-direction arm (see build_length_split.py).
    p.add_argument(
        "--train_length_subset",
        choices=["chosen_longer", "rejected_longer"],
        default=None,
        help="Restrict training data to one response-length-direction arm; "
        "None (default) trains on the full mixed set.",
    )
    p.add_argument("--length_split_file", default="results/length_split/split_indices.json")
    p.add_argument("--train_data_files", default=None,
                   help="local training file(s), e.g. a v2 counterfactual parquet. When set, "
                        "--train_dataset is the builder name ('parquet') and --train_split the "
                        "split name inside it ('train').")
    p.add_argument("--tie_dataset", default="data/tie_mix.parquet",
                   help="Tie Training: path to the prebuilt mixed parquet (strict + length-ties "
                        "with an is_tie column); see scripts/build_tie_dataset.py.")

    # lengths
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--max_prompt_len", type=int, default=768)
    p.add_argument("--prompt_truncation_side", choices=["left", "right"], default="left")

    # optimization
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=None, help="default: 5e-7 dpo / 1e-6 simpo")
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--lr_scheduler_type", default="cosine")
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--gradient_checkpointing", action="store_true", default=False)
    p.add_argument("--max_grad_norm", type=float, default=1.0, help="grad-norm clip (HF default 1.0)")
    p.add_argument("--seed", type=int, default=42)

    # method-specific
    p.add_argument("--beta", type=float, default=None, help="default: 0.1 dpo / 2.0 simpo / 0.1 orpo")
    p.add_argument("--dpo_loss_type", default="sigmoid",
                   help="DPO loss_type. 'robust' = rDPO (Provably-Robust-DPO, label-noise robustness) "
                        "when paired with --label_smoothing>0.")
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="DPO label smoothing / rDPO flip probability epsilon (loss_type=robust).")
    p.add_argument("--ld_alpha", type=float, default=None,
                   help="LD-DPO alpha: down-weights the verbose (length-excess) tail of the longer "
                        "response. None = vanilla DPO; 1.0 = no weighting; 0.0 = mask the tail entirely.")
    p.add_argument("--cpo_alpha", type=float, default=0.0, help="SimPO: 0.0 = pure SimPO; >0 adds CPO/NLL term")
    p.add_argument("--simpo_gamma", type=float, default=0.5, help="SimPO target reward margin")
    p.add_argument(
        "--simpo_average_log_prob",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="SimPO only: if True (default) use length-normalized (average) log-probs; "
        "pass --no-simpo_average_log_prob to use summed log-probs (length-bias ablation).",
    )

    # Custom length-robust / spurious-correlation losses — see length_pref.py.
    p.add_argument("--rdpo_alpha", type=float, default=0.05,
                   help="R-DPO (2403.19159) length penalty coeff: subtract alpha*(len_w-len_l) from "
                        "the reward margin inside the sigmoid. 0 = vanilla DPO.")
    p.add_argument("--drdpo_beta_prime", type=float, default=1.0,
                   help="Dr.DPO (2407.07880) pairwise-DRO temperature beta'. Loss = -b' log E[exp(-L_i/b')]; "
                        "down-weights high-loss (noisy/length-exploiting) pairs. Large b' -> vanilla DPO.")
    # Round-7 Dr.DPO tuning mechanisms (see scripts/run_round7.sh).
    p.add_argument("--drdpo_stratify_length", action="store_true", default=False,
                   help="(B) Apply the Dr.DPO KL-DRO dual WITHIN each length environment "
                        "(chosen-longer vs not) and weight the two equally, making the robust "
                        "weighting length-aware. Round-7 §A found the plain dual is length-blind.")
    p.add_argument("--drdpo_length_penalty", action="store_true", default=False,
                   help="(C) Subtract R-DPO's alpha*(len_w-len_l) from the logit INSIDE the Dr.DPO "
                        "dual, so the robust weighting sees length-corrected losses. Uses --rdpo_alpha.")
    p.add_argument("--drdpo_buffer", type=int, default=0,
                   help="(E) Normalize the Dr.DPO weights over a FIFO buffer of this many recent "
                        "losses instead of the micro-batch (0 = off, per-micro-batch as in round 6).")
    p.add_argument("--lambda_len_inv", type=float, default=0.0,
                   help="(D) I4 length-invariance penalty (round-5 winner, lambda=5e-4): squared "
                        "difference of the mean raw policy margin between length environments.")
    # Support-Augmented KL-Dr.DPO (round-8) — see length_pref.py. OFF by default;
    # when off no extra columns are read, no extra forward is run, and dpo/rdpo/
    # drdpo behave exactly as before.
    p.add_argument("--use_counterfactual_kl_dro", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Aggregate each example's (rejected, rejected_double_prime, "
                        "rejected_prime) pairs with a KL-REGULARIZED (fixed-temperature) "
                        "DRO inner problem, then feed the per-example robust loss A_i to "
                        "the unchanged outer aggregation (mean for dpo/rdpo, Dr.DPO's dual "
                        "for drdpo). Requires --method dpo/rdpo/drdpo and a decomposed "
                        "train set.")
    p.add_argument("--cf_epsilon_length", type=float, default=0.10,
                   help="Nominal probability p_L of the length-matched counterfactual "
                        "(rejected_double_prime). Must be > 0.")
    p.add_argument("--cf_epsilon_length_syntax", type=float, default=0.05,
                   help="Nominal probability p_LS of the length+syntax-matched "
                        "counterfactual (rejected_prime). Must be > 0 and "
                        "eps_L + eps_LS < 1; p_0 = 1 - eps_L - eps_LS.")
    p.add_argument("--cf_kl_temperature", type=float, default=1.0,
                   help="Counterfactual KL temperature tau_v > 0 (distinct from DPO --beta "
                        "and Dr.DPO --drdpo_beta_prime). tau_v -> 0 gives the max over the "
                        "orbit; tau_v -> inf gives the nominal weighted mean.")

    p.add_argument("--lmpo_gamma_beta_ratio", type=float, default=0.55,
                   help="LMPO/SimPO target margin ratio; gamma = beta * ratio.")
    p.add_argument("--lmpo_lambda", type=float, default=0.2, help="LMPO probability-margin weight lambda.")
    p.add_argument("--lmpo_k", type=float, default=5.0, help="LMPO margin power k.")
    p.add_argument("--tie_weight", type=float, default=1.0,
                   help="Tie Training (2605.11134): weight on the tie (equal-utility) regularization term.")
    p.add_argument("--tie_len_tol", type=int, default=5,
                   help="Tie Training: |len_chosen - len_rejected| <= tol (tokens) flags a length-tie pair "
                        "whose preference is regularized toward indifference.")

    # SIPO (Semantics-Isolating Preference Optimization) — see sipo.py.
    p.add_argument("--sipo_lambda_len", type=float, default=1.0, help="weight of the length-invariance term (g3->0)")
    p.add_argument("--sipo_lambda_syn", type=float, default=1.0, help="weight of the syntax-invariance term (g2->0)")
    p.add_argument("--sipo_base_reward", choices=["sum", "mean"], default="sum", help="preference reward: summed or mean logp")
    p.add_argument("--sipo_inv_reward", choices=["sum", "mean"], default=None,
                   help="invariance reward scale (default: = base_reward). 'sum' gives the invariance "
                        "penalty ~response-length more leverage to close g2/g3 while preference stays on base_reward.")
    p.add_argument("--sipo_inv", choices=["l2", "hinge", "smooth_l1"], default="smooth_l1", help="invariance penalty form")
    p.add_argument("--sipo_inv_eps", type=float, default=0.0, help="hinge tolerance (nats) for --sipo_inv hinge")
    p.add_argument("--sipo_sft_alpha", type=float, default=0.0,
                   help="weight of an SFT/NLL anchor on chosen (raises logp(chosen) so g1 stays high "
                        "while syntax invariance raises rp to close g2). 0 = pure contrastive SIPO.")
    # ITER-2 rebuild: per-position syntax objective + length-balanced acc term.
    p.add_argument("--sipo_syn_mode", choices=["perpos", "kl", "reward_diff"], default="perpos",
                   help="syntax invariance mechanism: 'perpos' (default) = mean per-position squared "
                        "per-token logp difference between rp/rpp (full-magnitude gradient); 'kl' = mean "
                        "per-position symmetric KL between next-token distributions; 'reward_diff' = legacy "
                        "squared summed-reward difference (structurally stuck on this near-identical pair).")
    p.add_argument("--sipo_syn_reduction", choices=["seqsum", "mean"], default="seqsum",
                   help="perpos syntax reduction: 'seqsum' (default) sums squared per-position diffs "
                        "within a sequence then batch-means (undiluted gradient on the few syntax "
                        "positions); 'mean' averages over all response tokens (diluted).")
    p.add_argument("--sipo_lambda_acc", type=float, default=0.0,
                   help="weight of the length-BALANCED chosen>rejected preference term (summed logp, the "
                        "scale eval accuracy uses). Attacks acc (C4) and length_acc_gap (C5) directly. 0 = off.")
    p.add_argument("--sipo_beta_acc", type=float, default=0.1,
                   help="logistic temperature for the acc term. Use ~3.0 for acc_reward=mean "
                        "(per-token margin ~0.3 nats), ~0.1 for acc_reward=sum (~6 nat margin).")
    p.add_argument("--sipo_syn_detach", action=argparse.BooleanOptionalAction, default=True,
                   help="reward_diff syntax only: detach the rpp target so the gradient RAISES rp toward "
                        "a fixed rpp (closes g2) instead of dragging rpp down (which diverges g3, since rpp "
                        "is shared by g2 and g3). --no-sipo_syn_detach recovers the symmetric penalty.")
    p.add_argument("--sipo_len_reward", choices=["mean", "sum"], default="mean",
                   help="reward scale for the length-invariance term (g3). Default 'mean' is the stable "
                        "regime (iter03 g3=-0.5); 'sum' fights the length signal and g3 diverges. Decoupled "
                        "from --sipo_inv_reward so syntax can use 'sum' leverage while length stays on 'mean'.")
    p.add_argument("--sipo_acc_reward", choices=["mean", "sum"], default="mean",
                   help="reward scale for the chosen>rejected acc term: 'mean' (default) trains a "
                        "length-NEUTRAL per-token margin that lifts both length arms symmetrically and "
                        "shrinks length_acc_gap; 'sum' (legacy) is length-biased and inflates the gap.")

    # eval / logging
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--eval_at_start", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--no_save", action="store_true", default=False)
    p.add_argument("--report_to", choices=["wandb", "none"], default="wandb")
    p.add_argument("--attn_implementation", default="sdpa")

    args = p.parse_args()

    # Method-specific defaults. Round-6 length-robust runs all use the headline
    # 5e-6 LR (matching SimPO v2 / DPO v3) unless overridden on the CLI.
    if args.learning_rate is None:
        args.learning_rate = {"simpo": 1e-6, "sipo": 5e-6}.get(args.method, 5e-7)
    if args.beta is None:
        # summed-logp methods (dpo/rdpo/drdpo/sampo/tie/orpo) use small beta;
        # length-averaged (mean-logp) methods (simpo/lmpo) use large beta;
        # sipo uses summed-logp rewards so beta must stay small (cf. §9.3).
        args.beta = {"simpo": 2.0, "lmpo": 2.5, "sipo": 0.05, "orpo": 0.1}.get(args.method, 0.1)
    if args.output_dir is None:
        args.output_dir = os.path.join("checkpoints", f"{args.method}-qwen2.5-1.5B-tldr")
    return args


if __name__ == "__main__":
    main()

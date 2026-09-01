"""DPO with optional Group-DRO over response-length direction.

Tests the idea: split each batch into {chosen-longer, rejected-longer} groups and
optimize the WORST group (group-DRO) instead of the plain mean, to stop the model
keying on length. Reference-based DPO (sum-logp, sigmoid), matching DPO v3's
length-robust regime; ref model = frozen base policy.

  loss_off    = mean_i  -logσ(β·Δ_i)                          # plain DPO
  loss_worst  = max( mean_{chosen-longer} , mean_{rejected-longer} ) of -logσ(β·Δ_i)
  Δ_i = (logπθ(chosen) − logπref(chosen)) − (logπθ(rejected) − logπref(rejected))

Trains on (prompt, chosen, rejected); groups by chosen vs rejected response tokens.
Reuses sipo._encode / sipo._response_logps and train.DecomposedLogpCallback.
"""
import argparse
import os

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

import train  # noqa: F401  (tokenizer network-hang workaround + eval callback)
from sipo import _encode, _forward_logps


class DPODROCollator:
    def __init__(self, tokenizer, max_len=1024, prompt_truncation_side="left"):
        self.tok = tokenizer; self.max_len = max_len; self.side = prompt_truncation_side

    def __call__(self, examples):
        pad = self.tok.pad_token_id
        enc = []
        for key in ("chosen", "rejected"):  # role-blocked: chosen(B) then rejected(B)
            for ex in examples:
                enc.append(_encode(self.tok, ex["prompt"], ex[key], self.max_len, self.side))
        L = max(len(e["input_ids"]) for e in enc)
        ids, am, lm = [], [], []
        for e in enc:
            x = e["input_ids"]; p = L - len(x)
            ids.append(x + [pad] * p); am.append([1] * len(x) + [0] * p)
            m = [0] * L
            for j in range(e["prompt_len"], len(x)):
                m[j] = 1
            lm.append(m)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(am),
                "loss_mask": torch.tensor(lm)}


class DPODROTrainer(Trainer):
    def __init__(self, *a, ref_model=None, beta=0.1, group_dro="off", **k):
        super().__init__(*a, **k)
        self.ref_model = ref_model
        self.beta = beta
        self.group_dro = group_dro

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        ii, am, lm = inputs["input_ids"], inputs["attention_mask"], inputs["loss_mask"]
        pol, counts = _forward_logps(model, ii, am, lm)[:2]           # summed logp, counts [2B]
        if self.ref_model.device != ii.device:
            self.ref_model.to(ii.device)
        with torch.no_grad():
            ref = _forward_logps(self.ref_model, ii, am, lm)[0]
        B = pol.shape[0] // 2
        pc, pr = pol[:B], pol[B:]
        rc, rr = ref[:B], ref[B:]
        cc, cr = counts[:B], counts[B:]                               # token counts
        delta = (pc - rc) - (pr - rr)
        per_ex = -F.logsigmoid(self.beta * delta)                    # [B]

        chosen_longer = cc > cr                                       # group A
        if self.group_dro == "worst":
            ga, gb = chosen_longer, ~chosen_longer
            la = per_ex[ga].mean() if ga.any() else None
            lb = per_ex[gb].mean() if gb.any() else None
            if la is None:
                loss = lb
            elif lb is None:
                loss = la
            else:
                loss = torch.maximum(la, lb)
        else:
            loss = per_ex.mean()

        with torch.no_grad():
            self.log_history_extra = {
                "dro/loss_chosen_longer": float(per_ex[chosen_longer].mean()) if chosen_longer.any() else float("nan"),
                "dro/loss_rejected_longer": float(per_ex[~chosen_longer].mean()) if (~chosen_longer).any() else float("nan"),
                "dro/frac_chosen_longer": float(chosen_longer.float().mean()),
            }
        return (loss, {"delta": delta}) if return_outputs else loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group_dro", choices=["off", "worst"], default="off")
    p.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--train_dataset", default="Bojian92/tldr_preference_decomposed")
    p.add_argument("--train_split", default="train")
    p.add_argument("--eval_dataset", default="Bojian92/tldr_preference_decomposed")
    p.add_argument("--eval_split", default="validation")
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=256)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--prompt_truncation_side", default="left")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--result_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_save", action="store_true")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16)
    ref = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16)
    ref.eval()
    for pm in ref.parameters():
        pm.requires_grad_(False)

    train_ds = load_dataset(args.train_dataset, split=args.train_split)
    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    eval_ds = load_dataset(args.eval_dataset, split=args.eval_split)
    if args.max_eval_samples:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))
    print(f"[dro_dpo] group_dro={args.group_dro} train_n={len(train_ds)} eval_n={len(eval_ds)}")

    ta = TrainingArguments(
        output_dir=args.output_dir, per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs, lr_scheduler_type="cosine", warmup_ratio=0.1,
        logging_steps=args.eval_every, save_strategy="no", bf16=True, seed=args.seed,
        remove_unused_columns=False, report_to=[],
    )
    result_dir = os.path.join(args.result_dir, "dpo")
    callback = train.DecomposedLogpCallback(
        tokenizer=tok, eval_dataset=eval_ds, eval_every=args.eval_every, result_dir=result_dir,
        method="dpo", max_len=args.max_len, eval_batch_size=args.eval_batch_size,
        prompt_truncation_side=args.prompt_truncation_side, eval_at_start=True, use_wandb=False,
    )
    trainer = DPODROTrainer(
        model=model, args=ta, train_dataset=train_ds, processing_class=tok,
        data_collator=DPODROCollator(tok, args.max_len, args.prompt_truncation_side),
        ref_model=ref, beta=args.beta, group_dro=args.group_dro,
    )
    trainer.add_callback(callback)
    trainer.train()
    if not args.no_save:
        trainer.save_model(os.path.join(args.output_dir, "final"))
        tok.save_pretrained(os.path.join(args.output_dir, "final"))
    print(f"Done. -> {result_dir}")


if __name__ == "__main__":
    main()

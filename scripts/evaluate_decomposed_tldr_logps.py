import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


RESPONSE_KEYS = ("chosen", "rejected_prime", "rejected_double_prime", "rejected")


def _str_to_dtype(dtype_name):
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def _load_model_and_tokenizer(args):
    torch_dtype = _str_to_dtype(args.torch_dtype)
    trust_remote_code = not args.no_trust_remote_code

    adapter_config = None
    try:
        from peft import PeftConfig

        adapter_config = PeftConfig.from_pretrained(args.model_name_or_path)
    except Exception:
        adapter_config = None

    tokenizer_name = args.tokenizer_name_or_path or args.model_name_or_path
    if adapter_config is not None and args.tokenizer_name_or_path is None:
        tokenizer_name = args.model_name_or_path

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=trust_remote_code,
            use_fast=not args.disable_fast_tokenizer,
        )
    except Exception:
        if adapter_config is None:
            raise
        tokenizer = AutoTokenizer.from_pretrained(
            adapter_config.base_model_name_or_path,
            trust_remote_code=trust_remote_code,
            use_fast=not args.disable_fast_tokenizer,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch_dtype,
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    if adapter_config is not None:
        from peft import PeftModel

        base_model_name = args.base_model_name_or_path or adapter_config.base_model_name_or_path
        model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
        model = PeftModel.from_pretrained(model, args.model_name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)

    if not args.device_map:
        model.to(torch.device(args.device))
    model.eval()
    return model, tokenizer


def _load_split(args):
    dataset = args.dataset_name_or_path
    if os.path.isdir(dataset):
        data = load_from_disk(dataset)
    elif Path(dataset).suffix.lower() in {".json", ".jsonl", ".csv"}:
        ext = Path(dataset).suffix.lower().lstrip(".")
        data = load_dataset("json" if ext == "jsonl" else ext, data_files=dataset)
    else:
        data = load_dataset(dataset)

    if hasattr(data, "keys"):
        if args.split not in data:
            raise ValueError(f"Split {args.split!r} not found. Available splits: {list(data.keys())}")
        data = data[args.split]

    missing = [key for key in (args.prompt_key, *RESPONSE_KEYS) if key not in data.column_names]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}. Found columns: {data.column_names}")

    if args.max_samples is not None:
        data = data.select(range(min(args.max_samples, len(data))))
    return data


def _format_prompt_and_response(tokenizer, prompt, response, args):
    if args.input_template is not None:
        prompt = args.input_template.format(prompt)

    if args.apply_chat_template:
        prompt_messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            prompt_messages + [{"role": "assistant", "content": response}],
            tokenize=False,
        )
        if not full_text.startswith(prompt_text):
            raise ValueError("Chat template full text does not start with the generation prompt.")
        response_text = full_text[len(prompt_text) :]
    else:
        prompt_text = prompt
        response_text = response
        if args.add_eos:
            response_text = response_text.rstrip("\n")
            eos_token = tokenizer.eos_token or ""
            if eos_token and not response_text.endswith(eos_token):
                response_text = response_text + " " + eos_token

    return prompt_text, response_text


def _truncate_prompt(prompt_ids, response_ids, max_length, prompt_truncation_side):
    prompt_truncated = False
    response_truncated = False

    if len(prompt_ids) + len(response_ids) <= max_length:
        return prompt_ids, response_ids, prompt_truncated, response_truncated

    max_prompt_len = max_length - len(response_ids)
    if max_prompt_len > 0:
        prompt_truncated = len(prompt_ids) > max_prompt_len
        if prompt_truncation_side == "left":
            prompt_ids = prompt_ids[-max_prompt_len:]
        else:
            prompt_ids = prompt_ids[:max_prompt_len]
        return prompt_ids, response_ids, prompt_truncated, response_truncated

    prompt_keep = 1 if prompt_ids else 0
    response_keep = max_length - prompt_keep
    if response_keep <= 0:
        raise ValueError("max_length must leave room for at least one response token.")

    prompt_truncated = len(prompt_ids) > prompt_keep
    response_truncated = len(response_ids) > response_keep
    response_ids = response_ids[:response_keep]
    prompt_ids = prompt_ids[:prompt_keep]
    return prompt_ids, response_ids, prompt_truncated, response_truncated


def _encode_example(tokenizer, prompt, response, args):
    prompt_text, response_text = _format_prompt_and_response(tokenizer, prompt, response, args)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]

    if not response_ids:
        raise ValueError("Encountered an empty response after tokenization.")

    prompt_ids, response_ids, prompt_truncated, response_truncated = _truncate_prompt(
        prompt_ids,
        response_ids,
        args.max_len,
        args.prompt_truncation_side,
    )
    input_ids = prompt_ids + response_ids
    return {
        "input_ids": input_ids,
        "prompt_len": len(prompt_ids),
        "response_len": len(response_ids),
        "total_len": len(input_ids),
        "prompt_truncated": prompt_truncated,
        "response_truncated": response_truncated,
    }


def _pad_encoded_batch(encoded_batch, tokenizer, device):
    max_len = max(len(item["input_ids"]) for item in encoded_batch)
    pad_token_id = tokenizer.pad_token_id
    input_ids = []
    attention_mask = []
    prompt_lens = []
    for item in encoded_batch:
        ids = item["input_ids"]
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_token_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
        prompt_lens.append(item["prompt_len"])

    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(attention_mask, dtype=torch.long, device=device),
        prompt_lens,
    )


def _get_batch_logps(logits, labels, attention_mask, prompt_lens):
    if logits.shape[:-1] != labels.shape:
        raise ValueError(f"Logits shape {logits.shape[:-1]} does not match labels shape {labels.shape}.")

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]

    loss_masks = attention_mask.clone().bool()
    for mask, prompt_len in zip(loss_masks, prompt_lens):
        mask[:prompt_len] = False
    loss_masks = loss_masks[:, 1:]

    labels[~loss_masks] = 0
    vocab_size = logits.size(-1)
    if (labels < 0).any() or (labels >= vocab_size).any():
        raise ValueError("Token IDs are outside the model vocabulary. Check tokenizer/model compatibility.")

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    token_counts = loss_masks.sum(-1)
    logps_sum = (per_token_logps * loss_masks).sum(-1)
    logps_mean = logps_sum / token_counts.clamp(min=1)
    return logps_sum, logps_mean, token_counts


def _score_encoded_batch(model, tokenizer, encoded_batch):
    device = next(model.parameters()).device
    input_ids, attention_mask, prompt_lens = _pad_encoded_batch(encoded_batch, tokenizer, device)
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
    return _get_batch_logps(output.logits.float(), input_ids, attention_mask, prompt_lens)


def _rows_from_batch(batch, start_index, scores, mean_scores, token_counts, encoded_batch, args):
    rows = []
    num_examples = len(batch[args.prompt_key])
    scores = scores.view(num_examples, len(RESPONSE_KEYS)).cpu()
    mean_scores = mean_scores.view(num_examples, len(RESPONSE_KEYS)).cpu()
    token_counts = token_counts.view(num_examples, len(RESPONSE_KEYS)).cpu()

    for i in range(num_examples):
        values = {key: float(scores[i, j].item()) for j, key in enumerate(RESPONSE_KEYS)}
        mean_values = {key: float(mean_scores[i, j].item()) for j, key in enumerate(RESPONSE_KEYS)}
        counts = {key: int(token_counts[i, j].item()) for j, key in enumerate(RESPONSE_KEYS)}

        row = {
            "index": start_index + i,
            "chosen_logp": values["chosen"],
            "rejected_prime_logp": values["rejected_prime"],
            "rejected_double_prime_logp": values["rejected_double_prime"],
            "rejected_logp": values["rejected"],
            "chosen_mean_logp": mean_values["chosen"],
            "rejected_prime_mean_logp": mean_values["rejected_prime"],
            "rejected_double_prime_mean_logp": mean_values["rejected_double_prime"],
            "rejected_mean_logp": mean_values["rejected"],
            "chosen_response_tokens": counts["chosen"],
            "rejected_prime_response_tokens": counts["rejected_prime"],
            "rejected_double_prime_response_tokens": counts["rejected_double_prime"],
            "rejected_response_tokens": counts["rejected"],
            "logp_chosen_minus_rejected_prime": values["chosen"] - values["rejected_prime"],
            "logp_rejected_prime_minus_rejected_double_prime": values["rejected_prime"]
            - values["rejected_double_prime"],
            "logp_rejected_double_prime_minus_rejected": values["rejected_double_prime"] - values["rejected"],
            "logp_chosen_minus_rejected": values["chosen"] - values["rejected"],
        }
        row["component_gap_sum"] = (
            row["logp_chosen_minus_rejected_prime"]
            + row["logp_rejected_prime_minus_rejected_double_prime"]
            + row["logp_rejected_double_prime_minus_rejected"]
        )

        if args.include_text:
            row[args.prompt_key] = batch[args.prompt_key][i]
            for key in RESPONSE_KEYS:
                row[key] = batch[key][i]

        offset = i * len(RESPONSE_KEYS)
        for j, key in enumerate(RESPONSE_KEYS):
            encoded = encoded_batch[offset + j]
            row[f"{key}_total_tokens"] = encoded["total_len"]
            row[f"{key}_prompt_truncated"] = encoded["prompt_truncated"]
            row[f"{key}_response_truncated"] = encoded["response_truncated"]

        rows.append(row)
    return rows


def evaluate(args):
    model, tokenizer = _load_model_and_tokenizer(args)
    dataset = _load_split(args)

    rows = []
    for start in tqdm(range(0, len(dataset), args.batch_size), desc="Scoring decomposed TLDR"):
        end = min(start + args.batch_size, len(dataset))
        batch = dataset[start:end]
        encoded_batch = []
        for i in range(end - start):
            prompt = batch[args.prompt_key][i]
            for response_key in RESPONSE_KEYS:
                encoded_batch.append(_encode_example(tokenizer, prompt, batch[response_key][i], args))

        scores, mean_scores, token_counts = _score_encoded_batch(model, tokenizer, encoded_batch)
        rows.extend(_rows_from_batch(batch, start, scores, mean_scores, token_counts, encoded_batch, args))

    df = pd.DataFrame(rows)
    out_dir = os.path.dirname(args.result_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.result_path, index=False)

    summary_cols = [
        "logp_chosen_minus_rejected_prime",
        "logp_rejected_prime_minus_rejected_double_prime",
        "logp_rejected_double_prime_minus_rejected",
        "logp_chosen_minus_rejected",
    ]
    print(f"Saved {len(df)} rows to {args.result_path}")
    print(df[summary_cols].mean().to_string())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="XueyingJia/qwen2.5-1.5B-Instruct-tldr-ours")
    parser.add_argument("--base_model_name_or_path", type=str, default=None)
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--dataset_name_or_path", type=str, default="Bojian92/tldr_preference_decomposed")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--prompt_key", type=str, default="prompt")
    parser.add_argument("--result_path", type=str, default="results/core_v1_conservative/decomposed_tldr_logps.csv")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument("--prompt_truncation_side", choices=["left", "right"], default="right")
    parser.add_argument("--apply_chat_template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--add_eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_text", action="store_true", default=False)
    parser.add_argument("--torch_dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--no_trust_remote_code", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())

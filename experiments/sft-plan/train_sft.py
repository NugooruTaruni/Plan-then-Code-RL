#!/usr/bin/env python3
"""
train_sft.py — SFT-only baseline: LoRA fine-tune Qwen2.5-Coder-3B-Instruct on the
plan-annotated MBPP-full train set (data/sft_train.jsonl from annotate_plans.py). No RL.

LoRA config matches the GRPO runs (rank 16, alpha 32, all linear layers). Loss is on the
assistant response only (<plan>...</plan> <code>...</code>); the prompt is the Optional
Plan prompt, so the model is evaluated with eval/optional_plan/vllm_eval.py as-is.

Usage (from this folder):
  python train_sft.py --out checkpoints/sft --merged-dir merged
"""

import os, json, math, argparse

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Trainer, TrainingArguments)

BASE = "Qwen/Qwen2.5-Coder-3B-Instruct"
HERE = os.path.dirname(os.path.abspath(__file__))


def tokenize(tok, max_len):
    def _f(ex):
        user = [{"role": "user", "content": ex["prompt"]}]
        prompt_txt = tok.apply_chat_template(user, add_generation_prompt=True, tokenize=False)
        full_txt = tok.apply_chat_template(
            user + [{"role": "assistant", "content": ex["response"]}], tokenize=False)
        assert full_txt.startswith(prompt_txt), "chat template prefix mismatch"
        prompt_ids = tok(prompt_txt, add_special_tokens=False)["input_ids"]
        full_ids = tok(full_txt, add_special_tokens=False)["input_ids"]
        assert full_ids[:len(prompt_ids)] == prompt_ids, "token prefix mismatch"
        full_ids = full_ids[:max_len]
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids),
                "labels": labels[:len(full_ids)]}
    return _f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",       default=os.path.join(HERE, "data", "sft_train.jsonl"))
    ap.add_argument("--out",        default=os.path.join(HERE, "checkpoints", "sft"))
    ap.add_argument("--merged-dir", default=os.path.join(HERE, "merged"))
    ap.add_argument("--epochs",     type=float, default=3)
    ap.add_argument("--lr",         type=float, default=1e-4)
    ap.add_argument("--max-len",    type=int, default=2048)
    ap.add_argument("--seed",       type=int, default=0)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    ds = Dataset.from_list(rows).map(tokenize(tok, a.max_len),
                                     remove_columns=["task_id", "prompt", "response"])
    print(f"{len(ds)} training examples")

    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    total_steps = math.ceil(len(ds) / 8) * math.ceil(a.epochs)
    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=4, gradient_accumulation_steps=2,   # effective batch 8, as GRPO
        lr_scheduler_type="cosine", warmup_steps=max(1, round(0.03 * total_steps)), bf16=True,
        logging_steps=5, save_strategy="epoch", save_total_limit=1,
        report_to="none", seed=a.seed,
    )
    Trainer(model=model, args=args, train_dataset=ds,
            data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)).train()

    model.save_pretrained(os.path.join(a.out, "lora_adapter"))
    merged = model.merge_and_unload()
    merged.save_pretrained(a.merged_dir)
    tok.save_pretrained(a.merged_dir)
    print("merged ->", a.merged_dir)


if __name__ == "__main__":
    main()

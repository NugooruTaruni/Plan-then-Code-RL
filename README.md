# Plan-then-Code: Learning Adaptive Planning via RL with Dual Rewards

**DS-GA 3001 Reinforcement Learning · NYU Center for Data Science · Spring 2026**  
**Team No Free Lunch Policy** · Anagha Radhakrishna Palandye · Taruni Nugooru · Mehul Goel

---

## Overview

Plan-then-Code trains a code LLM (Qwen2.5-Coder-3B-Instruct) to optionally generate a structured plan before writing code, using GRPO with a dual reward signal:

```
R = (α × plan_score × test_pass) + (β × test_pass)
```

The plan is scored by a frozen LLM judge (GPT-4o-mini) across 4 dimensions — Decomposition, Edge Cases, Approach, and Clarity. Code correctness is verified by unit tests (RLVR). The plan reward is **conditional on code passing tests**, preventing reward hacking.

### Research Hypotheses

| # | Hypothesis | Metric |
|---|---|---|
| H1 | Dual rewards beat code-only RLVR on pass@1 | pass@1 on MBPP and HumanEval+ |
| H2 | Agent plans more on hard tasks, skips on easy | Planning frequency vs task difficulty |
| H3 | Plan quality correlates with test outcomes | Pearson r between plan score and test pass |

---

## Results

| Model | MBPP pass@1 | MBPP +feedback (self repair) | MBPP +feedback (GPT-4o-mini judge) | MBPP pass@5 | HumanEval+ pass@1 | HumanEval+ +feedback (self repair) | HumanEval+ +feedback (GPT-4o-mini judge) | HumanEval+ pass@5 |
|---|---|---|---|---|---|---|---|---|
| Base Model | 57.6% | — | — | — | 71.5% | — | — | — |
| Plan-No-Reward *(ablation)* | 57.4% | — | — | — | — | — | — | — |
| Forced Plan (step 140) | 60.6% | 61.2% | 68.6% | **74.4%** | 73.8% | 76.8% | 79.9% | 87.2% |
| **Optional Plan (step 185)** ★ | **61.0%** | **67.2%** | **68.8%** | 73.1% | **78.0%** | **81.7%** | **85.4%** | **89.7%** |

★ Best overall model · **bold** = best in column

Optional Plan (trained on MBPP-full, 374 tasks) is best on 7 of 8 metrics; Forced Plan remains best on MBPP pass@5 (74.4% vs 73.1%). Step 185 was chosen from the top-5 checkpoints by mean pass@1 over MBPP and HumanEval+ — logs for the other four are in `eval/optional_plan/logs/other_top_steps/`. Optional Plan was evaluated with `eval/optional_plan/vllm_eval.py`; Base and Forced Plan with the original eval scripts.

### Extra Experiments: is it the RL?

Two no-RL baselines on the same base model, evaluated on the same tasks and tests with the same eval code as Optional Plan (seed 0).

| Model | MBPP pass@1 | MBPP +feedback (self repair) | MBPP +feedback (GPT-4o-mini judge) | MBPP pass@5 | HumanEval+ pass@1 | HumanEval+ +feedback (self repair) | HumanEval+ +feedback (GPT-4o-mini judge) | HumanEval+ pass@5 |
|---|---|---|---|---|---|---|---|---|
| Few-shot Plan-then-Code *(base, 3-shot, no RL)* | 58.4% | 64.8% | **70.4%** | **74.5%** | 76.8% | **82.9%** | **85.4%** | 87.7% |
| SFT-only *(plan-annotated MBPP train, no RL)* | 54.0% | 60.6% | 65.2% | 72.3% | 62.2% | 67.1% | 75.0% | 79.4% |
| **Optional Plan (step 185, GRPO)** ★ | **61.0%** | **67.2%** | 68.8% | 73.1% | **78.0%** | 81.7% | **85.4%** | **89.7%** |

**Q: Does RL training add anything over just prompting the base model to plan?**
Yes, mainly in first-attempt accuracy. GRPO is ahead on pass@1 (+2.6 on MBPP, +1.2 on HumanEval+), on MBPP self-repair (+2.4) and on HumanEval+ pass@5 (+2.0). The 3-shot prompted base model is ahead on MBPP judge feedback (+1.6), MBPP pass@5 (+1.4) and HumanEval+ self-repair (+1.2), and the two tie on HumanEval+ judge feedback. So RL makes the model right more often on the first try. Given feedback or several samples, prompting the base model to plan gets about as far.

**Q: Do the gains come from supervised training on plans, or from GRPO?**
From GRPO. SFT on 373 plan-annotated MBPP train tasks (the same tasks GRPO trains on: GPT-4o-mini plans plus the MBPP reference solutions) is the weakest model on all 8 metrics. It even falls below the prompted base model (MBPP pass@1 54.0% vs 58.4%; HumanEval+ 62.2% vs 76.8%). Its failures are wrong answers, not format errors: it copies MBPP's short reference solutions, which miss HumanEval+'s edge cases. Reward-driven GRPO, by contrast, improves both benchmarks.

All three models plan on ≥98.8% of tasks. Scores use fence-aware code extraction ([`eval/fence_rescore.py`](eval/fence_rescore.py)): an answer that opens `<code>` but closes its code with a Markdown fence instead of `</code>` is cut at the fence and re-tested, rather than failing on the trailing explanation. This only changes the few-shot model, by up to +2.4 points; the trained models' scores are identical under strict scoring (`summary_fence_aware.json` in each logs folder has both). Results are from a single seed, so gaps of about 1 point are within noise. Code: [`experiments/fewshot-plan/`](experiments/fewshot-plan/), [`experiments/sft-plan/`](experiments/sft-plan/); logs: [`eval/fewshot_plan/logs/`](eval/fewshot_plan/logs/), [`eval/sft_plan/logs/`](eval/sft_plan/logs/).

---

## Repository Structure

```
plan-then-code-rl/
│
├── README.md
│
├── experiments/
│   ├── forced-plan/
│   │   ├── exp1_run_training.sh
│   │   ├── exp1_reward_fn.py
│   │   ├── exp1_convert_to_parquet.py
│   │   ├── exp1_convert_humaneval_to_parquet.py
│   │   └── data/
│   │       ├── exp1_full_train.parquet
│   │       ├── exp1_full_val.parquet
│   │       ├── exp1_full_test.parquet
│   │       └── humaneval_plus_test.parquet
│   │
│   ├── optional-plan/
│   │   ├── train.sh
│   │   ├── reward_fn.py
│   │   ├── convert_to_parquet.py
│   │   └── code_cleaner.py
│   │
│   ├── plan_no_reward/
│   │   ├── run_training.sh
│   │   ├── reward_fn.py
│   │   ├── convert_to_parquet_mbpp.py
│   │   ├── convert_to_parquet_humaneval.py
│   │   └── data/
│   │       ├── mbpp_train_full.parquet
│   │       ├── mbpp_val_full.parquet
│   │       ├── mbpp_test_full.parquet
│   │       └── humaneval_plus_test_full.parquet
│   │
│   ├── fewshot-plan/                  # extra experiment: few-shot baseline (no RL)
│   │   └── fewshot_prompts.py
│   │
│   ├── sft-plan/                      # extra experiment: SFT-only baseline (no RL)
│   │   ├── annotate_plans.py
│   │   ├── train_sft.py
│   │   └── data/
│   │       └── sft_train.jsonl
│   │
│   └── merge_checkpoint.py
│
├── eval/
│   ├── fence_rescore.py               # fence-aware re-scoring of saved logs (no GPU)
│   ├── plan_rescore.py                # re-scores plan quality (GPT-4o-mini) in judge-feedback logs
│   ├── forced_plan/
│   │   ├── exp1_evaluate_upd2.py
│   │   ├── exp1_evaluate_feedback_upd.py
│   │   ├── exp1_evaluate_feedback_upd_humaneval.py
│   │   ├── exp1_eval_passk_mbpp.py
│   │   ├── exp1_eval_passk_humaneval.py
│   │   └── logs/
│   │       ├── Eval_MBPP_Baseline/
│   │       ├── Eval_MBPP_pass@1/
│   │       ├── Eval_MBPP_pass@5/
│   │       ├── Eval_MBPP_Feedback_NoJudge/
│   │       ├── Eval_MBPP_Feedback_W_Judge/
│   │       ├── Eval_Humaneval_Baseline/
│   │       ├── Eval_Humaneval_pass@1/
│   │       ├── Eval_Humaneval_pass@5/
│   │       ├── Eval_Humaneval_Feedback_NoJudge/
│   │       └── Eval_Humaneval_Feedback_W_Judge/
│   │
│   ├── optional_plan/
│   │   ├── vllm_eval.py
│   │   ├── run_all_mbpp.py
│   │   ├── run_all_humaneval.py
│   │   ├── runfix.py
│   │   ├── code_cleaner.py
│   │   └── logs/                      # step 185
│   │       ├── summary_mbpp.json
│   │       ├── summary_humaneval.json
│   │       ├── eval.log
│   │       ├── mbpp_pass1/
│   │       ├── mbpp_pass5/
│   │       ├── mbpp_feedback_judge/
│   │       ├── mbpp_feedback_nojudge/
│   │       ├── humaneval_pass1/
│   │       ├── humaneval_pass5/
│   │       ├── humaneval_feedback_judge/
│   │       ├── humaneval_feedback_nojudge/
│   │       └── other_top_steps/
│   │           ├── step140/
│   │           ├── step175/
│   │           ├── step180/
│   │           └── step190/
│   │
│   ├── fewshot_plan/
│   │   ├── fewshot_eval.py
│   │   └── logs/                      # same layout as optional_plan/logs
│   │
│   ├── sft_plan/
│   │   └── logs/                      # evaluated with optional_plan/vllm_eval.py
│   │
│   └── plan_no_reward/
│       ├── eval_feedback_mbpp_optional_no_reward_pass1.py
│       ├── eval_feedback_mbpp_optional_no_reward_judge_and_no_judge.py
│       └── Logs/
│           ├── Training Logs/
│           ├── eval_mbpp_full__optional_no_reward_pass1/
│           └── eval_mbpp_full_optional_no_reward_judge_and_nojudge/
│
└── analysis/
    ├── parse_training_log.py
    ├── analyze_categories_of_tasks.py
    ├── analyze_training.py
    ├── h1.py
    ├── h2.py
    ├── h3.py
    └── plot_category_comparison.py
```

---

## Setup

```bash
# Clone repo
git clone https://github.com/palandyeanagha/Plan-then-Code-RL.git
cd Plan-then-Code-RL

# Install dependencies
pip install verl transformers torch vllm pandas numpy matplotlib scipy wandb

# Set API keys
export OPENAI_API_KEY="sk-..."
export WANDB_API_KEY="..."
```

---

## Data

Parquet files are included under each experiment's `data/` folder. To regenerate from source:

```python
from datasets import load_dataset
ds_mbpp = load_dataset("google-research-datasets/mbpp", "full")
ds_he   = load_dataset("evalplus/humanevalplus")
```

```bash
# Forced plan
python experiments/forced-plan/exp1_convert_to_parquet.py
python experiments/forced-plan/exp1_convert_humaneval_to_parquet.py

# Optional plan
python experiments/optional-plan/convert_to_parquet.py

# Plan-no-reward
python experiments/plan_no_reward/convert_to_parquet_mbpp.py
python experiments/plan_no_reward/convert_to_parquet_humaneval.py
```

---

## Training

### Forced Plan
```bash
export OPENAI_API_KEY="sk-..."
bash experiments/forced-plan/exp1_run_training.sh
```

### Optional Plan
```bash
export OPENAI_API_KEY="sk-..."
bash experiments/optional-plan/train.sh
```

### Plan-No-Reward (Ablation)
```bash
bash experiments/plan_no_reward/run_training.sh
```

### SFT-only Baseline (no RL)
```bash
export OPENAI_API_KEY="sk-..."
cd experiments/sft-plan
python annotate_plans.py   # GPT-4o-mini plans + MBPP reference code → data/sft_train.jsonl
python train_sft.py        # LoRA SFT (r16, α32), 3 epochs, lr 1e-4 → merged/
```

**Key hyperparameters:**

| Parameter | Forced Plan | Optional Plan | Plan-No-Reward |
|---|---|---|---|
| Learning Rate | 3e-6 | 5e-6 | 3e-6 |
| KL Coefficient | 0.01 | 0.001 | 0.001 |
| Plan Reward α | 0.5 | 0.5 | 0 |
| Code Reward β | 1.0 | 1.0 | 1.0 |
| Rollouts (n) | 5 | 5 | 5 |
| Best Checkpoint | Step 140 | Step 185 | — |

---

## Evaluation

```bash
# pass@1 MBPP
python eval/forced_plan/exp1_evaluate_upd2.py \
    --checkpoint <path> --split test

# pass@1 with feedback + judge
python eval/forced_plan/exp1_evaluate_feedback_upd.py \
    --checkpoint <path> --split test --judge gpt-4o-mini

# pass@k MBPP
python eval/forced_plan/exp1_eval_passk_mbpp.py \
    --checkpoint <path> --num-samples 5 --k-values 5

# pass@k HumanEval+
python eval/forced_plan/exp1_eval_passk_humaneval.py \
    --checkpoint <path> --num-samples 5 --k-values 5

# Optional plan — all 8 evals (pass@1, pass@5, +feedback self repair, +feedback judge) with vLLM
cd experiments/optional-plan
python convert_to_parquet.py              # MBPP-full train/val/test
python convert_to_parquet.py --humaneval  # HumanEval+
cd ../../eval/optional_plan
python vllm_eval.py --checkpoint TaruniNugooru/qwen2.5-coder-3b-optional-plan-full-step185 \
    --dataset mbpp --data ../../experiments/optional-plan/data/mbpp_test_optional_plan.parquet --outdir logs
python vllm_eval.py --checkpoint TaruniNugooru/qwen2.5-coder-3b-optional-plan-full-step185 \
    --dataset humaneval --data ../../experiments/optional-plan/data/humaneval_plan_optional_plan.parquet --outdir logs
```

Large per-task logs are stored gzipped (`*.jsonl.gz`); read them with `gzip -dc file.jsonl.gz`.

```bash
# Few-shot baseline — base model, 3-shot plan-then-code prompt (run from eval/fewshot_plan)
python fewshot_eval.py --dataset mbpp      --data ../../experiments/optional-plan/data/mbpp_test_optional_plan.parquet      --outdir logs
python fewshot_eval.py --dataset humaneval --data ../../experiments/optional-plan/data/humaneval_plan_optional_plan.parquet --outdir logs

# SFT-only baseline — same vllm_eval.py as Optional Plan (run from eval/optional_plan)
python vllm_eval.py --checkpoint TaruniNugooru/qwen2.5-coder-3b-sft-plan --dataset mbpp \
    --data ../../experiments/optional-plan/data/mbpp_test_optional_plan.parquet --outdir ../sft_plan/logs
python vllm_eval.py --checkpoint TaruniNugooru/qwen2.5-coder-3b-sft-plan --dataset humaneval \
    --data ../../experiments/optional-plan/data/humaneval_plan_optional_plan.parquet --outdir ../sft_plan/logs
```

---

## Analysis

```bash
# Parse training log → CSV
python analysis/parse_training_log.py \
    --log grpo_optional_training_full.log \
    --out training_curves_optional_plan.csv

# Category analysis
python analysis/analyze_categories_of_tasks.py \
    --step140 eval_step140_mbpp_pass@1.jsonl \
    --base    raw_Qwen2.5-Coder-3B-Instruct.jsonl \
    --out     category_analysis_forced_plan.json

# H1 / H2 / H3 plots
python analysis/h1.py --out plots_h1/

python analysis/h2.py \
    --optional optional_plan_mbpp_pass1.json \
    --base     raw_Qwen2.5-Coder-3B-Instruct.jsonl \
    --out      plots_h2/

python analysis/h3.py \
    --jsonl step_140_eval_feedback_w_judge_mbpp.jsonl \
    --out   plots_h3/

# 3-model category comparison
python analysis/plot_category_comparison.py \
    --forced     category_analysis_forced_plan.json \
    --optional   category_analysis_optional_plan.json \
    --base-jsonl raw_Qwen2.5-Coder-3B-Instruct.jsonl \
    --out        plots_comparison/
```

---

## Checkpoints

| Model | HuggingFace |
|---|---|
| Forced Plan (step 140) | [palandyeanagha/qwen2.5-coder-3b-plan-then-code-step140](https://huggingface.co/palandyeanagha/qwen2.5-coder-3b-plan-then-code-step140) |
| Forced Plan (step 200) | [palandyeanagha/qwen2.5-coder-3b-plan-then-code-step200](https://huggingface.co/palandyeanagha/qwen2.5-coder-3b-plan-then-code-step200) |
| **Optional Plan (step 185)** ★ | [TaruniNugooru/qwen2.5-coder-3b-optional-plan-full-step185](https://huggingface.co/TaruniNugooru/qwen2.5-coder-3b-optional-plan-full-step185) |
| Optional Plan (step 180) | [TaruniNugooru/qwen2.5-coder-3b-optional-plan-full-step180](https://huggingface.co/TaruniNugooru/qwen2.5-coder-3b-optional-plan-full-step180) |
| Optional Plan (step 175) | [TaruniNugooru/qwen2.5-coder-3b-optional-plan-full-step175](https://huggingface.co/TaruniNugooru/qwen2.5-coder-3b-optional-plan-full-step175) |
| Optional Plan (step 190) | [TaruniNugooru/qwen2.5-coder-3b-optional-plan-full-step190](https://huggingface.co/TaruniNugooru/qwen2.5-coder-3b-optional-plan-full-step190) |
| Optional Plan (step 140) | [TaruniNugooru/qwen2.5-coder-3b-optional-plan-full-step140](https://huggingface.co/TaruniNugooru/qwen2.5-coder-3b-optional-plan-full-step140) |
| Optional Plan — LoRA adapters (all steps) | [TaruniNugooru/plan-then-code-optional-full-adapters](https://huggingface.co/TaruniNugooru/plan-then-code-optional-full-adapters) |
| SFT-only baseline (no RL) | [TaruniNugooru/qwen2.5-coder-3b-sft-plan](https://huggingface.co/TaruniNugooru/qwen2.5-coder-3b-sft-plan) |

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model     = AutoModelForCausalLM.from_pretrained("TaruniNugooru/qwen2.5-coder-3b-optional-plan-full-step185")
tokenizer = AutoTokenizer.from_pretrained("TaruniNugooru/qwen2.5-coder-3b-optional-plan-full-step185")
```

To merge and upload your own checkpoint:
```bash
python experiments/merge_checkpoint.py \
    --step 140 --upload \
    --hf-repo palandyeanagha/qwen2.5-coder-3b-plan-then-code-step140
```

---

## Compute

- **Hardware:** 1× NVIDIA A100 80GB (Lightning.ai; Vast.ai for Optional Plan)
- **Framework:** VERL + vLLM (rollouts) + HuggingFace Transformers (eval); vLLM for Optional Plan eval
- **Tracking:** W&B (`plan-then-code` project)
- **Training time:** ~8 hours per experiment

---

## Citation

```bibtex
@misc{planthencode2026,
  title  = {Plan-then-Code: Learning Adaptive Planning via RL with Dual Rewards},
  author = {Palandye, Anagha Radhakrishna and Nugooru, Taruni and Goel, Mehul},
  year   = {2026},
  note   = {DS-GA 3001 Final Project, NYU Center for Data Science},
  url    = {https://github.com/palandyeanagha/Plan-then-Code-RL}
}
```

#!/bin/bash
# Plan-then-Code: Experiment 1 — All tests + function name prompt
# Changes vs baseline run_training.sh:
#   - data files: exp1_full_train.parquet / exp1_full_val.parquet (374 train tasks)
#   - max_prompt_length: 768 → 1024  (headroom for 3 test cases)
#   - experiment_name: exp1-full-train
#   - reward_fn: exp1_reward_fn.py

set -e

export WANDB_PROJECT="plan-then-code"
export WANDB_RUN_NAME="exp1-full-train"
export RAY_memory_usage_threshold=0.98
export RAY_memory_monitor_refresh_ms=2000

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set!"
    echo "  export OPENAI_API_KEY='sk-...'"
    exit 1
fi

mkdir -p results checkpoints logs

# Count actual tasks in parquet
TRAIN_TASKS=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('data/exp1_full_train.parquet')))" 2>/dev/null || echo "~56")
VAL_TASKS=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('data/exp1_full_val.parquet')))" 2>/dev/null || echo "90")

echo "============================================================"
echo " Plan-then-Code GRPO Training — Experiment 1"
echo " Prompt: all 3 test cases + explicit function name"
echo " Train data : data/exp1_full_train.parquet (${TRAIN_TASKS} tasks, 100% of full)"
echo " Val data   : data/exp1_full_val.parquet   (${VAL_TASKS} tasks, full)"
echo " GPU memory : $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) MiB"
echo "============================================================"

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files=data/exp1_full_train.parquet \
    data.val_files=data/exp1_full_val.parquet \
    data.train_batch_size=8 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    "data.truncation=error" \
    data.dataloader_num_workers=2 \
    \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-Coder-3B-Instruct \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.use_remove_padding=True \
    \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.use_torch_compile=false \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.use_torch_compile=false \
    \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n_gpus_per_node=1 \
    \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=5 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_local_dir=checkpoints/exp1_full \
    +trainer.remove_previous_ckpt_in_save=False \
    'trainer.logger=["console","wandb"]' \
    trainer.project_name=plan-then-code \
    trainer.experiment_name=exp1-full-train \
    \
    reward.custom_reward_function.path=exp1_reward_fn.py \
    reward.custom_reward_function.name=compute_score \
    reward.num_workers=1

echo "Training complete. Checkpoints in checkpoints/exp1_full/"

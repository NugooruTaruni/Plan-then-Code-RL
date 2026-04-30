#!/bin/bash
# Plan-then-Code: Forced-Plan GRPO Training
# Usage: bash run_training.sh 2>&1 | tee training_log.txt

set -x

export WANDB_PROJECT="plan-then-code"
export WANDB_RUN_NAME="optional-no-reward-mbpp-full-v1"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set!"
    echo "  export OPENAI_API_KEY='sk-...'"
    exit 1
fi

mkdir -p results checkpoints

python -m verl.trainer.main_ppo algorithm.adv_estimator=grpo data.train_files=data/mbpp_train_full.parquet data.val_files=data/mbpp_val_full.parquet data.train_batch_size=8 data.max_prompt_length=512 data.max_response_length=1024 data.filter_overlong_prompts=True "data.truncation=error" data.dataloader_num_workers=1 actor_rollout_ref.model.path=Qwen/Qwen2.5-Coder-3B-Instruct actor_rollout_ref.model.lora_rank=16 actor_rollout_ref.model.lora_alpha=32 actor_rollout_ref.actor.optim.lr=3e-6 actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.actor.ppo_mini_batch_size=8 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=5 actor_rollout_ref.rollout.temperature=0.7 actor_rollout_ref.rollout.gpu_memory_utilization=0.5 actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.enforce_eager=False actor_rollout_ref.rollout.free_cache_engine=True actor_rollout_ref.rollout.n_gpus_per_node=1 trainer.n_gpus_per_node=1 trainer.nnodes=1 reward.custom_reward_function.path=reward_fn.py reward.custom_reward_function.name=compute_score reward.num_workers=1 trainer.total_epochs=5 trainer.save_freq=20 trainer.test_freq=20 trainer.default_local_dir=checkpoints/ 'trainer.logger=["console","wandb"]' trainer.project_name=plan-then-code trainer.experiment_name=optional-no-reward-mbpp-full-v1
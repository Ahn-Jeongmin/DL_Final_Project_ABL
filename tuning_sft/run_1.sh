#!/bin/bash
#SBATCH --job-name=slurm_test
#SBATCH --nodelist=server2
#SBATCH --output=slurm_logs/training_%j.out
#SBATCH --error=slurm_logs/training_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00

# 프로젝트 디렉토리로 이동
# cd /home/mindw/slurm_test

# UV 가상환경 활성화
source .venv/bin/activate

# 환경 확인
echo "Python path: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# 실제 작업 실행
# CUDA_VISIBLE_DEVICES=5,6 \
torchrun \
    --master-port 29501 \
    --nproc_per_node=1 \
    main_train.py \
        --trainer_seed 42 \
        --trainer_strategy zero_2 \
        --trainer_precision 'bf16-mixed' \
        --trainer_n_gpu 1 \
        --trainer_train_batch_size 4 \
        --trainer_eval_batch_size 8 \
        --trainer_val_check_interval 0.25 \
        --trainer_log_root logs_0 \
        --optimize_learning_rate 5e-5 \
        --optimize_warmup_steps 50 \
        --optimize_num_train_epochs 3 \
        --optimize_gradient_accumulation_steps 1 \
        --model_config_pretrained_model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
        --model_config_lora_r 8 \
        --model_config_lora_a 16 \
        --dataset_system_prompt_version 1 \
        --dataset_entity_description 0 \
        --debug_test1000 0 \
        --valid_gen_config modeldefault

        # --trainer_init_ckpt 'logs_1/checkpoints/version_73/clean_epoch=1-step=32000-avg_train_loss=0.714204.ckpt' \
        # --n_train 1024000 \
        # --n_valid 4096 \

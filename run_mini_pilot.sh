#!/usr/bin/env bash
set -euo pipefail

# Submit once through scripts/submit_mini_pilot.py; CUDA never runs on a login node.
: "${SLURM_JOB_ID:?An explicit one-GPU Slurm allocation is required}"
: "${PILOT_OUTPUT_DIR:?A unique outputs/pilot_run_* directory is required}"
PROJECT_DIR="/storage/home/ydk297/projects/poisoning"
cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
# Slurm supplies CUDA_VISIBLE_DEVICES; preserve the allocated device mapping.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

.venv-editor/bin/python train_and_evaluate_pilot.py \
    --dataset_path data/vqav2_mini_32_16_16 \
    --output_dir "$PILOT_OUTPUT_DIR" \
    --num_train_images 32 --num_val_images 16 --num_test_images 16 \
    --lora_rank 8 --base_learning_rate 1e-5 --force_fp32_lora True \
    --max_inner_steps 100 --rsi_rounds 3 --seed 42 --precision fp32 \
    --methods frozen fixed ddpo_only rule_rsi rl_rsi

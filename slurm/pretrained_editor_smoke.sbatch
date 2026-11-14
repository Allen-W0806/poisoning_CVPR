#!/bin/bash
#SBATCH --job-name=p2-editor
#SBATCH --partition=epyc-gpu
#SBATCH --account=allusers
#SBATCH --gres=gpu:tesla_a100_80gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:15:00
# Submit through scripts/submit_phase2.py for unique logs and the cumulative budget guard.
set -Eeuo pipefail
cd /storage/home/ydk297/projects/poisoning
source .venv-editor/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME=/storage/home/ydk297/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
python scripts/smoke_pretrained_editor.py

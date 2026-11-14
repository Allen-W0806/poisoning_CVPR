#!/bin/bash
#SBATCH --job-name=poison-smoke
#SBATCH --partition=epyc-gpu
#SBATCH --account=allusers
#SBATCH --gres=gpu:tesla_a100_80gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
set -Eeuo pipefail
cd /storage/home/ydk297/projects/poisoning
source .venv/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME=/storage/home/ydk297/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
trap 'rc=$?; python scripts/finalize_job.py "$rc"; exit "$rc"' EXIT
hostname
nvidia-smi
python scripts/run_smoke_suite.py --device cuda --include-qwen

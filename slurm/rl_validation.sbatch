#!/bin/bash
#SBATCH --job-name=poison-rl
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
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
trap 'rc=$?; python scripts/finalize_job.py "$rc"; exit "$rc"' EXIT
nvidia-smi
python scripts/smoke_rl.py --device cuda

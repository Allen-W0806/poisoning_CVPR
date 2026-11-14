#!/bin/bash
#SBATCH --job-name=nvq-frozen
#SBATCH --account=allusers
#SBATCH --partition=epyc-gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --comment=overnight_benign_20261112T051015Z:frozen:0
#SBATCH --chdir=/storage/home/ydk297/projects/poisoning/outputs/overnight_benign_20261112T051015Z/code
#SBATCH --output=/storage/home/ydk297/projects/poisoning/outputs/overnight_benign_20261112T051015Z/logs/%x-%j.log
#SBATCH --error=/storage/home/ydk297/projects/poisoning/outputs/overnight_benign_20261112T051015Z/logs/%x-%j.log
#SBATCH --gres=gpu:tesla_a100_80gb:1
set -euo pipefail
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HOME=/storage/home/ydk297/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p /storage/home/ydk297/projects/poisoning/tmp/overnight_benign_20261112T051015Z/frozen_0
export TMPDIR=/storage/home/ydk297/projects/poisoning/tmp/overnight_benign_20261112T051015Z/frozen_0
trap '/storage/home/ydk297/projects/poisoning/.venv-editor/bin/python -c '"'"'from pathlib import Path;import shutil;p=Path('"'"'"'"'"'"'"'"'/storage/home/ydk297/projects/poisoning/tmp/overnight_benign_20261112T051015Z/frozen_0'"'"'"'"'"'"'"'"').resolve();r=Path('"'"'"'"'"'"'"'"'/storage/home/ydk297/projects/poisoning/tmp/overnight_benign_20261112T051015Z'"'"'"'"'"'"'"'"').resolve();assert p.is_relative_to(r) and p!=r;shutil.rmtree(p,ignore_errors=True)'"'"'' EXIT
/storage/home/ydk297/projects/poisoning/.venv-editor/bin/python -B /storage/home/ydk297/projects/poisoning/outputs/overnight_benign_20261112T051015Z/code/scripts/night_vqa_worker.py --campaign /storage/home/ydk297/projects/poisoning/outputs/overnight_benign_20261112T051015Z --stage frozen --attempt 0

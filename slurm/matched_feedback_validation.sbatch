#!/bin/bash
#SBATCH --job-name=matched-benign-feedback
#SBATCH --partition=epyc-gpu
#SBATCH --account=allusers
#SBATCH --gres=gpu:tesla_a100_80gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:10:00
set -Eeuo pipefail
cd /storage/home/ydk297/projects/poisoning
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME=/storage/home/ydk297/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
source .venv-editor/bin/activate
python -B scripts/run_matched_feedback.py --run-id "$1" --phase "$2"
if [ "$2" = "pilot" ]; then
  python -B scripts/validate_ddpo_wiring.py --run-id "${1}_tiny_checkpoint"
fi

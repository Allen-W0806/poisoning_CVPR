#!/bin/bash
#SBATCH --partition=epyc-gpu
#SBATCH --account=allusers
#SBATCH --gres=gpu:tesla_a100_80gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
set -Eeuo pipefail
cd /storage/home/ydk297/projects/poisoning
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME=/storage/home/ydk297/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
source .venv-editor/bin/activate
if [ "$1" = "qwen" ]; then
  python -B scripts/run_qwen_replication.py --run-id "$2" --phase main
elif [ "$1" = "ip2pix_instrumented" ]; then
  python -B scripts/run_ip2pix_checkpoint_instrumented.py --run-id "$2"
else
  python -B scripts/run_ip2pix_checkpoint_diagnostic.py --run-id "$2"
fi

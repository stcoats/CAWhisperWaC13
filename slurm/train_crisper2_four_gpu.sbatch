#!/bin/bash -l
#SBATCH --job-name=ca_crisper2_lora
#SBATCH --partition=gpumedium
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:gh200:4
#SBATCH --mem=300G
#SBATCH --output=logs/train_%x_%j.out
#SBATCH --error=logs/train_%x_%j.err

set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to this repository}"
: "${PROJECT_DATA:?Set PROJECT_DATA to the root used by manifest audio paths}"
: "${MANIFEST:?Set MANIFEST to the two-source verbatim JSONL manifest}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for the adapter and trainer state}"

if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  source "${VENV_ACTIVATE}"
fi

export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN

mkdir -p "${OUTPUT_DIR}" "${PROJECT_ROOT}/logs"

torchrun --standalone --nproc_per_node=4 \
  "${PROJECT_ROOT}/scripts/train/train_crisper2_lora.py" \
  --project "${PROJECT_DATA}" \
  --manifest "${MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --model nyralabs/CrisperWhisper2.0_large \
  --decoder-policy crisper_verbatim \
  --learning-rate 1e-4 \
  --epochs 10 \
  --batch-size 2 \
  --eval-batch-size 2 \
  --gradient-accumulation 2 \
  --warmup-ratio 0.1 \
  --weight-decay 0.01 \
  --early-stopping-patience 3 \
  --num-workers 0

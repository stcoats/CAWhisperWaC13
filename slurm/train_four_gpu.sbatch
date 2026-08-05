#!/bin/bash -l
#SBATCH --job-name=ca_whisper_train
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
: "${MANIFEST:?Set MANIFEST to the paired-policy JSONL manifest}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for checkpoints and trainer state}"
: "${TRAINING_POLICY:?Set TRAINING_POLICY to dual or verbatim}"
: "${OPTIMIZATION:?Set OPTIMIZATION to lora or full}"

if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  source "${VENV_ACTIVATE}"
fi

case "${OPTIMIZATION}:${TRAINING_POLICY}" in
  lora:dual) learning_rate=1e-4 ;;
  full:dual|full:verbatim) learning_rate=1e-5 ;;
  *) echo "Unsupported condition: ${OPTIMIZATION}:${TRAINING_POLICY}" >&2; exit 2 ;;
esac

export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN

mkdir -p "${OUTPUT_DIR}" "${PROJECT_ROOT}/logs"

torchrun --standalone --nproc_per_node=4 \
  "${PROJECT_ROOT}/scripts/train/train_whisper.py" \
  --manifest "${MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --model openai/whisper-large-v3 \
  --optimization "${OPTIMIZATION}" \
  --training-policy "${TRAINING_POLICY}" \
  --learning-rate "${learning_rate}" \
  --epochs 10 \
  --batch-size 2 \
  --eval-batch-size 2 \
  --gradient-accumulation 2 \
  --warmup-ratio 0.1 \
  --weight-decay 0.01 \
  --early-stopping-patience 3 \
  --num-workers 0

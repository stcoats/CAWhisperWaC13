#!/bin/bash -l
#SBATCH --job-name=ca_whisper_eval
#SBATCH --partition=gpumedium
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:gh200:1
#SBATCH --mem=120G
#SBATCH --output=logs/eval_%x_%j.out
#SBATCH --error=logs/eval_%x_%j.err

set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to this repository}"
: "${MANIFEST:?Set MANIFEST to the paired-policy JSONL manifest}"
: "${MODEL_PATH:?Set MODEL_PATH to a model or adapter directory}"
: "${SYSTEM_NAME:?Set SYSTEM_NAME for the output metadata}"
: "${CORPUS:?Set CORPUS to the manifest corpus label}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for predictions and metrics}"

if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  source "${VENV_ACTIVATE}"
fi

dual_flag=()
if [[ "${DUAL_POLICY:-0}" == 1 ]]; then
  dual_flag=(--dual-policy)
fi

mkdir -p "${OUTPUT_DIR}" "${PROJECT_ROOT}/logs"

python "${PROJECT_ROOT}/scripts/evaluate/evaluate_models.py" \
  --manifest "${MANIFEST}" \
  --corpus "${CORPUS}" \
  --system "${SYSTEM_NAME}" \
  --base-model "${MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  "${dual_flag[@]}"

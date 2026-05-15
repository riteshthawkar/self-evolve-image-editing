#!/usr/bin/env bash
#SBATCH --job-name=qedit-eval
#SBATCH --output=outputs/logs/slurm_%x_%j.out
#SBATCH --error=outputs/logs/slurm_%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-${SCRATCH:-$PWD/.cache}/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM=false

CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the LoRA .safetensors path}"
MODEL_NAME="${MODEL_NAME:-$(basename "${CHECKPOINT%.safetensors}")}"
LIMIT="${LIMIT:-0}"

cmd=(
  bash scripts/run_edit_model_suite.sh
  --model-type lora
  --checkpoint "$CHECKPOINT"
  --model-name "$MODEL_NAME"
  --skip-validate
)

if [[ "$LIMIT" != "0" ]]; then
  cmd+=(--limit "$LIMIT")
fi

"${cmd[@]}"

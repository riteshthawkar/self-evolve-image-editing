#!/usr/bin/env bash
#SBATCH --job-name=qedit-lora
#SBATCH --output=outputs/logs/slurm_%x_%j.out
#SBATCH --error=outputs/logs/slurm_%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=112G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-${SCRATCH:-$PWD/.cache}/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM=false

TRAIN_MANIFEST="${TRAIN_MANIFEST:?Set TRAIN_MANIFEST, for example outputs/self_evolve/<run>/delta-ranker/round_03/train_manifest.json}"
RUN_NAME="${RUN_NAME:-self_evolve_lora_$(date +%Y%m%d_%H%M%S)}"
EPOCHS="${EPOCHS:-3}"
MAX_PIXELS="${MAX_PIXELS:-1048576}"
LORA_RANK="${LORA_RANK:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"

bash scripts/train_lora_2509.sh \
  --set "name=$RUN_NAME" \
  --set dataset.dataset_base_path=. \
  --set "dataset.dataset_metadata_path=$TRAIN_MANIFEST" \
  --set "dataset.max_pixels=$MAX_PIXELS" \
  --set "dataset.dataset_num_workers=$NUM_WORKERS" \
  --set "training.num_epochs=$EPOCHS" \
  --set "training.gradient_accumulation_steps=$GRAD_ACCUM" \
  --set "lora.lora_rank=$LORA_RANK" \
  --set "output.output_path=outputs/checkpoints/$RUN_NAME" \
  --set "output.command_file=outputs/logs/${RUN_NAME}_command.txt"

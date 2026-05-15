#!/usr/bin/env bash
#SBATCH --job-name=qedit-evolve
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

MANIFEST="${MANIFEST:-data/unlabeled/splits/coco2017/main_manifest.jsonl}"
RUN_NAME="${RUN_NAME:-delta_ranker_main_$(date +%Y%m%d_%H%M%S)}"
LIMIT="${LIMIT:-256}"
SAMPLES_PER_PROPOSAL="${SAMPLES_PER_PROPOSAL:-2}"
MAX_RECORDS_PER_ROUND="${MAX_RECORDS_PER_ROUND:-256}"
NUM_ROUNDS="${NUM_ROUNDS:-3}"
DTYPE="${DTYPE:-bfloat16}"

bash scripts/run_self_evolve_matrix.sh \
  --variant delta-ranker \
  --dtype "$DTYPE" \
  --limit "$LIMIT" \
  --output-prefix "outputs/self_evolve/$RUN_NAME" \
  --set dataset.source=jsonl \
  --set "dataset.manifest_jsonl=$MANIFEST" \
  --set "candidate_generation.samples_per_proposal=$SAMPLES_PER_PROPOSAL" \
  --set "curriculum.max_records_per_round=$MAX_RECORDS_PER_ROUND" \
  --set "curriculum.num_rounds=$NUM_ROUNDS"

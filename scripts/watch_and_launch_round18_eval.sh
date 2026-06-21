#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/share_6/users/ritesh_thawkar/self-evolve-image-editing}"
RUN_DIR="${RUN_DIR:-$ROOT/outputs/self_evolve/qwen_edit_2509_conservative_pairwise_full_v1_20260604T183826}"
CHECKPOINT="${CHECKPOINT:-$RUN_DIR/round_18/training_output/pytorch_lora_weights.safetensors}"
SUMMARY="${SUMMARY:-$RUN_DIR/round_18/summary.json}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-180}"
TRAIN_TMUX_SESSION="${TRAIN_TMUX_SESSION:-uug_full_3d}"
STOP_TRAINING_AFTER_ROUND="${STOP_TRAINING_AFTER_ROUND:-1}"

PARTITION="${PARTITION:-gpu}"
GRES="${GRES:-gpu:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM="${MEM:-96G}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"
JOB_NAME="${JOB_NAME:-eval_r18_edit}"

cd "$ROOT"

echo "[watch-r18] Waiting for round 18 checkpoint and summary."
echo "[watch-r18] Checkpoint: $CHECKPOINT"
echo "[watch-r18] Summary:    $SUMMARY"

while [[ ! -f "$CHECKPOINT" || ! -f "$SUMMARY" ]]; do
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[watch-r18] $ts still waiting; next check in ${CHECK_INTERVAL_SECONDS}s."
  sleep "$CHECK_INTERVAL_SECONDS"
done

echo "[watch-r18] Round 18 checkpoint is ready."

if [[ "$STOP_TRAINING_AFTER_ROUND" == "1" ]]; then
  if tmux has-session -t "$TRAIN_TMUX_SESSION" 2>/dev/null; then
    echo "[watch-r18] Stopping training tmux session $TRAIN_TMUX_SESSION to free the GPU."
    tmux kill-session -t "$TRAIN_TMUX_SESSION"
    sleep 20
  else
    echo "[watch-r18] Training tmux session $TRAIN_TMUX_SESSION is not active."
  fi
fi

unset SLURM_JOB_ID SLURM_STEP_ID SLURM_JOB_NODELIST

echo "[watch-r18] Requesting Slurm allocation for ImgEdit and GEdit eval."
srun \
  --partition="$PARTITION" \
  --gres="$GRES" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --mem="$MEM" \
  --time="$TIME_LIMIT" \
  --job-name="$JOB_NAME" \
  bash "$ROOT/scripts/eval_round18_imgedit_gedit.sh"

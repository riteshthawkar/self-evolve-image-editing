#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${SLURM_JOB_ID:-}" && "${ALLOW_LOGIN_NODE:-0}" != "1" ]]; then
  echo "Refusing to run self-evolve outside a Slurm allocation. Start this inside an srun/sbatch resource session." >&2
  exit 1
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-/share_6/users/ritesh_thawkar/condaenvs/qedit/bin/python}"

exec "$PYTHON" -m qwen_edit_project.self_evolve.run_loop \
  --config configs/self_evolve/qwen_edit_2509_balanced_cepr_v2.yaml \
  "$@"

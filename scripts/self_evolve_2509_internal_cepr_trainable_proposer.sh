#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHONPATH="${PYTHONPATH:-src}" "${PYTHON:-python3}" -m qwen_edit_project.self_evolve.run_loop \
  --config configs/self_evolve/qwen_edit_2509_internal_cepr_trainable_proposer.yaml "$@"

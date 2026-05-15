#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <checkpoint> [extra args...]"
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"${PYTHON:-python3}" -m qwen_edit_project.train.launch_validate --config configs/train/lora_2509.yaml --mode lora --checkpoint "$1" "${@:2}"

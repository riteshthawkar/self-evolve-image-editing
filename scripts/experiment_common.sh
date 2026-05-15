#!/usr/bin/env bash

experiment_is_dry_run() {
  [[ "${EXPERIMENT_DRY_RUN:-0}" == "1" ]]
}

experiment_enable_dry_run() {
  export EXPERIMENT_DRY_RUN=1
}

experiment_repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

experiment_run() {
  printf '+'
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
  if experiment_is_dry_run; then
    return 0
  fi
  "$@"
}

experiment_latest_checkpoint() {
  local checkpoint_dir="$1"
  local python_bin="${PYTHON:-python3}"
  "$python_bin" - "$checkpoint_dir" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
if not root.exists():
    raise SystemExit(1)
candidates = sorted(root.rglob("*.safetensors"), key=lambda path: path.stat().st_mtime_ns)
if not candidates:
    raise SystemExit(1)
print(candidates[-1])
PY
}

experiment_require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    if experiment_is_dry_run; then
      echo "Dry run: skipping missing file check for $path" >&2
      return 0
    fi
    echo "Required file not found: $path" >&2
    exit 1
  fi
}

experiment_require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    if experiment_is_dry_run; then
      echo "Dry run: skipping missing directory check for $path" >&2
      return 0
    fi
    echo "Required directory not found: $path" >&2
    exit 1
  fi
}

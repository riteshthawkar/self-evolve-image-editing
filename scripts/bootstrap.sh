#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$ROOT/third_party"

clone_or_update() {
  local repo_url="$1"
  local target_dir="$2"
  if [[ -d "$target_dir/.git" ]]; then
    git -C "$target_dir" fetch --all --tags --prune
    git -C "$target_dir" pull --ff-only
  else
    git clone "$repo_url" "$target_dir"
  fi
}

clone_or_update https://github.com/modelscope/DiffSynth-Studio.git "$THIRD_PARTY/diffsynth-studio"
clone_or_update https://github.com/QwenLM/Qwen-Image.git "$THIRD_PARTY/qwen-image"
clone_or_update https://github.com/PKU-YuanGroup/Edit-R1.git "$THIRD_PARTY/edit-r1"
clone_or_update https://github.com/stepfun-ai/Step1X-Edit.git "$THIRD_PARTY/step1x-edit"
clone_or_update https://github.com/PKU-YuanGroup/ImgEdit.git "$THIRD_PARTY/imgedit"
clone_or_update https://github.com/djghosh13/geneval.git "$THIRD_PARTY/geneval"
clone_or_update https://github.com/TencentQQGYLab/ELLA.git "$THIRD_PARTY/ella"
clone_or_update https://github.com/OneIG-Bench/OneIG-Benchmark.git "$THIRD_PARTY/oneig-bench"

if [[ -f "$ROOT/patches/imgedit_env_key.patch" ]]; then
  if git -C "$THIRD_PARTY/imgedit" apply --check "$ROOT/patches/imgedit_env_key.patch" >/dev/null 2>&1; then
    git -C "$THIRD_PARTY/imgedit" apply "$ROOT/patches/imgedit_env_key.patch"
  fi
fi

if [[ -f "$ROOT/patches/oneig_diversity_fix.patch" ]]; then
  if git -C "$THIRD_PARTY/oneig-bench" apply --check "$ROOT/patches/oneig_diversity_fix.patch" >/dev/null 2>&1; then
    git -C "$THIRD_PARTY/oneig-bench" apply "$ROOT/patches/oneig_diversity_fix.patch"
  fi
fi

{
  echo "DiffSynth-Studio: $(git -C "$THIRD_PARTY/diffsynth-studio" rev-parse HEAD)"
  echo "Qwen-Image: $(git -C "$THIRD_PARTY/qwen-image" rev-parse HEAD)"
  echo "Edit-R1: $(git -C "$THIRD_PARTY/edit-r1" rev-parse HEAD)"
  echo "Step1X-Edit: $(git -C "$THIRD_PARTY/step1x-edit" rev-parse HEAD)"
  echo "ImgEdit: $(git -C "$THIRD_PARTY/imgedit" rev-parse HEAD)"
  echo "GenEval: $(git -C "$THIRD_PARTY/geneval" rev-parse HEAD)"
  echo "ELLA: $(git -C "$THIRD_PARTY/ella" rev-parse HEAD)"
  echo "OneIG-Benchmark: $(git -C "$THIRD_PARTY/oneig-bench" rev-parse HEAD)"
} > "$THIRD_PARTY/LOCKFILE.md"

echo "Bootstrap complete."

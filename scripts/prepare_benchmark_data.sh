#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  cat <<'EOF'
Usage: bash scripts/prepare_benchmark_data.sh [options]

Prepares benchmark data and scorer assets used by the evaluation wrappers.

Stages:
  repos       Clone/update third_party benchmark repos with scripts/bootstrap.sh
  gedit       Cache the Hugging Face GEdit-Bench dataset
  imgedit     Download ImgEdit Benchmark.tar and map Basic-Bench assets
  geneval     Download GenEval Mask2Former detector checkpoint
  dpgbench    Verify DPG-Bench prompt CSV from ELLA
  oneig       Verify OneIG-Bench prompt CSVs
  check       Print readiness checks for all benchmark assets
  all         Run repos, gedit, imgedit, geneval, dpgbench, oneig, check

Options:
  --stage STAGE                Stage to run. Default: all
  --skip-repos                 Do not run scripts/bootstrap.sh before data stages
  --hf-home PATH               Hugging Face cache root. Default: .cache/huggingface
  --download-root PATH         Download/extract root. Default: data/downloads/benchmarks
  --imgedit-hf-dataset ID      ImgEdit HF dataset ID. Default: sysuyy/ImgEdit
  --imgedit-materialize MODE   symlink or copy original_images. Default: symlink
  --geneval-detector-dir PATH  Default: data/external/geneval_detector
  -h, --help                   Show this message.

Environment:
  OPENAI_API_KEY is needed later for GEdit/ImgEdit scoring, not for downloading.
EOF
}

STAGE="all"
RUN_REPOS=1
HF_HOME_DEFAULT="$ROOT/.cache/huggingface"
DOWNLOAD_ROOT="data/downloads/benchmarks"
IMGEDIT_HF_DATASET="sysuyy/ImgEdit"
IMGEDIT_MATERIALIZE="symlink"
GENEVAL_DETECTOR_DIR="data/external/geneval_detector"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift ;;
    --skip-repos) RUN_REPOS=0 ;;
    --hf-home) HF_HOME_DEFAULT="$2"; shift ;;
    --download-root) DOWNLOAD_ROOT="$2"; shift ;;
    --imgedit-hf-dataset) IMGEDIT_HF_DATASET="$2"; shift ;;
    --imgedit-materialize) IMGEDIT_MATERIALIZE="$2"; shift ;;
    --geneval-detector-dir) GENEVAL_DETECTOR_DIR="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
  shift
done

export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_HOME" "$DOWNLOAD_ROOT" data/processed/benchmark data/external outputs/logs

run_repos() {
  if (( RUN_REPOS )); then
    bash scripts/bootstrap.sh
  fi
}

hf_download() {
  local repo_type="$1"
  local repo_id="$2"
  local local_dir="$3"
  shift 3
  mkdir -p "$local_dir"
  if command -v hf >/dev/null 2>&1; then
    hf download "$repo_id" --repo-type "$repo_type" --local-dir "$local_dir" "$@"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$repo_id" --repo-type "$repo_type" --local-dir "$local_dir" "$@"
  else
    "${PYTHON:-python3}" - "$repo_type" "$repo_id" "$local_dir" "$@" <<'PY'
from huggingface_hub import snapshot_download
import sys

repo_type, repo_id, local_dir, *args = sys.argv[1:]
allow_patterns = []
i = 0
while i < len(args):
    if args[i] == "--include":
        i += 1
        while i < len(args) and not args[i].startswith("--"):
            allow_patterns.append(args[i])
            i += 1
        continue
    i += 1
snapshot_download(
    repo_id=repo_id,
    repo_type=repo_type,
    local_dir=local_dir,
    allow_patterns=allow_patterns or None,
)
PY
  fi
}

download_url() {
  local url="$1"
  local target="$2"
  mkdir -p "$(dirname "$target")"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$url" -o "$target"
  elif command -v wget >/dev/null 2>&1; then
    wget -q "$url" -O "$target"
  else
    "${PYTHON:-python3}" - "$url" "$target" <<'PY'
from pathlib import Path
from urllib.request import urlopen
import sys

url, target = sys.argv[1:]
Path(target).write_bytes(urlopen(url).read())
PY
  fi
}

cache_gedit() {
  "${PYTHON:-python3}" - <<'PY'
from datasets import load_dataset

dataset = load_dataset("stepfun-ai/GEdit-Bench", split="train")
print(f"Cached GEdit-Bench train split with {len(dataset)} records.")
PY
}

find_first() {
  local root="$1"
  shift
  find "$root" "$@" -print -quit 2>/dev/null || true
}

copy_file_if_different() {
  local source_path="$1"
  local target_path="$2"
  mkdir -p "$(dirname "$target_path")"
  if [[ -e "$target_path" ]]; then
    if "${PYTHON:-python3}" - "$source_path" "$target_path" <<'PY'
import os
import sys

try:
    same = os.path.samefile(sys.argv[1], sys.argv[2])
except FileNotFoundError:
    same = False
sys.exit(0 if same else 1)
PY
    then
      return
    fi
  fi
  cp -f "$source_path" "$target_path"
}

safe_link_or_copy_dir() {
  local source_dir="$1"
  local target_dir="$2"
  local mode="$3"
  mkdir -p "$(dirname "$target_dir")"
  if [[ "$mode" == "copy" ]]; then
    mkdir -p "$target_dir"
    rsync -a "$source_dir"/ "$target_dir"/
    return
  fi
  if [[ "$mode" != "symlink" ]]; then
    echo "Unsupported materialize mode: $mode" >&2
    exit 1
  fi
  local link_target="$source_dir"
  if [[ "$source_dir" != /* ]]; then
    link_target="$ROOT/$source_dir"
  fi
  if [[ -L "$target_dir" ]]; then
    rm "$target_dir"
  elif [[ -d "$target_dir" && -n "$(find "$target_dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Keeping existing non-empty directory: $target_dir"
    return
  elif [[ -d "$target_dir" ]]; then
    rmdir "$target_dir"
  fi
  ln -s "$link_target" "$target_dir"
}

prepare_imgedit() {
  local download_dir="$DOWNLOAD_ROOT/imgedit"
  local extract_dir="$download_dir/extracted"
  mkdir -p "$download_dir" "$extract_dir"

  hf_download dataset "$IMGEDIT_HF_DATASET" "$download_dir" --include "Benchmark.tar"

  local tar_path
  tar_path="$(find_first "$download_dir" -type f -name "Benchmark.tar")"
  if [[ -z "$tar_path" ]]; then
    echo "ImgEdit Benchmark.tar was not found under $download_dir" >&2
    exit 1
  fi

  if [[ ! -f "$extract_dir/.extracted" || "$tar_path" -nt "$extract_dir/.extracted" ]]; then
    rm -rf "$extract_dir"
    mkdir -p "$extract_dir"
    tar -xf "$tar_path" -C "$extract_dir"
    date > "$extract_dir/.extracted"
  fi

  local basic_json prompts_json original_dir
  basic_json="$(find_first "$extract_dir" -type f \( -iname "basic_edit.json" -o -iname "Basic_Edit.json" \))"
  prompts_json="$(find_first "$extract_dir" -type f -path "*/Basic/prompts.json")"
  if [[ -z "$prompts_json" ]]; then
    prompts_json="$(find_first "$extract_dir" -type f -iname "prompts.json")"
  fi
  original_dir="$(find_first "$extract_dir" -type d \( -iname "original_images" -o -iname "origin_images" -o -iname "original_image" \))"
  if [[ -z "$original_dir" ]]; then
    # Current sysuyy/ImgEdit Benchmark.tar stores Basic-Bench source images as
    # Benchmark/singleturn/<category>/<image>.jpg, while basic_edit.json uses
    # relative ids like animal/000342021.jpg.
    original_dir="$(find_first "$extract_dir" -type d -path "*/Benchmark/singleturn")"
  fi

  mkdir -p data/processed/benchmark/imgedit third_party/imgedit/Benchmark/Basic
  if [[ -z "$basic_json" ]]; then
    basic_json="$download_dir/basic_edit.json"
    download_url \
      "https://raw.githubusercontent.com/PKU-YuanGroup/ImgEdit/main/Benchmark/Basic/basic_edit.json" \
      "$basic_json"
  fi
  if [[ -z "$prompts_json" ]]; then
    if [[ -f third_party/imgedit/Benchmark/Basic/prompts.json ]]; then
      prompts_json="third_party/imgedit/Benchmark/Basic/prompts.json"
    else
      prompts_json="$download_dir/prompts.json"
      download_url \
        "https://raw.githubusercontent.com/PKU-YuanGroup/ImgEdit/main/Benchmark/Basic/prompts.json" \
        "$prompts_json"
    fi
  fi

  if [[ -z "$basic_json" || -z "$prompts_json" || -z "$original_dir" ]]; then
    echo "Could not locate all ImgEdit Basic-Bench assets after extraction." >&2
    echo "basic_edit.json: ${basic_json:-MISSING}" >&2
    echo "prompts.json: ${prompts_json:-MISSING}" >&2
    echo "original_images: ${original_dir:-MISSING}" >&2
    echo "Inspect with: find $extract_dir -maxdepth 5 -type f | sed -n '1,120p'" >&2
    exit 1
  fi

  copy_file_if_different "$basic_json" data/processed/benchmark/imgedit/basic_edit.json
  copy_file_if_different "$prompts_json" third_party/imgedit/Benchmark/Basic/prompts.json
  safe_link_or_copy_dir "$original_dir" data/processed/benchmark/imgedit/original_images "$IMGEDIT_MATERIALIZE"
  echo "Prepared ImgEdit Basic-Bench assets."
}

prepare_geneval() {
  local detector_dir="$GENEVAL_DETECTOR_DIR"
  mkdir -p "$detector_dir"
  if [[ -x third_party/geneval/evaluation/download_models.sh ]]; then
    bash third_party/geneval/evaluation/download_models.sh "$detector_dir"
  else
    wget -c \
      https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth \
      -O "$detector_dir/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
  fi
  test -f third_party/geneval/prompts/evaluation_metadata.jsonl
  echo "Prepared GenEval prompts and detector checkpoint."
}

prepare_dpgbench() {
  test -f third_party/ella/dpg_bench/dpg_bench.csv
  echo "DPG-Bench prompt CSV is present."
}

prepare_oneig() {
  test -f third_party/oneig-bench/OneIG-Bench.csv
  test -f third_party/oneig-bench/OneIG-Bench-ZH.csv
  echo "OneIG-Bench prompt CSVs are present."
}

check_path() {
  local label="$1"
  local path="$2"
  if [[ -e "$path" ]]; then
    echo "[ok]      $label: $path"
  else
    echo "[missing] $label: $path"
  fi
}

check_all() {
  check_path "GEdit repo scorer" "third_party/step1x-edit/GEdit-Bench/run_gedit_score.py"
  check_path "ImgEdit basic_edit" "data/processed/benchmark/imgedit/basic_edit.json"
  check_path "ImgEdit original images" "data/processed/benchmark/imgedit/original_images"
  check_path "ImgEdit prompts" "third_party/imgedit/Benchmark/Basic/prompts.json"
  check_path "GenEval prompts" "third_party/geneval/prompts/evaluation_metadata.jsonl"
  check_path "GenEval detector" "$GENEVAL_DETECTOR_DIR/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
  check_path "DPG-Bench prompts" "third_party/ella/dpg_bench/dpg_bench.csv"
  check_path "OneIG EN CSV" "third_party/oneig-bench/OneIG-Bench.csv"
  check_path "OneIG ZH CSV" "third_party/oneig-bench/OneIG-Bench-ZH.csv"
  if [[ -f secret.env || -n "${OPENAI_API_KEY:-}" ]]; then
    echo "[ok]      OpenAI key source for GPT-based edit scoring"
  else
    echo "[missing] OpenAI key source: create secret.env or export OPENAI_API_KEY before GEdit/ImgEdit scoring"
  fi
}

case "$STAGE" in
  repos) run_repos ;;
  gedit) run_repos; cache_gedit ;;
  imgedit) run_repos; prepare_imgedit ;;
  geneval) run_repos; prepare_geneval ;;
  dpgbench) run_repos; prepare_dpgbench ;;
  oneig) run_repos; prepare_oneig ;;
  check) check_all ;;
  all)
    run_repos
    cache_gedit
    prepare_imgedit
    prepare_geneval
    prepare_dpgbench
    prepare_oneig
    check_all
    ;;
  *)
    echo "Unsupported stage: $STAGE" >&2
    usage >&2
    exit 1
    ;;
esac

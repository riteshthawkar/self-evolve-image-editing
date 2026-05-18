# Benchmark Data Setup

Use this before running the evaluation exporters/scorers. It prepares the scorer
repos and the benchmark assets expected by the repo configs.

## One Command

```bash
cd ~/self-evolve-image-editing
conda activate /share_6/users/ritesh_thawkar/condaenvs/qedit

export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"

bash scripts/prepare_benchmark_data.sh --stage all
```

This runs:

- `scripts/bootstrap.sh` for scorer repos
- Hugging Face cache warmup for `stepfun-ai/GEdit-Bench`
- `sysuyy/ImgEdit` download of `Benchmark.tar`
- ImgEdit Basic-Bench path mapping
- GenEval Mask2Former detector download
- DPG-Bench prompt CSV check
- OneIG-Bench prompt CSV check

## Check Only

```bash
bash scripts/prepare_benchmark_data.sh --stage check
```

Expected required paths:

```text
third_party/step1x-edit/GEdit-Bench/run_gedit_score.py
data/processed/benchmark/imgedit/basic_edit.json
data/processed/benchmark/imgedit/original_images/
third_party/imgedit/Benchmark/Basic/prompts.json
third_party/geneval/prompts/evaluation_metadata.jsonl
data/external/geneval_detector/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth
third_party/ella/dpg_bench/dpg_bench.csv
third_party/oneig-bench/OneIG-Bench.csv
third_party/oneig-bench/OneIG-Bench-ZH.csv
```

## Individual Stages

GEdit:

```bash
bash scripts/prepare_benchmark_data.sh --stage gedit
```

ImgEdit:

```bash
bash scripts/prepare_benchmark_data.sh --stage imgedit
```

GenEval:

```bash
bash scripts/prepare_benchmark_data.sh --stage geneval
```

Generation benchmark prompt checks:

```bash
bash scripts/prepare_benchmark_data.sh --stage dpgbench
bash scripts/prepare_benchmark_data.sh --stage oneig
```

## Notes

- GEdit and ImgEdit scoring require an OpenAI key. Create `secret.env` or export
  `OPENAI_API_KEY` before scoring.
- GenEval scoring also requires a compatible MMDetection environment. This
  script downloads the detector checkpoint but does not install MMDetection.
- DPG-Bench and OneIG-Bench scorers may download additional model weights when
  scoring starts.
- ImgEdit uses only the benchmark subset from `Benchmark.tar`, not the full
  1.2M training dataset.
